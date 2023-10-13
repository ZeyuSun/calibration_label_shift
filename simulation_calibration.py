import itertools
# from multiprocessing import Pool
from multiprocess import Pool
import numpy as np
import pandas as pd
import scipy
from matplotlib import colors, pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm

from calibration import OracleCalibrator, BinnedOracleCalibrator, HistogramCalibrator, PlattCalibrator, ScalingBinningCalibrator
from utils import expectation, sigmoid, logit, dlogit
from bounds import CAL, SHA, RISK


class CalibrationSimulationBase:
    """Calibration simulation base class.
    Subclass this class and implement the following methods:
    - generate_data
    - py_given_z
    - pz
    """

    def pz(self, z):  # p(z)
        raise NotImplementedError

    def py_given_z(self, z):  # P[Y=1 | Z=z]
        raise NotImplementedError

    def generate_data(self, n):  # n: sample size
        raise NotImplementedError

    def run_single(self, n, B, i):
        np.random.seed(i)
        z, y = self.generate_data(n)
        calibrator = HistogramCalibrator(n_bins=B, strategy='quantile')
        calibrator.fit(z, y)
        metrics = evaluate(calibrator, self.py_given_z, self.pz)
        metrics.update({'n': n, 'B': B, 'i': i})
        return metrics

    def run(self, n_list=None, B_list=None, i_list=None):
        if n_list is None:
            n_list = np.logspace(2, 7, 7, dtype=int)
        if B_list is None:
            B_list = np.logspace(0.8, 3, 10, dtype=int)
        if i_list is None:
            i_list = np.arange(10)
        prod = itertools.product(n_list, B_list, i_list)
        # results = [self.run_single(*args) for args in prod]
        with Pool() as p:
            results = p.starmap(self.run_single, prod)
        df = pd.DataFrame(results)
        return df

    def plot(self, df_raw):
        n_list = np.unique(df_raw['n'])
        B_list = np.unique(df_raw['B'])
        delta = 0.1  # bounds fail with probability at most delta
        self.K = getattr(self, 'K', None)  # smoothness constant
        df = df_raw.groupby(['n', 'B']).median().reset_index()
        df_lower = df_raw.groupby(['n', 'B']).quantile(q=delta)
        df_upper = df_raw.groupby(['n', 'B']).quantile(q=1-delta)
        df = df.join(df_lower, on=['n', 'B'], rsuffix='_lower')
        df = df.join(df_upper, on=['n', 'B'], rsuffix='_upper')
        figsize = (2.5, 2.5)

        # cal vs. B
        plt.figure(figsize=figsize)
        slopes_weights = []
        for i, n in enumerate(n_list[::2]):
            subdf = df.loc[(df['n'] == n) & (df['B'] <= n / 2)]
            _B_list = subdf['B'].to_numpy()
            B_grid = np.linspace(_B_list[0], _B_list[-1], 100)

            reg = LinearRegression().fit(np.log(_B_list).reshape(-1, 1), np.log(subdf['cal']))
            slopes_weights.append([reg.coef_[0], len(_B_list)])

            plt.plot(_B_list, subdf['cal'], c=f'C{i}', label=f'n={n:.1e}')
            plt.fill_between(_B_list, subdf['cal_lower'], subdf['cal_upper'], color=f'C{i}', alpha=0.2)
            plt.plot(B_grid, CAL(B_grid, n, delta), c=f'C{i}', ls='--')
        slopes, weights = zip(*slopes_weights)
        print('cal vs. B', np.average(slopes, weights=weights))
        plt.xscale('log'); plt.yscale('log'); plt.grid()
        plt.xlabel('B'); plt.ylabel('Calibration Risk'); plt.legend(framealpha=0.3)
        plt.savefig('cal_vs_B.pdf', bbox_inches='tight')

        # cal vs. n
        plt.figure(figsize=figsize)
        slopes_weights = []
        for i, B in enumerate(B_list[::3]):
            subdf = df.loc[(df['B'] == B) & (df['n'] >= 2 * B)]
            _n_list = subdf['n'].to_numpy()
            n_grid = np.linspace(_n_list[0], _n_list[-1], 100)

            reg = LinearRegression().fit(np.log(_n_list).reshape(-1, 1), np.log(subdf['cal']))
            slopes_weights.append([reg.coef_[0], len(_n_list)])

            plt.plot(_n_list, subdf['cal'], c=f'C{i}', label=f'B={B}')
            plt.fill_between(_n_list, subdf['cal_lower'], subdf['cal_upper'], color=f'C{i}', alpha=0.2)
            plt.plot(n_grid, CAL(B, n_grid, delta), c=f'C{i}', ls='--')
        slopes, weights = zip(*slopes_weights)
        print('cal vs. n', np.average(slopes, weights=weights))
        plt.xscale('log'); plt.yscale('log'); plt.grid()
        plt.xlabel('n'); plt.ylabel('Calibration Risk'); plt.legend(framealpha=0.3)
        plt.savefig('cal_vs_n.pdf', bbox_inches='tight')

        # sha vs. B (all n)
        plt.figure(figsize=figsize)
        for i, n in enumerate(n_list[::2]):
            subdf = df.loc[df['n'] == n]
            _B_list = subdf['B'].to_numpy()
            B_grid = np.linspace(_B_list[0], _B_list[-1], 100)
            plt.plot(_B_list, subdf['sha'], c=f'C{i}', label=f'n={n:.1e}')
            plt.plot(B_grid, SHA(B_grid, self.K), c=f'C{i}', ls='--')
        plt.xscale('log'); plt.yscale('log'); plt.grid()
        plt.xlabel('B'); plt.ylabel('Sharpness Risk'); plt.legend(framealpha=0.3)
        plt.savefig('sha_vs_B_all_n.pdf', bbox_inches='tight')

        # sha vs. B
        plt.figure(figsize=figsize)
        n = n_list[-1]
        subdf = df.loc[df['n'] == n]
        _B_list = subdf['B'].to_numpy()
        B_grid = np.linspace(_B_list[0], _B_list[-1], 100)
        reg = LinearRegression().fit(np.log(_B_list).reshape(-1, 1), np.log(subdf['sha']))
        print('sha vs. B', reg.coef_[0])
        plt.plot(_B_list, subdf['sha'], 'o-', c='C0', label='$R^{\mathrm{sha}}$')
        plt.fill_between(_B_list, subdf['sha_lower'], subdf['sha_upper'], color='C0', alpha=0.2)
        plt.plot(B_grid, SHA(B_grid, self.K), c='C0', ls='--', label='bound')
        plt.xscale('log'); plt.yscale('log'); plt.grid()
        plt.xlabel('B'); plt.ylabel('Sharpness Risk'); plt.legend(framealpha=0.3)
        plt.savefig('sha_vs_B.pdf', bbox_inches='tight')

        # risk vs. B
        plt.figure(figsize=figsize)
        for i, n in enumerate(n_list[::2]):
            subdf = df.loc[(df['n'] == n) & (df['B'] <= n / 2)]
            _B_list = subdf['B'].to_numpy()
            B_grid = np.linspace(_B_list[0], _B_list[-1], 100)
            plt.plot(_B_list, subdf['risk'], c=f'C{i}', label=f'n={n:.1e}')
            plt.fill_between(_B_list, subdf['risk_lower'], subdf['risk_upper'], color=f'C{i}', alpha=0.2)
            plt.plot(B_grid, RISK(B_grid, n, delta, self.K), c=f'C{i}', ls='--')
        plt.xscale('log'); plt.yscale('log'); plt.grid()
        plt.xlabel('B'); plt.ylabel('Risk'); plt.legend(framealpha=0.3)
        plt.savefig('risk_vs_B.pdf', bbox_inches='tight')

        # risk vs. n
        plt.figure(figsize=figsize)
        for i, B in enumerate(B_list[::3]):
            subdf = df.loc[(df['B'] == B) & (df['n'] >= 2 * B)]
            _n_list = subdf['n'].to_numpy()
            n_grid = np.linspace(_n_list[0], _n_list[-1], 100)
            plt.plot(_n_list, subdf['risk'], c=f'C{i}', label=f'B={B}')
            plt.fill_between(_n_list, subdf['risk_lower'], subdf['risk_upper'], color=f'C{i}', alpha=0.2)
            plt.plot(n_grid, RISK(B, n_grid, delta, self.K), c=f'C{i}', ls='--')
        plt.xscale('log'); plt.yscale('log'); plt.grid()
        plt.xlabel('n'); plt.ylabel('Risk'); plt.legend(framealpha=0.3)
        plt.savefig('risk_vs_n.pdf', bbox_inches='tight')

        # optimal B vs. n
        plt.figure(figsize=figsize)
        idx_list = np.zeros(len(n_list), dtype=int)  # index of empirically optimal B in B_list
        idx_grid = np.zeros(len(n_list), dtype=int)  # index of theoretically optimal B in B_grid
        B_grid = np.logspace(np.log10(B_list[0]), np.log10(10 * B_list[-1]), 100)
        for i, n in enumerate(n_list):
            subdf = df.loc[df['n'] == n]
            risks = subdf['risk'].to_numpy()
            idx_list[i] = np.argmin(risks)

            _B_grid = B_grid[B_grid <= n / 2]
            bounds = RISK(_B_grid, n, delta, self.K)
            idx_grid[i] = np.nanargmin(bounds)  # argmin returns first nan index

        # optimal B vs. n (linear scale)
        plt.figure(figsize=figsize)
        plt.plot(n_list, B_list[idx_list], 'x-', label='experiment')
        plt.plot(n_list, B_grid[idx_grid], 'o--', label='theory')
        plt.xscale('log'); plt.grid()
        plt.xlabel('n'); plt.ylabel('Optimal B'); plt.legend(framealpha=0.3)
        plt.savefig('opt_B_linear.pdf', bbox_inches='tight')

        # optimal B vs. n (log scale)
        plt.figure(figsize=figsize)
        plt.plot(n_list, B_list[idx_list], 'x-', label='experiment')
        plt.plot(n_list, B_grid[idx_grid], 'o--', label='theory')
        plt.plot(n_list, n_list ** (1/3), ':', label='$n^{1/3}$')
        plt.xscale('log'); plt.yscale('log'); plt.grid()
        plt.xlabel('n'); plt.ylabel('Optimal B'); plt.legend(framealpha=0.3)
        plt.savefig('opt_B_log.pdf', bbox_inches='tight')

        # optimal B vs. n (risk surface)
        fig, ax = plt.subplots(figsize=figsize)
        risks_list = [df.loc[df['n'] == n, 'risk'].to_list() for n in n_list]
        im = ax.imshow(np.array(risks_list), origin='lower', cmap='coolwarm', norm=colors.LogNorm())
        ax.set_xticks(np.arange(len(B_list))[::2], B_list[::2])
        ax.set_yticks(np.arange(len(n_list))[::2], [f'{n:.1e}' for n in n_list[::2]])
        ax.scatter(idx_list, np.arange(len(n_list)), marker='o', c='k')
        fig.colorbar(im, label='Risk', orientation='horizontal', location='top', pad=0.03)
        plt.xlabel('B'); plt.ylabel('n')
        plt.savefig('opt_B_risk_surface.pdf', bbox_inches='tight')

    def run_calibrators(self, n=1000, B=None, seed=None, plot=True):
        np.random.seed(seed)
        if B is None:
            B = round(1 * (n ** (1/3)))
        z, y = self.generate_data(n)

        scaling_binning = ScalingBinningCalibrator(n_bins=B).fit(z, y)
        platt = PlattCalibrator().fit(z, y)
        calibrators = {
            'Optimal': OracleCalibrator(self.py_given_z),
            'Platt': platt,
            'Hybrid': scaling_binning,
            'UWB': HistogramCalibrator(n_bins=B, strategy='uniform').fit(z, y),
            'UMB': HistogramCalibrator(n_bins=B, strategy='quantile').fit(z, y),
            # 'Hybrid-oracle': BinnedOracleCalibrator(scaling_binning.bins, platt, self.pz)
            # 'Isotonic': IsotonicCalibrator().fit(z, y),
        }

        # plot
        if plot:
            calibrators['Optimal'].plot(label='$h^*$', ls='--', lw=1.5)
            calibrators['Platt'].plot(label='Platt', lw=1.5)
            calibrators['Hybrid'].plot(label='Hybrid', lw=1.5)
            calibrators['UWB'].plot(label='UWB', lw=1.5)
            calibrators['UMB'].plot(label='UMB', lw=1.5, set_layout=True)
            # plt.savefig('compare_logistic.pdf', bbox_inches='tight') # this is in base class; save outside

        # table
        metrics = []
        for name, calibrator in tqdm(calibrators.items()):
            m = evaluate(calibrator, self.py_given_z, self.pz)
            for k, v in m.items():
                metrics.append({
                    'Calibrator': name,
                    'Metric': k,
                    'Value': v,
                })
        df = pd.DataFrame(metrics)

        columns = {'cal': '$\REL$', 'sha': '$\GRP$', 'risk': '$R$', 'bs': 'MSE'}
        indices = ['Platt', 'Hybrid', 'UWB', 'UMB'] # slice(None)  # ['Platt', 'Hybrid', 'UMB']
        df_wide = (
            df
            .set_index(['Calibrator', 'Metric'])['Value'].unstack(-1)
            .loc[indices, columns.keys()]
            .reindex(indices)
            .rename(columns=columns)
            .rename_axis(None, axis=0)
        )
        df_str = (df_wide.style
            .highlight_min(axis=0, props='bfseries: ;', subset=[c for c in df_wide if c not in ['Method', 'Estimator']])
            .format(precision=6)
            .to_latex(hrules=True, column_format='c' * (df_wide.shape[1] + 1))
        )
        return df_wide, df_str, metrics

    def run_calibrators_asymp(self, n_list=None, B_list=None, i_list=None):
        if n_list is None:
            n_list = [5000]  # np.logspace(3, 7, 2, dtype=int)
        if B_list is None:
            B_list = np.logspace(1, 3, 3, dtype=int)  #np.logspace(0.8, 3, 2, dtype=int)
        if i_list is None:
            i_list = np.arange(1)
        prod = list(itertools.product(n_list, B_list, i_list))
        # prod = [(n, B, i) for n, B, i in prod if n >= 2 * B] # IndexError: list index out of range
        def f(n, B, i):
            _, _, metrics = self.run_calibrators(n, B, i, plot=False)
            for m in metrics:
                m.update({'n': n, 'B': B, 'i': i})
            return metrics

        # results = []
        # for args in tqdm(prod, total=len(prod)):
        #     results.extend(f(*args))
        # results = [f(*args) for args in prod]
        with Pool() as p:
            results = p.starmap(f, prod)
        results = [m for metrics in results for m in metrics]
        df = pd.DataFrame(results)
        return df

    def plot_asymp(self, df):
        df = df.loc[
            (df['Calibrator'].isin(['Platt', 'Hybrid', 'UMB', 'UWB'])) &
            (df['Metric'].isin(['cal', 'sha', 'risk']))
        ]
        fig = px.line(df, x='B', y='Value', color='Calibrator', facet_row='Metric', log_x=True, log_y=True, line_dash='Calibrator', symbol='Calibrator')
        fig.update_yaxes(matches=None, showticklabels=True)
        fig.update_layout(
            template='simple_white+gridon',
            width=250,
            height=450,
            legend=dict(
                title_text="",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=go.layout.Margin(
                l=0, #left margin
                r=0, #right margin
                b=0, #bottom margin
                t=0  #top margin
            ),
        )
        for i, facet_name in enumerate(['R', 'R<sup>sha</sup>', 'R<sup>cal</sup>']):
            # plotly latex rendering is ugly #[r"$R$", r"$R^{\text{sha}}$", r"$R^{\text{cal}}$"]):
            fig.update_yaxes(title_text=facet_name, row=i+1, col=1)
        fig.for_each_annotation(lambda a: a.update(text=''))
        return fig

    def tabulate(self, df):
        columns = {'cal': '$\REL$', 'sha': '$\GRP$', 'risk': '$R$', 'bs': 'MSE'}
        indices = ['Platt', 'Hybrid', 'UWB', 'UMB']

        df_agg = (
            df
            .groupby(['Calibrator', 'Metric'])['Value']
            .agg(lambda s: np.percentile(s, 90))
            # .agg(lambda s: ufloat(s.mean(), s.std().clip(min=0.0001)) if s.mean() > 0 else 0)
        )
        df_wide = (
            df_agg
            .unstack(-1)
            .loc[indices, columns.keys()]
            .reindex(indices)
            .rename(columns=columns)
            .rename_axis(index=None, columns={'Metric': 'Metric ($\\times 10^{-3}$)'})
            * 10 ** 3
        )
        df_str = (
            df_wide.style
            .highlight_min(axis=0, props='bfseries: ;')
            .format(precision=3)
            .to_latex(hrules=True, column_format='c' * (df_wide.shape[1] + 1))
        )
        return df_wide, df_str


class GaussianMixtureSimulation(CalibrationSimulationBase):
    """Simulation 1: Gaussian mixture data and sigmoid classifier.
    Y ~ Bernoulli(p)
    X|Y=0 ~ N(-mu, 1)
    X|Y=1 ~ N(mu, 1)
    Z = f(X) = sigmoid(X)
    """

    def __init__(self, mu=2, p=0.5):
        self.mu = mu
        self.p = p
        self.f = sigmoid
        self.f_inv = logit
        self.f_inv_der = dlogit
        self.K = 20  # estimated by simulation

    def px(self, x):
        likelihood_0 = scipy.stats.norm.pdf(x, loc=-self.mu, scale=1)
        likelihood_1 = scipy.stats.norm.pdf(x, loc=self.mu, scale=1)
        return self.p * likelihood_1 + (1 - self.p) * likelihood_0

    def pz(self, z):
        """
        z = f(x)
        p(z) = p(x) dx/dz
        dx/dz: jacobian
        """
        x = self.f_inv(z)
        px = self.px(x)
        jac = self.f_inv_der(z)
        return px * jac

    def py_given_z(self, z):
        x = self.f_inv(z)
        logodds = 2 * self.mu * x + np.log(self.p / (1 - self.p))
        return sigmoid(logodds)

    def generate_data(self, n):
        y = np.random.binomial(1, self.p, size=n)
        n1 = np.sum(y)
        x = np.zeros(n)
        x[y == 0] = np.random.normal(-self.mu, 1, size=n-n1)
        x[y == 1] = np.random.normal(self.mu, 1, size=n1)
        z = self.f(x)
        return z, y


class BetaCalibrationSimulation(CalibrationSimulationBase):
    """Simulation 2:
    Z ~ Uniform[0, 1]
    Y | Z ~ Bernoulli with p(y=1|z) = 1 / (1 + 1 / (e^c z^a / (1-z)^b))

    p(y=1|z) is Beta calibration [Kull'17](https://doi.org/10.1214/17-EJS1338SI).
    We set Z ~ Uniform instead of Beta mixture for simplicity.
    """
    # def __init__(self, a=0.2, b=5, m=0.5):
    def __init__(self, a=1, b=1, m=None, c=None):
        self.a = a
        self.b = b
        assert m is None and c is not None or m is not None and c is None
        if c is None:
            self.c = b * np.log(1 - m) - a * np.log(m)
        else:
            self.c = c
        self.K = self.get_K()

    def pz(self, z):
        return 1

    def py_given_z(self, z):
        # logodds = self.c +  self.a * np.log(z) - self.b * np.log(1 - z)
        # return sigmoid(logodds)
        return 1 / (1 + 1 / (np.exp(self.c) * z ** self.a / (1 - z) ** self.b))

    def generate_data(self, n):
        z = np.random.uniform(size=n)
        y = np.random.binomial(1, self.py_given_z(z))
        return z, y

    def get_K(self):
        mu = self.py_given_z

        def dmu(z):
            s = mu(z)
            t1 = self.a / z + self.b / (1 - z)
            return s * (1 - s) * t1

        def ddmu(z):
            s = mu(z)
            t1 = self.a / z + self.b / (1 - z)
            t2 = - self.a / z ** 2 + self.b / (1 - z) ** 2
            return (1 - 2 * s) * t1 ** 2 + t2

        def neg(f):
            def neg_f(z):
                return -f(z)
            return neg_f

        res = scipy.optimize.minimize_scalar(neg(dmu), bounds=(0, 1), method='bounded')
        # x0 = 0.5
        # res = scipy.optimize.minimize(neg(dmu), x0, jac=neg(ddmu)) # not bounded, ddmu can be small
        return dmu(res.x)


def evaluate(calibrator, py_given_z, pz):
    """Evaluate population metrics of a calibrator."""
    oracle = OracleCalibrator(py_given_z)
    if hasattr(calibrator, 'bins'):
        binned_oracle = BinnedOracleCalibrator(calibrator.bins, py_given_z, pz)
    elif hasattr(calibrator, 'z_'):
        binned_oracle = BinnedOracleCalibrator(calibrator.z_, py_given_z, pz)
    else:  # no binning, no sharpness loss
        binned_oracle = oracle
    results = {
        'cal': expectation(lambda z: (calibrator(z) - binned_oracle(z)) ** 2, pz),  # calibration risk; reliability
        'sha': expectation(lambda z: (binned_oracle(z) - oracle(z)) ** 2, pz),  # sharpness risk; grouping risk
        # 'risk': expectation(lambda z: (calibrator(z) - oracle(z)) ** 2, pz),  # total risk
        'ref': expectation(lambda z: (oracle(z) * (1-oracle(z))) ** 2, pz),  # refinement
    }
    results['risk'] = results['cal'] + results['sha']  # total risk
    results['bs'] = results['risk'] + results['ref']  # Brier score; mean squared error
    return results