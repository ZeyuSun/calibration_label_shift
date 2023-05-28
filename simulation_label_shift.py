from itertools import product
from multiprocessing import Pool
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy
from uncertainties import ufloat

from calibration import HistogramCalibrator, LabelShiftCalibrator, CompositeCalibrator, OracleCalibrator
from simulation_calibration import evaluate
from utils import dlogit, logit, sigmoid


class GaussianMixture:
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

    def sample(self, n):
        y = np.random.binomial(1, self.p, size=n)
        n1 = np.sum(y)
        x = np.zeros(n)
        x[y == 0] = np.random.normal(-self.mu, 1, size=n-n1)
        x[y == 1] = np.random.normal(self.mu, 1, size=n1)
        z = self.f(x)
        return z, y


class LabelShiftSimulation:
    def __init__(self):
        self.ds = GaussianMixture(p=0.5)  # source data
        self.ns = 1000  # source sample size

    def run_single(self, nt, pt, i, return_calibrators=False):
        np.random.seed(i)
        results = []
        params = {'nt': nt, 'pt': pt, 'i': i}
        dt = GaussianMixture(p=pt)  # target data

        # Source: only calibrate on source data
        zs, ys = self.ds.sample(self.ns)
        Bs = int(self.ns ** (1 / 3))
        hs = HistogramCalibrator(n_bins=Bs, strategy='quantile').fit(zs, ys)
        metrics = evaluate(hs, dt.py_given_z, dt.pz)
        results.extend([params | {'method': 'source', 'metric': k, 'value': v}
                        for k, v in metrics.items()])

        # Target: only calibrate on target data
        zt, yt = dt.sample(nt)
        Bt = int(nt ** (1 / 3))
        ht = HistogramCalibrator(n_bins=Bt, strategy='quantile').fit(zt, yt)
        metrics = evaluate(ht, dt.py_given_z, dt.pz)
        results.extend([params | {'method': 'target', 'metric': k, 'value': v}
                        for k, v in metrics.items()])

        # Label shift: only perform label shift correction
        hls = LabelShiftCalibrator(np.mean(ys), np.mean(yt))
        metrics = evaluate(hls, dt.py_given_z, dt.pz)
        results.extend([params | {'method': 'label shift', 'metric': k, 'value': v}
                        for k, v in metrics.items()])

        # Composite: calibrate on source data first, then perform label shift correction
        hc = CompositeCalibrator([hs, hls])
        metrics = evaluate(hc, dt.py_given_z, dt.pz)
        results.extend([params | {'method': 'composite', 'metric': k, 'value': v}
                        for k, v in metrics.items()])

        if return_calibrators:
            calibrators = {
                'source': hs,
                'target': ht,
                'label shift': hls,
                'composite': hc,
            }
            results = (results, calibrators)
        return results

    def run(self, nt_list=None, pt_list=None, i_list=None):
        """
        Args:
            nt_list: list of target sample sizes
            pt_list: list of target label proportions
            i_list: list of random seeds
        """
        if nt_list is None:
            nt_list = [100]
        if pt_list is None:
            pt_list = [0.1]
        if i_list is None:
            i_list = [0, 1, 2]
        prod = list(product(nt_list, pt_list, i_list))
        # results = [self.run_single(*args) for args in prod]
        with Pool() as pool:
            results = pool.starmap(self.run_single, prod)
        results = [item for sublist in results for item in sublist]
        df = pd.DataFrame(results)
        return df

    def tabulate(self, df, to_latex=False):
        """
        Args:
            df: output of self.run()
            to_latex: return a latex string if True; otherwise return a dataframe
        """
        nt, pt = 100, 0.1
        df = (
            df
            .loc[(df['nt'] == nt) & (df['pt'] == pt), :]
            .drop(columns=['nt', 'pt', 'i'])
            .groupby(['method', 'metric'])
            .agg(lambda s: ufloat(s.mean(), s.std().clip(min=0.0001))
                if s.mean() > 0 else 0)
            .reset_index()
        )

        df_wide = (
            df
            # df.pivot <=> df.set_index(['method', 'metric'])['value'].unstack(-1)
            .pivot(index=['method'], columns=['metric'], values='value')
            .loc[['source', 'target', 'label shift', 'composite'], ['cal', 'sha', 'risk', 'bs']]
            .rename_axis(index=str.capitalize, columns='')
        )

        if not to_latex:
            return df_wide

        df_wide = (
            df_wide
            .rename(columns={'cal': '$\REL$', 'sha': '$\GRP$', 'risk': '$R$', 'bs': 'MSE'}, index={'source': '\Source{}', 'target': r'\Target{}', 'label shift': r'\Labelshift{}', 'composite': r'\Composite{}'})
            .reset_index()
        )

        df_wide.insert(1, 'Estimator', ['$\hat{h}_P$', '$\hat{h}_Q^{\\textrm{target}}$', '$\hat{g}$', '$\hat{h}_{Q}$'])

        props = {
            'latex': 'bfseries: ;',
            'html': 'background-color: yellow',
        }
        style = 'latex'
        df_str = (
            df_wide.style
            .hide(axis='index')
            .highlight_min(axis=0, props=props[style], subset=[c for c in df_wide if c not in ['Method', 'Estimator']])
            .to_latex(hrules=True, column_format='c' * df_wide.shape[1])
            .replace('+/-', '$ \pm $')
        )
        return df_str

    def plot(self):
        nt, pt = 1000, 0.1
        self.ns = 100000
        dt = GaussianMixture(p=pt)
        _, calibrators = self.run_single(nt=nt, pt=pt, i=0, return_calibrators=True)
        calibrators['oracle'] = OracleCalibrator(dt.py_given_z)

        # calibration curves
        symbols = {
            'oracle': '$h^*_Q$',
            'source': '$\hat{h}_P$',
            'target': '$\hat{h}_Q^{target}$',  # '$\hat{h}_Q^{\\textrm{target}}$',
            'label shift': '$\hat{g}$',
            'composite': '$\hat{h}_{Q}$',
        }
        for key, symbol in symbols.items():
            if key == 'composite':
                kwargs = {
                    'set_layout': True,
                    'lw': 4,
                }
            else:
                kwargs = {'lw': 2}
            calibrators[key].plot(label=symbol, **kwargs)
        plt.savefig('simulation_label_shift.pdf', bbox_inches='tight')

        # density p_Q(z)
        plt.figure(figsize=(3.5, 3.5))
        zz = np.linspace(0, 1, 1000)
        plt.plot(zz, dt.pz(zz), c='k', label='$p_Q(z)$')
        plt.grid()
        plt.legend()
        plt.xlabel('Predicted Probability $z$')
        plt.ylabel('Density')
