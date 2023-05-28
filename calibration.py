import warnings
import numpy as np
from matplotlib import pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from utils import expectation, logit, interpolate_nan


class BaseCalibrator:
    def fit(self, z, y):
        return self

    def predict(self, z):
        pass

    def __call__(self, z):
        return self.predict(z)

    def plot(self, *args, ax=None, set_layout=None, **kwargs):
        if ax is None:
            if plt.get_fignums():
                ax = plt.gca()
            else:
                _, ax = plt.subplots(figsize=(3.5, 3.5))

        if hasattr(self, 'bins'):
            ax.stairs(self.bin_mean, *args, edges=self.bins, baseline=None, **kwargs)
            ax.set_ylim(bottom=-0.05)
        elif hasattr(self, 'z_'):
            ax.plot(self.z_, self.y_, *args, **kwargs)
        else:
            zz = np.linspace(0, 1, 1000)
            ax.plot(zz, self.predict(zz), *args, **kwargs)

        if set_layout:
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            plt.xlabel('Predicted Probability')
            plt.ylabel('Empirical Frequency')
            ax.set_aspect('equal', 'box')  # plt.axis('scaled'): ylim bottom change back to 0
            plt.grid()
            plt.legend(framealpha=0.4)
        return ax


class CompositeCalibrator(BaseCalibrator):
    def __init__(self, calibrators):
        self.calibrators = calibrators
        if len(calibrators) == 2 and isinstance(calibrators[0], HistogramCalibrator):
            self.bins = self.calibrators[0].bins
            self.bin_mean = self.calibrators[1](self.calibrators[0].bin_mean)

    def predict(self, z):
        for calibrator in self.calibrators:
            z = calibrator.predict(z)
        return z


class OracleCalibrator(BaseCalibrator):
    def __init__(self, py_given_z):
        self.py_given_z = py_given_z

    def predict(self, z):
        return self.py_given_z(z)


class BinnedOracleCalibrator(BaseCalibrator):
    def __init__(self, bins, py_given_z, pz):
        self.bins = bins
        self.bin_mean = [
            expectation(py_given_z, pz, z_min=bins[i], z_max=bins[i+1])
            for i in range(len(bins) - 1)
        ]

    def predict(self, z):
        binids = np.digitize(z, self.bins) - 1
        return self.bin_mean[binids]


class HistogramCalibrator(BaseCalibrator):
    def __init__(self, n_bins=10, strategy='uniform'):
        self.n_bins = n_bins
        self.strategy = strategy
        self.bins = None  # edges
        self.bin_mean = None  # heights

    def fit(self, z, y):
        if self.strategy == 'uniform':
            bins = np.linspace(0, 1, self.n_bins + 1)
        elif self.strategy == 'quantile':
            bins = np.quantile(z, np.linspace(0, 1, self.n_bins + 1))
        else:
            raise ValueError('strategy should be either "uniform" or "quantile".')
        bins[0], bins[-1] = 0, 1 + 1e-8  # 1e-16 doesn't work: 1 + 1e-16 == 1
        binids = np.digitize(z, bins) - 1
        bin_total = np.bincount(binids, minlength=self.n_bins)
        bin_true = np.bincount(binids, weights=y, minlength=self.n_bins)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            # fill nan by interpolation assuming smoothness
            self.bin_mean = interpolate_nan(bin_true / bin_total)
        self.bins = bins
        return self

    def predict(self, z):
        binids = np.digitize(z, self.bins) - 1
        return self.bin_mean[binids]


class IsotonicCalibrator(BaseCalibrator):
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds='clip')
        self.z_ = None  # thresholds
        self.y_ = None  # predictions

    def fit(self, z, y):
        self.iso.fit(z, y)
        self.z_ = self.iso.X_thresholds_
        self.y_ = self.iso.y_thresholds_
        return self

    def predict(self, z):
        # NOTE: linear interpolation
        return self.iso.predict(z)


class PlattCalibrator(BaseCalibrator):
    def __init__(self):
        self.reg = LogisticRegression()

    def fit(self, z, y):
        x = logit(z).reshape(-1, 1)
        self.reg.fit(x, y)
        return self

    def predict(self, z):
        x = logit(z).reshape(-1, 1)
        return self.reg.predict_proba(x)[:, 1]


class LabelShiftCalibrator(BaseCalibrator):
    def __init__(self, pi_source, pi_target):
        w1 = pi_target / pi_source
        w0 = (1 - pi_target) / (1 - pi_source)
        self.r = w0 / w1  # odds ratio

    def predict(self, z):
        z = np.maximum(z, 1e-16)
        return 1 / (1 + self.r * (1 - z) / z)
