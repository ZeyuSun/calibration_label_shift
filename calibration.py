import warnings
import numpy as np
from matplotlib import pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from utils import expectation, sigmoid, logit, interpolate_nan


class BaseCalibrator:
    def fit(self, z, y):
        return self

    def predict(self, z):
        pass

    def __call__(self, z):
        return self.predict(z)

    def plot(self, ax=None, set_layout=None, *args, **kwargs):
        if ax is None:
            if plt.get_fignums():
                ax = plt.gca()
            else:
                _, ax = plt.subplots(figsize=(3, 3))

        if hasattr(self, 'bins'):
            ax.stairs(self.bin_mean, edges=self.bins, baseline=None, *args, **kwargs)
            ax.set_ylim(bottom=-0.05)
        elif hasattr(self, 'z_'):
            ax.plot(self.z_, self.y_, *args, **kwargs)
        else:
            zz = np.linspace(0, 1, 1000)
            ax.plot(zz, self.predict(zz), *args, **kwargs)

        if set_layout:
            plt.xlabel('Predicted Probability')
            plt.ylabel('Empirical Frequency')
            ax.set_aspect('equal', 'box')  # plt.axis('scaled'): ylim bottom change back to 0
            plt.grid()
            plt.legend()
        return ax


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

    def fit(self, z, y):
        if self.strategy == 'uniform':
            bins = np.linspace(0, 1, self.n_bins + 1)
        elif self.strategy == 'quantile':
            bins = np.quantile(z, np.linspace(0, 1, self.n_bins + 1))
        else:
            raise ValueError('strategy should be either "uniform" or "quantile".')
        bins[0], bins[-1] = 0, 1 + 1e-16
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