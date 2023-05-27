import warnings
import numpy as np
from scipy.integrate import quad

eps = 1e-16

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    # inverse of sigmoid
    p = np.clip(p, eps, 1-eps)
    return np.log(p / (1 - p))


def dlogit(p):
    # derivative of logit
    p = np.clip(p, eps, 1-eps)
    return 1 / (p * (1 - p))


def expectation(f, pz, z_min=None, z_max=None):
    """Compute conditional expectation
    E[f(z) | z_min <= z <= z_max], where Z ~ pz supported on [0,1]
    z_min, z_max: float. Default to 0 and 1 if None.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if z_min is None and z_max is None:
            z_min, z_max = 0, 1
            A = 1
        else:
            A = quad(pz, z_min, z_max)[0]
        B = quad(lambda z: f(z) * pz(z), z_min, z_max)[0]
    return B / A


def interpolate_nan(a):
    """Linear interpolation for nan values in a 1d array.
    Nans on the boundary are filled with the nearest non-nan value.
    """
    b = a.copy()
    nans = np.isnan(b)
    i = np.arange(len(b))
    b[nans] = np.interp(i[nans], i[~nans], b[~nans])
    return b