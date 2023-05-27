import numpy as np


def CAL(B, n, delta):
    a = np.log(4 * B / delta)
    b = 2 * (n/B - 1)
    return (np.sqrt(a / b) + B / n) ** 2


def SHA(B, K=None):
    if K is None:
        return 2 / B
    else:
        # if smoothness assumption (A3) holds with constant K
        return 8 * K**2 / B**2


def RISK(B, n, delta, K):
    return CAL(B, n, delta) + SHA(B, K)