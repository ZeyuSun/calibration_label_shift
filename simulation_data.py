import numpy as np
import matplotlib.pyplot as plt
import scipy
from utils import dlogit, logit, sigmoid


class BaseData:
    def __init__(self):
        self.K = None

    def pz(self, z):
        raise NotImplementedError

    def py_given_z(self, z):
        raise NotImplementedError

    def sample(self, n, variables='zy'):
        raise NotImplementedError

    def plot(self):
        x, y = self.sample(10000)
        xlim = [-10, 7.5]

        fig, ax = plt.subplots(2, 1)
        ax[0].hist([x[y==0], x[y==1]], histtype='step', stacked=False, bins=50)
        xlim = ax[0].get_xlim()

        xx = np.linspace(*xlim, 1000)
        ax[1].plot(xx, self.py_given_x(xx), label='p(y=1|x)')
        ax[1].set_xlim(xlim)
        plt.legend()


class GaussianMixtureData(BaseData):
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

    def py_given_x(self, x):
        logodds = 2 * self.mu * x + np.log(self.p / (1 - self.p))
        return sigmoid(logodds)

    def py_given_z(self, z):
        x = self.f_inv(z)
        return self.py_given_x(x)

    def sample(self, n, variables='zy'):
        """
        Args:
            n: sample size
            variables: string of variable names to return
        """
        y = np.random.binomial(1, self.p, size=n)
        n1 = np.sum(y)
        x = np.zeros(n)
        x[y == 0] = np.random.normal(-self.mu, 1, size=n-n1)
        x[y == 1] = np.random.normal(self.mu, 1, size=n1)
        z = self.f(x)
        data = {'x': x, 'z': z, 'y': y}
        return [data[var] for var in variables]


class GaussianMixtureData2(BaseData):
    def __init__(self):
        self.n_groups = 3
        self.params = [
            {'loc': -3, 'scale': 2},
            {'loc': 0, 'scale': 1},
            {'loc': 4, 'scale': 1.5},
        ]
        self.weights = [0.5, 0.2, 0.3]
        self.classes = np.array([0, 1, 1])
        self.K = None

    def sample(self, n, variables='xy'):
        assert variables == 'xy'
        g = np.random.choice(self.n_groups, size=n, p=self.weights)
        x = np.zeros(n)
        for i in range(self.n_groups):
            idx = g == i
            x[idx] = np.random.normal(**self.params[i], size=np.sum(idx))
        y = np.array(self.classes)[g]
        data = {'x': x, 'y': y}
        return [data[var] for var in variables]

    def px_given_g(self, x, g):
        return scipy.stats.norm.pdf(x, **self.params[g])

    def pg(self, g):
        return self.weights[g]

    def py_given_x(self, x):
        joint = np.array([self.px_given_g(x, g) * self.pg(g) for g in range(self.n_groups)])
        return np.sum(joint[self.classes == 1], axis=0) / np.sum(joint, axis=0)


class BetaCalibrationData(BaseData):
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

    def sample(self, n, variables='zy'):
        assert variables == 'zy'
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