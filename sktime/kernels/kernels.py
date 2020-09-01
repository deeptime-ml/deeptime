import numpy as np
from scipy.spatial import distance

from . import BaseKernel


class GaussianKernel(BaseKernel):
    r""" Implementation of the Gaussian kernel

    .. math::

        \kappa (x,y) = \exp \left(-\sum_k (x_k - y_k)^2 / (2\sigma^2)\right),

    where :math:`\sigma` is the bandwidth of the kernel."""

    def __init__(self, sigma):
        r""" Creates a new Gaussian kernel.

        Parameters
        ----------
        sigma : float
            The bandwidth.
        """
        self._sigma = np.array(sigma, dtype=np.float64)

    @property
    def sigma(self) -> np.ndarray:
        r""" Bandwidth of the Gaussian kernel.

        :getter: Yields the bandwidth.
        :type: (1,) ndarray
        """
        return self._sigma

    def _evaluate(self, x, y) -> float:
        return np.exp(-np.square(np.linalg.norm(x - y)) / (2 * np.square(self.sigma)))

    def apply(self, data_1: np.ndarray, data_2: np.ndarray) -> np.ndarray:
        s = self.sigma.astype(data_1.dtype)
        return np.exp(-distance.cdist(data_1, data_2, metric='sqeuclidean') / (2 * s * s))

    def derivative(self, x, y):
        r""" Evaluates the derivative of the kernel at points :math:`x\in\mathbb{R}^d` and :math:`y\in\mathbb{R}^d`.

        It is given by

        .. math::

            D_{(x,y)}\kappa (x,y) =  -\frac{1}{\sigma^2} (x - y)\kappa(x, y).

        Parameters
        ----------
        x : (d,) ndarray
            x coordinate
        y : (d,) ndarray
            y coordinate

        Returns
        -------
        d_k : float
            Derivative at `(x, y)`.
        """
        return -1 / np.square(self.sigma) * (x - y) * self(x, y)

    def dderivative(self, x, y):
        r""" Evaluates the second derivative of the kernel at points :math:`x\in\mathbb{R}^d`
        and :math:`y\in\mathbb{R}^d`.

        It is given by

        .. math::

            D^{(2)}_{(x,y)}\kappa(x,y) = \left(\frac{(x-y)(x-y)^\top}{\sigma^4}
                                               - \frac{1}{\sigma^2}\mathbb{I}_d\right)\kappa(x,y).

        Parameters
        ----------
        x : (d,) ndarray
            x coordinate
        y : (d,) ndarray
            y coordinate

        Returns
        -------
        dd_k : (d, d) ndarray
            Second derivative at (x, y).
        """
        d = 1 if x.ndim == 0 else x.shape[0]
        return (1 / self.sigma ** 4 * np.outer(x - y, x - y) - 1 / np.square(self.sigma) * np.eye(d)) * self(x, y)

    def laplace(self, x, y):
        r""" Evaluates the laplace :math:`\Delta \kappa(x,y)` of this kernel.

        It is given by

        .. math::

            \Delta\kappa (x,y) = \left(\frac{\|x - y\|^2}{\sigma^4} - \frac{d}{\sigma^2}\right) \kappa(x, y).

        Parameters
        ----------
        x : (d, ) ndarray
            x coordinate
        y : (d, ) ndarray
            y coordinate

        Returns
        -------
        l : float
            Laplace of the kernel at `(x, y)`.
        """
        return (1 / np.power(self.sigma, 4) * np.linalg.norm(x - y) ** 2 - len(x) / self.sigma ** 2) * self(x, y)

    def __str__(self):
        return f"GaussianKernel[sigma={self.sigma}]"


class GeneralizedGaussianKernel(BaseKernel):

    def __init__(self, sigmas):
        self._sigmas = sigmas.squeeze()
        self._D = np.diag(1. / (2 * np.square(sigmas)))
        self._sqrt_D = 1. / (np.sqrt(2) * sigmas)

    def _evaluate(self, x, y) -> float:
        diff = (x - y).squeeze()
        return np.exp(-diff @ self._D @ diff.T)

    def apply(self, data_1: np.ndarray, data_2: np.ndarray) -> np.ndarray:
        ri = np.expand_dims(data_1, axis=1)
        rj = np.expand_dims(data_2, axis=0)
        rij = (ri - rj) * self._sqrt_D
        D = np.add.reduce(np.square(rij), axis=-1, keepdims=False)
        return np.exp(-D)

    def derivative(self, x, y):
        return -2 * self._D @ (x - y) * self(x, y)

    def dderivative(self, x, y):
        return (np.outer(2 * self._D @ (x - y), 2 * self._D @ (x - y)) - 2 * self._D) * self(x, y)

    def laplace(self, x, y):
        return (np.linalg.norm(2 * self._D @ (x - y)) ** 2 - 2 * np.trace(self._D)) * self(x, y)

    def __str__(self):
        return f"GeneralizedGaussianKernel[sigmas={','.join(f'{s:.3f}' for s in self._sigmas)}]"
