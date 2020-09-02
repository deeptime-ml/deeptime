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

    def __str__(self):
        return f"GaussianKernel[sigma={self.sigma}]"


class GeneralizedGaussianKernel(BaseKernel):
    r""" Implementation of the generalized Gaussian kernel with bandwidth per dimension. It is defined by

    .. math::

        \kappa (x,y) = \exp \left( \frac{1}{2}(x-y)^\top \Sigma^{-1} (x-y) \right),

    where :math:`\Sigma = \mathrm{diag} (\sigma_1,\ldots,\sigma_d)` are the bandwidths of the kernel."""

    def __init__(self, sigmas):
        r"""Creates a new Generalized gaussian kernel.

        Parameters
        ----------
        sigmas : (d,) ndarray
            The bandwidths. Must match the dimension of the data that is used to evaluate the kernel.
        """
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

    def __str__(self):
        return f"GeneralizedGaussianKernel[sigmas={','.join(f'{s:.3f}' for s in self._sigmas)}]"


class LaplacianKernel(BaseKernel):
    r""" Implementation of the Laplacian kernel

    .. math::

        \kappa (x,y) = \exp \left( \| x-y \|_2 / \sigma\right),

    where :math:`\sigma` is the bandwidth of the kernel."""

    def __init__(self, sigma):
        self._sigma = sigma

    def _evaluate(self, x, y) -> float:
        return np.exp(-np.linalg.norm(x - y) / self._sigma)

    def apply(self, data_1: np.ndarray, data_2: np.ndarray) -> np.ndarray:
        return np.exp(-distance.cdist(data_1, data_2, metric='euclidean') / self._sigma)

    def __str__(self):
        return f"LaplacianKernel[sigma={self._sigma:.3f}]"


class PolynomialKernel(BaseKernel):
    r""" Implementation of the polynomial kernel

    .. math::

        \kappa (x,y) = (x^\top y + c)^d,

    where :math:`p` is the degree and :math:`c` is the inhomogeneity of the Ker."""

    def __init__(self, degree: int, inhomogeneity: float = 1.):
        r""" Creates a new polynomial kernel.

        Parameters
        ----------
        degree : int
            The degree, must be non-negative.
        inhomogeneity : float, optional, default=1.
            The inhomogeneity.
        """
        assert degree >= 0
        self.degree = degree
        self.inhomogeneity = inhomogeneity

    def _evaluate(self, x, y) -> float:
        return (self.inhomogeneity + np.dot(x, y)) ** self.degree

    def apply(self, data_1: np.ndarray, data_2: np.ndarray) -> np.ndarray:
        ri = np.expand_dims(data_1, axis=1)
        rj = np.expand_dims(data_2, axis=0)
        prod = ri * rj
        scalar_products = np.add.reduce(prod, axis=-1, keepdims=False)
        return (self.inhomogeneity + scalar_products) ** self.degree

    def __str__(self):
        return f"PolynomialKernel[degree={self.degree}, inhomogeneity={self.inhomogeneity}]"
