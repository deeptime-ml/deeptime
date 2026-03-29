import numpy as np
from scipy.spatial import distance

from . import Kernel


class GaussianKernel(Kernel):
    r""" Implementation of the Gaussian kernel

    .. math::

        \kappa (x,y) = \exp \left(-\sum_k (x_k - y_k)^2 / (2\sigma^2)\right),

    where :math:`\sigma` is the bandwidth of the kernel.

    Parameters
    ----------
    sigma : float
        The bandwidth.
    """

    valid_impls = 'cdist', 'binomial'  #: Valid implementation modes.

    def __init__(self, sigma, impl='cdist'):
        assert impl in GaussianKernel.valid_impls, f"impl needs to be one of {GaussianKernel.valid_impls}"
        self._sigma = np.array(sigma, dtype=np.float64)
        self.impl = impl

    @property
    def impl(self):
        return self._impl

    @impl.setter
    def impl(self, value):
        assert value in GaussianKernel.valid_impls, f"impl {value} not known, " \
                                                    f"supported are {GaussianKernel.valid_impls}"
        self._impl = value

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
        if self.impl == 'cdist':
            D = distance.cdist(data_1, data_2, metric='sqeuclidean')
        elif self.impl == 'binomial':
            x1_norm = np.square(data_1).sum(axis=-1, keepdims=True)
            x2_norm = np.square(data_2).sum(axis=-1, keepdims=True)
            D = x2_norm.T - 2. * data_1 @ data_2.T + x1_norm
            D = np.clip(D, a_min=1e-16, a_max=None)
        return np.exp(-D / (2 * self.sigma * self.sigma))

    def __str__(self):
        return f"GaussianKernel[sigma={self.sigma}, impl={self.impl}]"


class GeneralizedGaussianKernel(Kernel):
    r""" Implementation of the generalized Gaussian kernel with bandwidth per dimension. It is defined by

    .. math::

        \kappa (x,y) = \exp \left( \frac{1}{2}(x-y)^\top \Sigma^{-1} (x-y) \right),

    where :math:`\Sigma = \mathrm{diag} (\sigma_1,\ldots,\sigma_d)` are the bandwidths of the kernel.

    Parameters
    ----------
    sigmas : (d,) ndarray
        The bandwidths. Must match the dimension of the data that is used to evaluate the kernel.
    """

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

    def __str__(self):
        return f"GeneralizedGaussianKernel[sigmas={','.join(f'{s:.3f}' for s in self._sigmas)}]"


class LaplacianKernel(Kernel):
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


class PolynomialKernel(Kernel):
    r""" Implementation of the polynomial kernel

    .. math::

        \kappa (x,y) = (x^\top y + c)^d,

    where :math:`p` is the degree and :math:`c` is the inhomogeneity.

    Parameters
    ----------
    degree : int
        The degree, must be non-negative.
    inhomogeneity : float, must be non-negative, optional, default=1.
        The inhomogeneity.
    """

    def __init__(self, degree: int, inhomogeneity: float = 1.):
        assert degree >= 0
        assert inhomogeneity >= 0
        self.degree = degree
        self.inhomogeneity = inhomogeneity

    def _evaluate(self, x, y) -> float:
        return (self.inhomogeneity + np.dot(x, y)) ** self.degree

    def apply(self, data_1: np.ndarray, data_2: np.ndarray) -> np.ndarray:
        ri = np.expand_dims(data_1, axis=1)
        rj = np.expand_dims(data_2, axis=0)
        scalar_products = np.add.reduce(ri * rj, axis=-1, keepdims=False)
        return (self.inhomogeneity + scalar_products) ** self.degree

    def __str__(self):
        return f"PolynomialKernel[degree={self.degree}, inhomogeneity={self.inhomogeneity}]"
