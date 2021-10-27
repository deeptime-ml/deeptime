from typing import Optional

import numpy as np
import scipy

from ..base import Estimator
from ..decomposition import TransferOperatorModel
from ..kernels import Kernel
from ..numeric import sort_eigs
from ..util.types import to_dataset


class KernelCCAModel(TransferOperatorModel):
    r""" The model produced by the :class:`KernelCCA` estimator.

    Parameters
    ----------
    eigenvalues : (n, ) ndarray
        The eigenvalues.
    eigenvectors : (m, n) ndarray
        The eigenvectors of the nonlinear transform of the input data.

    See Also
    --------
    KernelCCA
    """

    def __init__(self, data, kernel: Kernel, eigenvalues: np.ndarray, eigenvectors: np.ndarray):
        self._kernel = kernel
        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors
        self._data = data
        super().__init__(np.diag(np.sqrt(eigenvalues)),
                         instantaneous_obs=lambda x: (self.kernel.apply(x, self._data) @ self.eigenvectors),
                         timelagged_obs=lambda x: (self.kernel.apply(x, self._data) @ self.eigenvectors))

    @property
    def kernel(self) -> Kernel:
        r""" The kernel that was used for estimation. """
        return self._kernel

    @property
    def eigenvalues(self) -> np.ndarray:
        return self._eigenvalues

    @property
    def eigenvectors(self) -> np.ndarray:
        return self._eigenvectors


class KernelCCA(Estimator):
    r""" Estimator implementing the kernelized version of canonical correlation analysis.
    :footcite:`bach2002kernel` (CCA :footcite:`hotelling1992relations`)

    Parameters
    ----------
    kernel : Kernel
        The kernel to be used, see :mod:`deeptime.kernels` for a selection of predefined kernels.
    n_eigs : int
        Number of eigenvalue/eigenvector pairs to use for low-rank approximation.
    epsilon : float, optional, default=1e-6
        Regularization parameter.

    See Also
    --------
    KernelCCAModel

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, kernel: Kernel, n_eigs: int, epsilon: float = 1e-6):
        super().__init__()
        self.kernel = kernel
        self.n_eigs = n_eigs
        self.epsilon = epsilon

    def fit(self, data, **kwargs):
        r""" Fit this estimator instance onto data.

        Parameters
        ----------
        data
            Input data, see :meth:`to_dataset <deeptime.util.types.to_dataset>` for options.
        **kwargs
            Kwargs, may contain lagtime.

        Returns
        -------
        self : KernelCCA
            Reference to self.
        """
        dataset = to_dataset(data, lagtime=kwargs.get("lagtime", None))
        x, y = dataset[:]
        gram_0 = self.kernel.gram(x)
        gram_t = self.kernel.gram(y)

        # center Gram matrices
        n = x.shape[0]
        I = np.eye(n)  # identity
        N = I - np.full((n, n), fill_value=1. / n)  # centering matrix
        G_0 = np.linalg.multi_dot([N, gram_0, N])
        G_1 = np.linalg.multi_dot([N, gram_t, N])

        K = scipy.linalg.solve(G_0 + self.epsilon * I, G_0, assume_a='sym')
        Ak = scipy.linalg.solve(G_1 + self.epsilon * I, G_1, assume_a='sym')
        A = K @ Ak
        eigenvalues, eigenvectors = scipy.linalg.eig(A)
        eigenvalues, eigenvectors = sort_eigs(eigenvalues, eigenvectors)

        # determine effective rank m and perform low-rank approximations.
        if eigenvalues.shape[0] > self.n_eigs:
            eigenvectors = eigenvectors[:, :self.n_eigs]
            eigenvalues = eigenvalues[:self.n_eigs]

        self._model = KernelCCAModel(x, self.kernel, eigenvalues, eigenvectors)
        return self

    def fetch_model(self) -> Optional[KernelCCAModel]:
        r""" Yields the latest estimated model or None.

        Returns
        -------
        model : KernelCCAModel or None
            The latest estimated model or None.
        """
        return super().fetch_model()
