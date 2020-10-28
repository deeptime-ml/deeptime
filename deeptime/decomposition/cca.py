from typing import Optional, Tuple

import numpy as np
import scipy

from ..base import Model, Estimator
from ..numeric import sort_eigs
from ..kernels import Kernel


class KernelCCAModel(Model):
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

    def __init__(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray):
        super().__init__()
        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors

    @property
    def eigenvalues(self) -> np.ndarray:
        return self._eigenvalues

    @property
    def eigenvectors(self) -> np.ndarray:
        return self._eigenvectors


class KernelCCA(Estimator):
    r""" Estimator implementing the kernelized version :cite:`kcca-bach2002kernel` of canonical correlation
    analysis (CCA :cite:`kcca-hotelling1992relations`).

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
    .. bibliography:: /references.bib
        :style: unsrt
        :filter: docname in docnames
        :keyprefix: kcca-
    """

    def __init__(self, kernel: Kernel, n_eigs: int, epsilon: float = 1e-6):
        super().__init__()
        self.kernel = kernel
        self.n_eigs = n_eigs
        self.epsilon = epsilon

    def fit(self, data: Tuple[np.ndarray, np.ndarray], **kwargs):
        r""" Fit this estimator instance onto data.

        Parameters
        ----------
        data : Tuple of np.ndarray
            Input data consisting of a pair of data matrices.
        **kwargs
            Ignored kwargs.

        Returns
        -------
        self : KernelCCA
            Reference to self.
        """
        gram_0 = self.kernel.gram(data[0])
        gram_t = self.kernel.gram(data[1])
        # center Gram matrices
        n = data[0].shape[0]
        I = np.eye(n)
        N = I - 1 / n * np.ones((n, n))
        G_0 = N @ gram_0 @ N
        G_1 = N @ gram_t @ N

        A = scipy.linalg.solve(G_0 + self.epsilon * I, G_0, assume_a='sym') \
            @ scipy.linalg.solve(G_1 + self.epsilon * I, G_1, assume_a='sym')

        eigenvalues, eigenvectors = scipy.linalg.eig(A)
        eigenvalues, eigenvectors = sort_eigs(eigenvalues, eigenvectors)

        # determine effective rank m and perform low-rank approximations.
        if eigenvalues.shape[0] > self.n_eigs:
            eigenvectors = eigenvectors[:, :self.n_eigs]
            eigenvalues = eigenvalues[:self.n_eigs]

        self._model = KernelCCAModel(eigenvalues, eigenvectors)
        return self

    def fetch_model(self) -> Optional[KernelCCAModel]:
        r""" Yields the latest estimated model or None.

        Returns
        -------
        model : KernelCCAModel or None
            The latest estimated model or None.
        """
        return super().fetch_model()
