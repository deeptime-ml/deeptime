from typing import Tuple

import numpy as np

from ..base import Estimator, Model
from ..kernels import Kernel
from ..numeric import sort_by_norm


class KernelEDMDModel(Model):
    def __init__(self, P, K):
        super().__init__()
        self.P = P
        self.K = K


class KernelEDMDEstimator(Estimator):

    def __init__(self, kernel: Kernel, epsilon: float = 0., n_eig: int = 5):
        super().__init__()
        self.kernel = kernel
        self.epsilon = epsilon
        self.n_eig = n_eig

    def fit(self, data: Tuple[np.ndarray, np.ndarray], **kwargs):
        gram_0 = self.kernel.gram(data[0])
        gram_1 = self.kernel.apply(*data)

        if self.epsilon > 0:
            reg = self.epsilon * np.eye(gram_0.shape[0])
        else:
            reg = 0
        A = np.linalg.pinv(gram_0 + reg, rcond=1e-15) @ gram_1
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvalues, eigenvectors = sort_by_norm(eigenvalues, eigenvectors)
        perron_frobenius_operator = eigenvectors
        koopman_operator = gram_0 @ eigenvectors

        self._model = KernelEDMDModel(perron_frobenius_operator, koopman_operator)

        return self
