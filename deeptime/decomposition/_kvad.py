from typing import Optional, Callable

import numpy as np
from deeptime.basis import Identity

from ..base import Transformer, Model, EstimatorTransformer
from ..covariance import Covariance
from ..kernels import Kernel
from ..numeric import spd_truncated_svd
from ..util.types import to_dataset


class KVADModel(Model, Transformer):

    def __init__(self, koopman_matrix: np.ndarray, observable_transform, covariances,
                 singular_values, singular_vectors, score):
        super(KVADModel, self).__init__()
        self.koopman_matrix = koopman_matrix
        self.observable_transform = observable_transform
        self.covariances = covariances
        self.singular_values = singular_values
        self.singular_vectors = singular_vectors
        self.score = score

    def transform(self, data, **kwargs):
        return self.covariances.whiten(self.observable_transform(data)) @ self.singular_vectors


class KVAD(EstimatorTransformer):

    def __init__(self, kernel: Kernel, lagtime: Optional[int] = None,
                 dim: Optional[int] = None,
                 epsilon: float = 1e-6,
                 observable_transform: Callable[[np.ndarray], np.ndarray] = Identity()):
        super().__init__()
        self.kernel = kernel
        self.dim = dim
        self.lagtime = lagtime
        self.epsilon = epsilon
        self.observable_transform = observable_transform

    @property
    def observable_transform(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._observable_transform

    @observable_transform.setter
    def observable_transform(self, value: Callable[[np.ndarray], np.ndarray]):
        self._observable_transform = value

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

    @property
    def dim(self) -> Optional[int]:
        return self._dim

    @dim.setter
    def dim(self, value: Optional[int]):
        self._dim = value

    def fit(self, data, **kwargs):
        dataset = to_dataset(data, lagtime=self.lagtime)
        x, y = dataset[:]

        chi_x = self.observable_transform(x)
        chi_y = self.observable_transform(y)

        N = y.shape[0]
        M = chi_x.shape[1]

        assert chi_x.shape == (N, M)
        assert chi_y.shape == (N, M)

        g_yy = self.kernel.gram(y)
        assert g_yy.shape == (N, N)

        cov = Covariance().fit(chi_x).fetch_model()
        chi_x_w = cov.whiten(chi_x, epsilon=self.epsilon)
        chi_y_w = cov.whiten(chi_y, epsilon=self.epsilon)

        x_g_x = np.linalg.multi_dot((chi_x_w.T, g_yy, chi_x_w)) / (N*N)
        singular_values, singular_vectors = spd_truncated_svd(x_g_x, dim=self.dim, eps=self.epsilon)

        f_x = chi_x_w @ singular_vectors
        f_y = chi_y_w @ singular_vectors

        K = np.zeros((len(singular_values) + 1, len(singular_values) + 1), dtype=y.dtype)
        K[0, 0] = 1
        K[0, 1:] = singular_vectors.T.dot(chi_y_w.mean(axis=0))
        K[1:, 1:] = 1 / N * f_x.T @ f_y

        score = np.sum(singular_values) + np.mean(g_yy)
        self._model = KVADModel(K, self.observable_transform, cov, singular_values, singular_vectors, score)
        return self
