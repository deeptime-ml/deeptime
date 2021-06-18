from typing import Optional, Callable

import numpy as np

from deeptime.basis import Identity
from . import TransferOperatorModel
from ..base import EstimatorTransformer
from ..covariance import Covariance
from ..kernels import Kernel
from ..numeric import spd_truncated_svd
from ..util.types import to_dataset


class _KVADTransform:

    def __init__(self, cov, obs_transform, singular_vectors):
        self.cov = cov
        self.obs_transform = obs_transform
        self.singular_vectors = singular_vectors

    def __call__(self, x):
        return self.cov.whiten(self.obs_transform(x)) @ self.singular_vectors


class KVADModel(TransferOperatorModel):
    r"""The model produced by the :class:`KVAD` estimator.

    Parameters
    ----------
    kernel : Kernel
        The kernel that was used for estimation and embedding.
    koopman_matrix : ndarray
        The estimated Koopman matrix.
    observable_transform : callable
        Transformation for data that was used for estimation.
    covariances : deeptime.covariance.CovarianceModel
        Estimated covariances for instantaneous data.
    singular_values : ndarray
        Singular values of truncated SVD.
    singular_vectors : ndarray
        Singular vectors of truncated SVD.
    score : float
        Estimated KVAD score.
    """

    def __init__(self, kernel, koopman_matrix: np.ndarray, observable_transform, covariances,
                 singular_values, singular_vectors, score):
        transf = _KVADTransform(covariances, observable_transform, singular_vectors)
        super(KVADModel, self).__init__(koopman_matrix=koopman_matrix,
                                        instantaneous_obs=transf,
                                        timelagged_obs=transf)
        self.kernel = kernel
        self.observable_transform = observable_transform
        self.covariances = covariances
        self.singular_values = singular_values
        self.singular_vectors = singular_vectors
        self.score = score


class KVAD(EstimatorTransformer):
    r""" An estimator for the "Kernel embedding based variational approach for dynamical systems" (KVAD).

    Theory and introduction into the method can be found in :footcite:`tian2020kernel`.

    Parameters
    ----------
    kernel : Kernel
        The kernel to be used, see :mod:`deeptime.kernels` for a selection of predefined kernels.
    lagtime : int, optional, default=None
        Lagtime if data is not a list of instantaneous and time-lagged data pairs but a trajectory instead.
    dim : int, optional, default=None
        Dimension cutoff parameter.
    epsilon : float, default=1e-6
        Regularization parameter for truncated SVD.
    observable_transform : callable, optional, default=Identity
        A feature transformation on the raw data which is used to estimate the model.

    See Also
    --------
    KVADModel

    References
    ----------
    .. footbibliography::
    """

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
        r""" Transforms observable instantaneous and time-lagged data into feature space.

        :type: Callable[[ndarray], ndarray]
        """
        return self._observable_transform

    @observable_transform.setter
    def observable_transform(self, value: Callable[[np.ndarray], np.ndarray]):
        self._observable_transform = value

    @property
    def epsilon(self):
        r""" Regularization parameter for truncated SVD.

        :type: float
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

    @property
    def dim(self) -> Optional[int]:
        r""" Dimension cutoff for the decomposition.

        :type: int or None
        """
        return self._dim

    @dim.setter
    def dim(self, value: Optional[int]):
        self._dim = value

    def fit(self, data, **kwargs):
        dataset = to_dataset(data, lagtime=self.lagtime)
        x, y = dataset[:]

        chi_x = self.observable_transform(x)
        chi_y = self.observable_transform(y)

        n_data = y.shape[0]
        chi_output_dim = chi_x.shape[1]

        assert chi_x.shape == (n_data, chi_output_dim)
        assert chi_y.shape == (n_data, chi_output_dim)

        g_yy = self.kernel.gram(y)
        assert g_yy.shape == (n_data, n_data)

        cov = Covariance().fit(chi_x).fetch_model()
        chi_x_w = cov.whiten(chi_x, epsilon=self.epsilon)
        chi_y_w = cov.whiten(chi_y, epsilon=self.epsilon)

        x_g_x = np.linalg.multi_dot((chi_x_w.T, g_yy, chi_x_w)) / (n_data * n_data)
        singular_values, singular_vectors = spd_truncated_svd(x_g_x, dim=self.dim, eps=self.epsilon)

        f_x = chi_x_w @ singular_vectors
        f_y = chi_y_w @ singular_vectors

        koopman_matrix = np.zeros((len(singular_values) + 1, len(singular_values) + 1), dtype=y.dtype)
        koopman_matrix[0, 0] = 1
        koopman_matrix[0, 1:] = singular_vectors.T.dot(chi_y_w.mean(axis=0))
        koopman_matrix[1:, 1:] = (1. / n_data) * f_x.T @ f_y

        score = np.sum(singular_values) + np.mean(g_yy)
        self._model = KVADModel(self.kernel, koopman_matrix, self.observable_transform,
                                cov, singular_values, singular_vectors, score)
        return self
