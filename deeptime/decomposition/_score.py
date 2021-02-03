import numbers
from typing import Optional, Union

import numpy as np

from deeptime.numeric import schatten_norm as _schatten_norm, is_sorted, spd_inv_sqrt
from ..covariance import CovarianceModel


class KoopmanModel:

    def __init__(self, koopman_matrix):
        self.koopman_matrix = koopman_matrix


class CovarianceKoopmanModel(KoopmanModel):
    def __init__(self, U, K, V, cov):
        assert K.ndim == 1
        super().__init__(np.diag(K))
        self.instantaneous_coefficients = U
        self.timelagged_coefficients = V
        self.singular_values = K
        self.cov = cov


def vamp_score(koopman_model: CovarianceKoopmanModel, r: Union[float, str],
               covariances_test: Optional[CovarianceModel] = None, dim=None, epsilon=1e-6):
    if dim is not None:
        dim = min(koopman_model.koopman_matrix.shape[0], dim)
    if isinstance(r, str):
        r = r.lower()
        assert r == 'e', "only VAMP-E supported, otherwise give as float >= 1"
    else:
        assert isinstance(r, numbers.Number) and r >= 1, "score only for r >= 1 or r = \"E\""
    if covariances_test is None:
        cov_test = koopman_model.cov
    else:
        cov_test = covariances_test
    assert koopman_model.cov.data_mean_removed == cov_test.data_mean_removed, \
        "Covariances must be consistent with respect to the data"
    if koopman_model.cov.cov_00.shape != cov_test.cov_00.shape:
        raise ValueError("Shape mismatch, the covariances ")
    if not is_sorted(koopman_model.singular_values, 'desc'):
        sort_ix = np.argsort(koopman_model.singular_values)[::-1][:dim]  # indices to sort in descending order
    else:
        sort_ix = np.arange(koopman_model.singular_values.shape[0])[:dim]  # already sorted

    U = koopman_model.instantaneous_coefficients[:, sort_ix]
    V = koopman_model.timelagged_coefficients[:, sort_ix]

    if r == 'e':
        K = np.diag(koopman_model.singular_values[sort_ix])
        # see https://arxiv.org/pdf/1707.04659.pdf eqn. (30)
        score = np.trace(2.0 * np.linalg.multi_dot([K, U.T, cov_test.cov_0t, V])
                         - np.linalg.multi_dot([K, U.T, cov_test.cov_00, U, K, V.T, cov_test.cov_tt, V]))
    else:
        # see https://arxiv.org/pdf/1707.04659.pdf eqn. (33)
        A = np.atleast_2d(spd_inv_sqrt(U.T.dot(cov_test.cov_00).dot(U), epsilon=epsilon))
        B = np.atleast_2d(U.T.dot(cov_test.cov_0t).dot(V))
        C = np.atleast_2d(spd_inv_sqrt(V.T.dot(cov_test.cov_tt).dot(V), epsilon=epsilon))
        ABC = np.linalg.multi_dot([A, B, C])
        score = _schatten_norm(ABC, r) ** r
    if koopman_model.cov.data_mean_removed:
        score += 1
    return score
