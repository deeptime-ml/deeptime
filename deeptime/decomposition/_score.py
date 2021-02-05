import numbers
from typing import Optional, Union

import numpy as np

from ..numeric import is_sorted, spd_inv_sqrt, schatten_norm


def vamp_score(koopman_model, r: Union[float, str],
               covariances_test=None, dim: Optional[int] = None, epsilon: float = 1e-10):
    """Compute the VAMP score between a covariance-based Koopman model and potentially a
    test model for cross-validation.

    Parameters
    ----------
    koopman_model : deeptime.decomposition.CovarianceKoopmanModel
        The model to score.
    r : float or str
        The type of score to evaluate. Can by an floating point value greater or equal to 1 or 'E', yielding the
        VAMP-r score or the VAMP-E score, respectively. :cite:`vampscore-wu2020variational`
        Typical choices (also accepted as inputs) are:

        *  'VAMP1'  Sum of singular values of the half-weighted Koopman matrix.
                    If the model is reversible, this is equal to the sum of
                    Koopman matrix eigenvalues, also called Rayleigh quotient :cite:`vampscore-wu2020variational`.
        *  'VAMP2'  Sum of squared singular values of the half-weighted Koopman
                    matrix :cite:`vampscore-wu2020variational`. If the model is reversible, this is
                    equal to the kinetic variance :cite:`vampscore-noe2015kinetic`.
        *  'VAMPE'  Approximation error of the estimated Koopman operator with respect to
                    the true Koopman operator up to an additive constant :cite:`vampscore-wu2020variational` .

    covariances_test : deeptime.covariance.CovarianceModel, optional, default=None

        If `test_model` is not None, this method computes the cross-validation score
        between self and `covariances_test`. It is assumed that self was estimated from
        the "training" data and `test_model` was estimated from the "test" data. The
        score is computed for one realization of self and `test_model`. Estimation
        of the average cross-validation score and partitioning of data into test and
        training part is not performed by this method.

        If `covariances_test` is None, this method computes the VAMP score for the model
        contained in self.

    dim : int, optional, default=None
        Artificially restrict the scoring to the top `dim` slowest processes.

    epsilon : float, default=1e-10


    Returns
    -------
    score : float
        If `test_model` is not None, returns the cross-validation VAMP score between
        self and `test_model`. Otherwise return the selected VAMP-score of self.

    Notes
    -----
    If the Koopman model was estimated using correlations that are based on data with its sample mean removed,
    this effectively removes the constant function from the singular function space and artificially lowers the score
    by 1. This is accounted for in this method, i.e., if :code:`koopman_model.cov.data_mean_removed` evaluates to
    `True`, the score is internally incremented by 1.

    The VAMP-:math:`r` and VAMP-E scores are computed according to :cite:`vampscore-wu2020variational`,
    Equation (33) and Equation (30), respectively.

    References
    ----------
    .. bibliography:: /references.bib
        :style: unsrt
        :filter: docname in docnames
        :keyprefix: vampscore-
    """
    if dim is not None:
        dim = min(koopman_model.koopman_matrix.shape[0], dim)
    if isinstance(r, str):
        r = r.lower()
        r = r.replace("vamp", "")
        if r.isnumeric():
            r = float(r)
        else:
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
        raise ValueError(f"Shape mismatch, the covariances had "
                         f"shapes {koopman_model.cov.cov_00.shape} and {cov_test.cov_00.shape}.")
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
        score = schatten_norm(ABC, r) ** r
    if koopman_model.cov.data_mean_removed:
        score += 1
    return score
