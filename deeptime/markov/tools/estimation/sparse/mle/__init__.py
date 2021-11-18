import warnings
import numpy as np
import scipy

from deeptime.util.exceptions import NotConvergedWarning
from .. import _mle_sparse_bindings as _bindings


def mle_trev(C, maxerr=1.0E-12, maxiter=int(1.0E6),
             warn_not_converged=True, return_statdist=False,
             eps_mu=1.0E-15):
    assert maxerr > 0, 'maxerr must be positive'
    assert maxiter > 0, 'maxiter must be positive'
    assert C.shape[0] == C.shape[1], 'C must be a square matrix.'
    from deeptime.markov.tools.estimation import is_connected
    assert is_connected(C, directed=True), 'C must be strongly connected'

    dtype = C.dtype
    if dtype not in (np.float32, np.float64, np.longdouble):
        dtype = np.float64

    C_sum_py = C.sum(axis=1).A1
    C_sum = C_sum_py.astype(dtype, order='C', copy=False)

    CCt = C + C.T
    # convert CCt to coo format
    CCt_coo = CCt.tocoo()
    n_data = CCt_coo.nnz
    CCt_data = CCt_coo.data.astype(dtype, order='C', copy=False)
    i_indices = CCt_coo.row.astype(int, order='C', copy=True)
    j_indices = CCt_coo.col.astype(int, order='C', copy=True)

    # prepare data array of T in coo format
    T_data = np.zeros(n_data, dtype=dtype, order='C')
    mu = np.zeros(C.shape[0], dtype=dtype, order='C')
    code = _bindings.mle_trev_sparse(T_data, CCt_data, i_indices, j_indices, n_data, C_sum, CCt.shape[0],
                                     maxerr, maxiter, mu, eps_mu)
    if code == -5 and warn_not_converged:
        warnings.warn("Reversible transition matrix estimation with fixed stationary distribution didn't converge.",
                      NotConvergedWarning)

    # T matrix has the same shape and positions of nonzero elements as CCt
    T = scipy.sparse.csr_matrix((T_data, (i_indices, j_indices)), shape=CCt.shape)
    from deeptime.markov.tools.estimation.sparse.transition_matrix import correct_transition_matrix
    T = correct_transition_matrix(T)
    if return_statdist:
        return T, mu
    else:
        return T


def mle_trev_given_pi(C, mu, maxerr=1.0E-12, maxiter=1000000, warn_not_converged=True):
    assert maxerr > 0, 'maxerr must be positive'
    assert maxiter > 0, 'maxiter must be positive'
    from deeptime.markov.tools.estimation import is_connected
    assert is_connected(C, directed=False), 'C must be (weakly) connected'
    dtype = C.dtype
    if dtype not in (np.float32, np.float64, np.longdouble):
        dtype = np.float64
    c_mu = mu.astype(dtype, order='C', copy=False)
    CCt_coo = (C + C.T).tocoo()
    assert CCt_coo.shape[0] == CCt_coo.shape[1] == c_mu.shape[0], 'Dimensions of C and mu don\'t agree.'
    n_data = CCt_coo.nnz
    CCt_data = CCt_coo.data.astype(dtype, order='C', copy=False)
    i_indices = CCt_coo.row.astype(np.uint64, order='C', copy=False)
    j_indices = CCt_coo.col.astype(np.uint64, order='C', copy=False)
    # prepare data array of T in coo format
    T_unnormalized_data = np.zeros(n_data, dtype=dtype, order='C')

    code = _bindings.mle_trev_given_pi_sparse(T_unnormalized_data, CCt_data, i_indices, j_indices, n_data, c_mu,
                                              CCt_coo.shape[0], maxerr, maxiter)

    if code == -5 and warn_not_converged:
        warnings.warn("Reversible transition matrix estimation with fixed stationary distribution didn't converge.",
                      NotConvergedWarning)

    # unnormalized T matrix has the same shape and positions of nonzero elements as the C matrix
    T_unnormalized = scipy.sparse.csr_matrix((T_unnormalized_data, (i_indices.copy(), j_indices.copy())),
                                             shape=CCt_coo.shape)
    # finish T by setting the diagonal elements according to the normalization constraint
    rowsum = T_unnormalized.sum(axis=1).A1
    T_diagonal = scipy.sparse.diags(np.maximum(1.0 - rowsum, 0.0), 0)

    return T_unnormalized + T_diagonal
