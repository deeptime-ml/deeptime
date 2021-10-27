import warnings

import numpy as np

from deeptime.util.exceptions import NotConvergedWarning


def mle_trev(C, maxerr=1.0e-12, maxiter=int(1.0E6), warn_not_converged=True, return_statdist=False,
             eps_mu=1.0e-15):
    from ._mle_bindings import mle_trev_dense
    from ...analysis import is_connected

    assert maxerr > 0, 'maxerr must be positive'
    assert maxiter > 0, 'maxiter must be positive'
    assert C.shape[0] == C.shape[1], 'C must be a square matrix.'
    assert is_connected(C, directed=True), 'C must be strongly connected'

    if C.dtype not in (np.float32, np.float64, np.longdouble):
        C = C.astype(np.float64)
    dtype = C.dtype

    C_sum = C.sum(axis=1).astype(dtype, order='C', copy=False)
    CCt = (C + C.T).astype(dtype, order='C', copy=False)
    T = np.zeros(C.shape, dtype=dtype, order='C')
    mu = np.zeros(C.shape[0], dtype=dtype, order='C')

    code = mle_trev_dense(T, CCt, C_sum, CCt.shape[0], maxerr, maxiter, mu, eps_mu)
    if code == -5 and warn_not_converged:
        warnings.warn('Reversible transition matrix estimation didn\'t converge.',
                      NotConvergedWarning)

    if return_statdist:
        return T, mu
    else:
        return T


def mle_trev_given_pi(C, mu, maxerr=1.0E-12, maxiter=1000000):
    from ._mle_bindings import mle_trev_given_pi_dense
    from ...analysis import is_connected

    assert maxerr > 0, 'maxerr must be positive'
    assert maxiter > 0, 'maxiter must be positive'
    assert is_connected(C, directed=False), 'C must be (weakly) connected'

    dtype = C.dtype
    if dtype not in (np.float32, np.float64, np.longdouble):
        dtype = np.float64

    c_C = C.astype(dtype, order='C', copy=False)
    c_mu = mu.astype(dtype, order='C', copy=False)

    assert c_C.shape[0] == c_C.shape[1] == c_mu.shape[0], 'Dimensions of C and mu don\'t agree.'

    T = np.zeros_like(c_C, dtype=dtype, order='C')

    code = mle_trev_given_pi_dense(T, c_C, c_mu, C.shape[0], maxerr, maxiter)

    if code == -5:
        warnings.warn('Reversible transition matrix estimation with fixed stationary distribution didn\'t converge.',
                      NotConvergedWarning)
    return T
