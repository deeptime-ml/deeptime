"""
    Sparse assessment module of msm analysis package
"""

import numpy as np
import scipy.sparse as sparse

from deeptime.numeric import allclose_sparse
from ._stationary_vector import stationary_distribution


def is_transition_matrix(T, tol):
    """
    True if T is a transition matrix

    Parameters
    ----------
    T : scipy.sparse matrix
        Matrix to check
    tol : float
        tolerance to check with

    Returns
    -------
    Truth value: bool
        True, if T is positive and normed
        False, otherwise

    """
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        return False

    if sparse.issparse(T):
        T = T.tocsr()  # compressed sparse row for fast row slicing
        values = T.data  # non-zero entries of T
    else:
        values = T

    is_positive = np.allclose(values, np.abs(values), rtol=tol)
    is_normed = np.allclose(T.sum(axis=1), 1.0, rtol=tol)

    return is_positive and is_normed


def is_rate_matrix(K, tol):
    r""" True if K is a rate matrix

    Parameters
    ----------
    K : scipy.sparse matrix
        Matrix to check
    tol : float
        tolerance to check with

    Returns
    -------
    Truth value : bool
        True, if K negated diagonal is positive and row sums up to zero.
        False, otherwise
    """
    if sparse.issparse(K):
        K = K.tocsr()

    # check rows sum up to zero.
    row_sum = K.sum(axis=1)
    sum_eq_zero = np.allclose(row_sum, 0., atol=tol)

    R = K - K.diagonal()
    if sparse.issparse(R):
        R = R.values  # extract nonzero entries
    off_diagonal_positive = np.allclose(R, np.abs(R), rtol=0, atol=tol)

    return off_diagonal_positive and sum_eq_zero


def is_reversible(T, mu=None, tol=1e-15):
    r"""
    checks whether T is reversible in terms of given stationary distribution.
    If no distribution is given, it will be calculated out of T.

    performs follwing check:
    :math:`\pi_i P_{ij} = \pi_j P_{ji}
    Parameters
    ----------
    T : scipy.sparse matrix
        Transition matrix
    mu : numpy.ndarray vector
        stationary distribution
    tol : float
        tolerance to check with

    Returns
    -------
    Truth value : bool
        True, if T is a stochastic matrix
        False, otherwise
    """
    if not is_transition_matrix(T, tol):
        raise ValueError("given matrix is not a valid transition matrix.")

    if sparse.issparse(T):
        T = T.tocsr()

    if mu is None:
        mu = stationary_distribution(T)

    if sparse.issparse(T):
        prod = sparse.construct.diags(mu) * T
    else:
        prod = mu[:, None] * T

    if sparse.issparse(T):
        return allclose_sparse(prod, prod.transpose(), rtol=tol)
    else:
        return np.allclose(prod, prod.transpose(), rtol=tol)


def is_connected(T, directed=True):
    r"""Check connectivity of the transition matrix.

    Return true, if the input matrix is completely connected,
    effectively checking if the number of connected components equals one.

    Parameters
    ----------
    T : scipy.sparse matrix
        Transition matrix
    directed : bool, optional
       Whether to compute connected components for a directed  or
       undirected graph. Default is True.

    Returns
    -------
    connected : boolean, returning true only if T is connected.


    """
    if not sparse.issparse(T):
        T = sparse.csr_matrix(T)
    nc = sparse.csgraph.connected_components(T, directed=directed, connection='strong', return_labels=False)
    return nc == 1


def is_ergodic(T, tol):
    """
    checks if T is 'ergodic'

    Parameters
    ----------
    T : scipy.sparse matrix
        Transition matrix
    tol : float
        tolerance

    Returns
    -------
    Truth value : bool
        True, if # strongly connected components = 1
        False, otherwise
    """
    if not sparse.issparse(T):
        T = sparse.csr_matrix(T)
    if not is_transition_matrix(T, tol):
        raise ValueError("given matrix is not a valid transition matrix.")
    return is_connected(T, True)
