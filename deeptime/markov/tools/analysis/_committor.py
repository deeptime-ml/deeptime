import numpy as np
from scipy.linalg import solve
from scipy.sparse.linalg import spsolve

import scipy.sparse as sparse

from ._stationary_vector import stationary_distribution


def _set_up_linear_system(K, A, B):
    """Assemble left-hand side W for linear system"""
    """Equation (I)"""
    W = 1.0 * K
    """Equation (II)"""
    if sparse.issparse(W):
        W = W.todok()
        W[list(A), :] = 0.0
        W.tocsr()
        W = W + sparse.coo_matrix((np.ones(len(A)), (list(A), list(A))), shape=W.shape).tocsr()
    else:
        W[list(A), :] = 0.0
        W[list(A), list(A)] = 1.0
    """Equation (III)"""
    if sparse.issparse(W):
        W = W.todok()
        W[list(B), :] = 0.0
        W.tocsr()
        W = W + sparse.coo_matrix((np.ones(len(B)), (list(B), list(B))), shape=W.shape).tocsr()
    else:
        W[list(B), :] = 0.0
        W[list(B), list(B)] = 1.0
    return W


def forward_committor(T, A, B):
    r"""Forward committor between given sets.

    The forward committor u(x) between sets A and B is the probability
    for the chain starting in x to reach B before reaching A.

    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix
    A : array_like
        List of integer state labels for set A
    B : array_like
        List of integer state labels for set B

    Returns
    -------
    u : (M, ) ndarray
        Vector of forward committor probabilities

    Notes
    -----
    The forward committor is a solution to the following
    boundary-value problem

    .. math::

        \sum_j L_{ij} u_{j}=0    for i in X\(A u B) (I)
                      u_{i}=0    for i \in A        (II)
                      u_{i}=1    for i \in B        (III)

    with generator matrix L=(P-I).

    """
    X = set(range(T.shape[0]))
    A = set(A)
    B = set(B)
    AB = A.intersection(B)
    if len(AB) > 0:
        raise ValueError("Sets A and B have to be disjoint")
    L = T - np.eye(T.shape[0])  # Generator matrix

    W = _set_up_linear_system(L, A, B)
    """Assemble right hand side r for linear system"""
    """Equation (I+II)"""
    r = np.zeros(T.shape[0])
    """Equation (III)"""
    r[list(B)] = 1.0

    u = solve(W, r) if not sparse.issparse(W) else spsolve(W, r)
    return u


def backward_committor(T, A, B, mu=None):
    r"""Backward committor between given sets.

    The backward committor u(x) between sets A and B is the
    probability for the chain starting in x to have come from A last
    rather than from B.

    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix
    A : array_like
        List of integer state labels for set A
    B : array_like
        List of integer state labels for set B
    mu : (M, ) ndarray (optional)
        Stationary vector

    Returns
    -------
    u : (M, ) ndarray
        Vector of forward committor probabilities

    Notes
    -----
    The forward committor is a solution to the following
    boundary-value problem

    .. math::

        \sum_j K_{ij} \pi_{j} u_{j}=0    for i in X\(A u B) (I)
                                  u_{i}=1    for i \in A        (II)
                                  u_{i}=0    for i \in B        (III)

    with adjoint of the generator matrix K=(D_pi(P-I))'.

    """
    X = set(range(T.shape[0]))
    A = set(A)
    B = set(B)
    AB = A.intersection(B)
    if len(AB) > 0:
        raise ValueError("Sets A and B have to be disjoint")
    if mu is None:
        mu = stationary_distribution(T)
    if sparse.issparse(T):
        L = T - sparse.eye(T.shape[0], T.shape[0])
        D = sparse.diags([mu, ], [0, ])
        K = (D.dot(L)).T
    else:
        K = np.transpose(mu[:, np.newaxis] * (T - np.eye(T.shape[0])))

    """Assemble left-hand side W for linear system"""
    W = _set_up_linear_system(K, A, B)
    """Assemble right-hand side r for linear system"""
    """Equation (I)+(III)"""
    r = np.zeros(T.shape[0])
    """Equation (II)"""
    r[list(A)] = 1.0

    u = solve(W, r) if not sparse.issparse(W) else spsolve(W, r)

    return u
