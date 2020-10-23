r"""This module provides functions for the computation of forward and
backward comittors using sparse linear algebra.

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import numpy as np

from scipy.sparse import eye, coo_matrix, diags
from scipy.sparse.linalg import spsolve

from .stationary_vector import stationary_distribution


def forward_committor(T, A, B):
    r"""Forward committor between given sets.

    The forward committor u(x) between sets A and B is the probability
    for the chain starting in x to reach B before reaching A.

    Parameters
    ----------
    T : (M, M) scipy.sparse matrix
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
    notAB = X.difference(A).difference(B)
    if len(AB) > 0:
        raise ValueError("Sets A and B have to be disjoint")
    L = T - eye(T.shape[0], T.shape[0])

    """Assemble left hand-side W for linear system"""
    """Equation (I)"""
    W = 1.0 * L

    """Equation (II)"""
    W = W.todok()
    W[list(A), :] = 0.0
    W.tocsr()
    W = W + coo_matrix((np.ones(len(A)), (list(A), list(A))), shape=W.shape).tocsr()

    """Equation (III)"""
    W = W.todok()
    W[list(B), :] = 0.0
    W.tocsr()
    W = W + coo_matrix((np.ones(len(B)), (list(B), list(B))), shape=W.shape).tocsr()

    """Assemble right hand side r for linear system"""
    """Equation (I+II)"""
    r = np.zeros(T.shape[0])
    """Equation (III)"""
    r[list(B)] = 1.0

    u = spsolve(W, r)
    return u


def backward_committor(T, A, B):
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
    notAB = X.difference(A).difference(B)
    if len(AB) > 0:
        raise ValueError("Sets A and B have to be disjoint")
    pi = stationary_distribution(T)
    L = T - eye(T.shape[0], T.shape[0])
    D = diags([pi, ], [0, ])
    K = (D.dot(L)).T

    """Assemble left-hand side W for linear system"""
    """Equation (I)"""
    W = 1.0 * K

    """Equation (II)"""
    W = W.todok()
    W[list(A), :] = 0.0
    W.tocsr()
    W = W + coo_matrix((np.ones(len(A)), (list(A), list(A))), shape=W.shape).tocsr()

    """Equation (III)"""
    W = W.todok()
    W[list(B), :] = 0.0
    W.tocsr()
    W = W + coo_matrix((np.ones(len(B)), (list(B), list(B))), shape=W.shape).tocsr()

    """Assemble right-hand side r for linear system"""
    """Equation (I)+(III)"""
    r = np.zeros(T.shape[0])
    """Equation (II)"""
    r[list(A)] = 1.0

    u = spsolve(W, r)

    return u
