r"""This module provides sparse implementations for the computation of
expectation values for a given transition matrix.

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse.construct import diags

from .stationary_vector import stationary_distribution


def expected_counts(p0, T, N):
    r"""Compute expected transition counts for Markov chain after N steps.

    Expected counts are computed according to ..math::

    E[C_{ij}^{(n)}]=\sum_{k=0}^{N-1} (p_0^T T^{k})_{i} p_{ij}

    Parameters
    ----------
    p0 : (M,) ndarray
        Starting (probability) vector of the chain.
    T : (M, M) sparse matrix
        Transition matrix of the chain.
    N : int
        Number of steps to take from initial state.

    Returns
    --------
    EC : (M, M) sparse matrix
        Expected value for transition counts after N steps.

    """
    if (N <= 0):
        EC = coo_matrix(T.shape, dtype=float)
        return EC
    else:
        """Probability vector after (k=0) propagations"""
        p_k = 1.0 * p0
        """Sum of vectors after (k=0) propagations"""
        p_sum = 1.0 * p_k
        """Transpose T to use sparse dot product"""
        Tt = T.transpose()
        for k in np.arange(N - 1):
            """Propagate one step p_{k} -> p_{k+1}"""
            p_k = Tt.dot(p_k)
            """Update sum"""
            p_sum += p_k
        D_psum = diags(p_sum, 0)
        EC = D_psum.dot(T)
        return EC


def expected_counts_stationary(T, n, mu=None):
    r"""Expected transition counts for Markov chain in equilibrium.

    Since mu is stationary for T we have

    .. math::

        E(C^{(n)})=n diag(mu)*T.

    Parameters
    ----------
    T : (M, M) sparse matrix
        Transition matrix.
    n : int
        Number of steps for chain.
    mu : (M,) ndarray (optional)
        Stationary distribution for T. If mu is not specified it will be
        computed via diagonalization of T.

    Returns
    -------
    EC : (M, M) sparse matrix
        Expected value for transition counts after N steps.

    """
    if (n <= 0):
        EC = coo_matrix(T.shape, dtype=float)
        return EC
    else:
        if mu is None:
            mu = stationary_distribution(T)
        D_mu = diags(mu, 0)
        EC = n * D_mu.dot(T)
        return EC
