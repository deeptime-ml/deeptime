
# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# MSMTools is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

r"""This module provides dense implementations for the computation of
expectation values for a given transition matrix.

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import numpy as np

from .decomposition import rdl_decomposition
from .stationary_vector import stationary_distribution


def expected_counts(p0, T, n):
    r"""Compute expected transition counts for Markov chain after n steps.

    Expected counts are computed according to ..math::

    E[C_{ij}^{(n)}]=\sum_{k=0}^{n-1} (p_0^T T^{k})_{i} p_{ij}

    For N<=M, the sum is computed via successive summation of the
    following matrix vector products, p_1^t=p_0^t
    T,...,p_n^t=P_{n-1}^t T. For n>M, the sum is computed using the
    eigenvalue decomposition of T and applying the expression for a
    finite geometric series to each of the eigenvalues.

    Parameters
    ----------
    p0 : (M,) ndarray
        Starting (probability) vector of the chain.
    T : (M, M) ndarray
        Transition matrix of the chain.
    n : int
        Number of steps to take from initial state.

    Returns
    --------
    EC : (M, M) ndarray
        Expected value for transition counts after n steps.

    """
    M = T.shape[0]
    if n <= M:
        return ec_matrix_vector(p0, T, n)
    else:
        return ec_geometric_series(p0, T, n)


def expected_counts_stationary(T, n, mu=None):
    r"""Expected transition counts for Markov chain in equilibrium.

    Since mu is stationary for T we have

    .. math::

        E(C^{(N)})=N diag(mu)*T.

    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix.
    n : int
        Number of steps for chain.
    mu : (M,) ndarray (optional)
        Stationary distribution for T. If mu is not specified it will be
        computed via diagonalization of T.

    Returns
    -------
    EC : numpy array, shape=(n,n)
        Expected value for transition counts after a propagation of n steps.

    """
    if n <= 0:
        EC = np.zeros(T.shape)
        return EC
    else:
        if mu is None:
            mu = stationary_distribution(T)
        EC = n * mu[:, np.newaxis] * T
        return EC


def geometric_series(q, n):
    """
    Compute finite geometric series.

                                \frac{1-q^{n+1}}{1-q}   q \neq 1
        \sum_{k=0}^{n} q^{k}=
                                 n+1                     q  = 1

    Parameters
    ----------
    q : array-like
        The common ratio of the geometric series.
    n : int
        The number of terms in the finite series.

    Returns
    -------
    s : float or ndarray
        The value of the finite series.

    """
    q = np.asarray(q)
    if n < 0:
        raise ValueError('Finite geometric series is only defined for n>=0.')
    else:
        """q is scalar"""
        if q.ndim == 0:
            if q == 1:
                s = (n + 1) * 1.0
                return s
            else:
                s = (1.0 - q ** (n + 1)) / (1.0 - q)
                return s
        """q is ndarray"""
        s = np.zeros(np.shape(q), dtype=q.dtype)
        """All elements with value q=1"""
        ind = (q == 1.0)
        """For q=1 the sum has the value s=n+1"""
        s[ind] = (n + 1) * 1.0
        """All elements with value q\neq 1"""
        not_ind = np.logical_not(ind)
        s[not_ind] = (1.0 - q[not_ind] ** (n + 1)) / (1.0 - q[not_ind])
        return s


def ec_matrix_vector(p0, T, n):
    r"""Compute expected transition counts for Markov chain after n
    steps.

    Expected counts are computed according to ..math::

    E[C_{ij}^{(n)}]=\sum_{k=0}^{n-1} (p_0^t T^{k})_{i} p_{ij}

    The sum is computed via successive summation of the following
    matrix vector products, p_1^t=p_0^t T,...,p_n^t=P_{n-1}^t T.

    Such a direct approach can become prohibetively expensive for
    large n.  In this case the sum can be computed more efficiently
    using an eigenvalue decomposition of T and applying the closed
    form expression for a finite geometric series to each of the
    eigenvalues.

    Parameters
    ----------
    p0 : (M,) ndarray
        Starting (probability) vector of the chain.
    T : (M, M) ndarray
        Transition matrix of the chain.
    n : int
        Number of steps to take from initial state.

    Returns
    --------
    EC : (M, M) ndarray
        Expected value for transition counts after N steps.

    """
    if (n <= 0):
        EC = np.zeros(T.shape)
        return EC
    else:
        """Probability vector after (k=0) propagations"""
        p_k = 1.0 * p0
        """Sum of vectors after (k=0) propagations"""
        p_sum = 1.0 * p_k
        for k in range(n - 1):
            """Propagate one step p_{k} -> p_{k+1}"""
            p_k = np.dot(p_k, T)
            """Update sum"""
            p_sum += p_k
        """Expected counts"""
        EC = p_sum[:, np.newaxis] * T
        return EC


def ec_geometric_series(p0, T, n):
    r"""Compute expected transition counts for Markov chain after n
    steps.

    Expected counts are computed according to ..math::

    E[C_{ij}^{(n)}]=\sum_{k=0}^{n-1} (p_0^t T^{k})_{i} p_{ij}

    The sum is computed using the eigenvalue decomposition of T and
    applying the expression for a finite geometric series to each of
    the eigenvalues.

    For small n the computation of the eigenvalue decomposition can be
    much more expensive than a direct computation. In this case it is
    beneficial to compute the expected counts using successively
    computed matrix vector products p_1^t=p_0^t T, ... as increments.

    Parameters
    ----------
    p0 : (M,) ndarray
        Starting (probability) vector of the chain.
    T : (M, M) ndarray
        Transition matrix of the chain.
    n : int
        Number of steps to take from initial state.

    Returns
    --------
    EC : (M, M) ndarray
        Expected value for transition counts after N steps.

    """
    if (n <= 0):
        EC = np.zeros(T.shape)
        return EC
    else:
        R, D, L = rdl_decomposition(T)
        w = np.diagonal(D)
        L = np.transpose(L)

        D_sum = np.diag(geometric_series(w, n - 1))
        T_sum = np.dot(np.dot(R, D_sum), np.conjugate(np.transpose(L)))
        p_sum = np.dot(p0, T_sum)
        EC = p_sum[:, np.newaxis] * T
        """Truncate imginary part - which is zero, but we want real
        return values"""
        EC = EC.real
        return EC
