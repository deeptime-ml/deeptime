
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

import numpy as np

# import decomposition
from .stationary_vector import stationary_distribution

def is_transition_matrix(T, tol=1e-10):
    """
    Tests whether T is a transition matrix

    Parameters
    ----------
    T : ndarray shape=(n, n)
        matrix to test
    tol : float
        tolerance to check with

    Returns
    -------
    Truth value : bool
        True, if all elements are in interval [0, 1]
            and each row of T sums up to 1.
        False, otherwise
    """
    if T.ndim != 2:
        return False
    if T.shape[0] != T.shape[1]:
        return False
    dim = T.shape[0]
    X = np.abs(T) - T
    x = np.sum(T, axis=1)
    return np.abs(x - np.ones(dim)).max() < dim * tol and X.max() < 2.0 * tol


def is_rate_matrix(K, tol=1e-10):
    """
    True if K is a rate matrix
    Parameters
    ----------
    K : numpy.ndarray matrix
        Matrix to check
    tol : float
        tolerance to check with

    Returns
    -------
    Truth value : bool
        True, if K negated diagonal is positive and row sums up to zero.
        False, otherwise
    """
    R = K - K.diagonal()
    off_diagonal_positive = np.allclose(R, abs(R), 0.0, atol=tol)

    row_sum = K.sum(axis=1)
    row_sum_eq_0 = np.allclose(row_sum, 0.0, atol=tol)

    return off_diagonal_positive and row_sum_eq_0


def is_reversible(T, mu=None, tol=1e-10):
    r"""
    checks whether T is reversible in terms of given stationary distribution.
    If no distribution is given, it will be calculated out of T.

    It performs following check:
    :math:`\pi_i P_{ij} = \pi_j P_{ji}`

    Parameters
    ----------
    T : numpy.ndarray matrix
        Transition matrix
    mu : numpy.ndarray vector
        stationary distribution
    tol : float
        tolerance to check with

    Returns
    -------
    Truth value : bool
        True, if T is a reversible transitition matrix
        False, otherwise
    """
    if is_transition_matrix(T, tol):
        if mu is None:
            mu = stationary_distribution(T)
        X = mu[:, np.newaxis] * T
        return np.allclose(X, np.transpose(X),  atol=tol)
    else:
        raise ValueError("given matrix is not a valid transition matrix.")
