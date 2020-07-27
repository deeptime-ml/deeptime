
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

r"""This module provides functions for the coputation of stationary
vectors of stochastic matrices

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import numpy as np
from scipy.linalg import eig, lu_factor, lu_solve


def backward_iteration(A, mu, x0, tol=1e-14, maxiter=100):
    r"""Find eigenvector to approximate eigenvalue via backward iteration.

    Parameters
    ----------
    A : (N, N) ndarray
        Matrix for which eigenvector is desired
    mu : float
        Approximate eigenvalue for desired eigenvector
    x0 : (N, ) ndarray
        Initial guess for eigenvector
    tol : float
        Tolerace parameter for termination of iteration

    Returns
    -------
    x : (N, ) ndarray
        Eigenvector to approximate eigenvalue mu

    """
    T = A - mu * np.eye(A.shape[0])
    """LU-factor of T"""
    lupiv = lu_factor(T)
    """Starting iterate with ||y_0||=1"""
    r0 = 1.0 / np.linalg.norm(x0)
    y0 = x0 * r0
    """Local variables for inverse iteration"""
    y = 1.0 * y0
    r = 1.0 * r0
    for i in range(maxiter):
        x = lu_solve(lupiv, y)
        r = 1.0 / np.linalg.norm(x)
        y = x * r
        if r <= tol:
            return y
    msg = "Failed to converge after %d iterations, residuum is %e" % (maxiter, r)
    raise RuntimeError(msg)


def stationary_distribution_from_backward_iteration(P, eps=1e-15):
    r"""Fast computation of the stationary vector using backward
    iteration.

    Parameters
    ----------
    P : (M, M) ndarray
        Transition matrix
    eps : float (optional)
        Perturbation parameter for the true eigenvalue.

    Returns
    -------
    pi : (M,) ndarray
        Stationary vector

    """
    A = np.transpose(P)
    mu = 1.0 - eps
    x0 = np.ones(P.shape[0])
    y = backward_iteration(A, mu, x0)
    pi = y / y.sum()
    return pi


def stationary_distribution_from_eigenvector(T):
    r"""Compute stationary distribution of stochastic matrix T.

    The stationary distribution is the left eigenvector corresponding to the
    non-degenerate eigenvalue :math: `\lambda=1`.

    Input:
    ------
    T : numpy array, shape(d,d)
        Transition matrix (stochastic matrix).

    Returns
    -------
    mu : numpy array, shape(d,)
        Vector of stationary probabilities.

    """
    val, L = eig(T, left=True, right=False)

    """ Sorted eigenvalues and left and right eigenvectors. """
    perm = np.argsort(val)[::-1]

    val = val[perm]
    L = L[:, perm]
    """ Make sure that stationary distribution is non-negative and l1-normalized """
    nu = np.abs(L[:, 0])
    mu = nu / np.sum(nu)
    return mu


def stationary_distribution(T):
    r"""Compute stationary distribution of stochastic matrix T.

    Chooses the fastest applicable algorithm automatically

    Input:
    ------
    T : numpy array, shape(d,d)
        Transition matrix (stochastic matrix).

    Returns
    -------
    mu : numpy array, shape(d,)
        Vector of stationary probabilities.

    """
    fallback = False
    try:
        mu = stationary_distribution_from_backward_iteration(T)
        if np.any(mu < 0):  # numerical problem, fall back to more robust algorithm.
            fallback=True
    except RuntimeError:
        fallback = True

    if fallback:
        mu = stationary_distribution_from_eigenvector(T)
        if np.any(mu < 0):  # still? Then set to 0 and renormalize
            mu = np.maximum(mu, 0.0)
            mu /= mu.sum()

    return mu
