
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

'''
Created on Jan 13, 2014

@author: noe
'''

import numpy as np


def transition_matrix_non_reversible(C):
    r"""
    Estimates a non-reversible transition matrix from count matrix C

    T_ij = c_ij / c_i where c_i = sum_j c_ij

    Parameters
    ----------
    C: ndarray, shape (n,n)
        count matrix

    Returns
    -------
    T: Estimated transition matrix

    """
    # multiply by 1.0 to make sure we're not doing integer division
    rowsums = 1.0 * np.sum(C, axis=1)
    if np.min(rowsums) <= 0:
        raise ValueError(
            "Transition matrix has row sum of " + str(np.min(rowsums)) + ". Must have strictly positive row sums.")
    return np.divide(C, rowsums[:, np.newaxis])


def transition_matrix_reversible_pisym(C, return_statdist=False, **kwargs):
    r"""
    Estimates reversible transition matrix as follows:

    ..:math:
        p_{ij} = c_{ij} / c_i where c_i = sum_j c_{ij}
        \pi_j = \sum_j \pi_i p_{ij}
        x_{ij} = \pi_i p_{ij} + \pi_j p_{ji}
        p^{rev}_{ij} = x_{ij} / x_i where x_i = sum_j x_{ij}

    In words: takes the nonreversible transition matrix estimate, uses its
    stationary distribution to compute an equilibrium correlation matrix,
    symmetrizes that correlation matrix and then normalizes to the reversible
    transition matrix estimate.

    Parameters
    ----------
    C: ndarray, shape (n,n)
        count matrix

    Returns
    -------
    T: Estimated transition matrix

    """
    # nonreversible estimate
    T_nonrev = transition_matrix_non_reversible(C)
    from ...analysis import stationary_distribution
    pi = stationary_distribution(T_nonrev)
    # correlation matrix
    X = pi[:, None] * T_nonrev
    X = X.T + X
    # result
    T_rev = X / X.sum(axis=1)[:, None]
    if return_statdist:
        #np.testing.assert_allclose(pi, stationary_distribution(T_rev))
        #np.testing.assert_allclose(T_rev.T.dot(pi), pi)
        return T_rev, pi
    return T_rev
