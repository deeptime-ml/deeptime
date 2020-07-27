
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

r"""Dense implementation of hitting probabilities

.. moduleauthor:: F.Noe <frank DOT noe AT fu-berlin DOT de>

"""

import numpy as np


def hitting_probability(P, target):
    """
    Computes the hitting probabilities for all states to the target states.

    The hitting probability of state i to set A is defined as the minimal,
    non-negative solution of:

    .. math::
        h_i^A &= 1                    \:\:\:\:  i\in A \\
        h_i^A &= \sum_j p_{ij} h_i^A  \:\:\:\:  i \notin A

    Returns
    =======
    h : ndarray(n)
        a vector with hitting probabilities
    """
    if hasattr(target, "__len__"):
        target = np.array(target)
    else:
        target = np.array([target])
    # target size
    n = np.shape(P)[0]
    # nontarget
    nontarget = np.array(list(set(range(n)) - set(target)), dtype=int)
    # stable states
    stable = np.where(np.isclose(np.diag(P), 1) == True)[0]
    # everything else
    origin = np.array(list(set(nontarget) - set(stable)), dtype=int)
    # solve hitting probability problem (P-I)x = -b
    A = P[origin, :][:, origin] - np.eye((len(origin)))
    b = np.sum(-P[origin, :][:, target], axis=1)
    x = np.linalg.solve(A, b)
    # fill up full solution with 0's for stable states and 1's for target
    xfull = np.ones((n))
    xfull[origin] = x
    xfull[target] = 1
    xfull[stable] = 0

    return xfull
