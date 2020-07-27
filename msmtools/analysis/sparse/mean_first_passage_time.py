
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

r"""Sparse implementation of mean first passage time computation

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>
.. moduleauthor:: C.Wehmeyer <christoph DOT wehmeyer AT fu-berlin DOT de>

"""
import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve
from .decomposition import stationary_distribution


def mfpt(T, target):
    r"""Mean first passage times to a set of target states.

    Parameters
    ----------
    T : scipy.sparse matrix
        Transition matrix.
    target : int or list of int
        Target states for mfpt calculation.

    Returns
    -------
    m_t : ndarray, shape=(n,)
         Vector of mean first passage times to target states.

    Notes
    -----
    The mean first passage time :math:`\mathbf{E}_x[T_Y]` is the expected
    hitting time of one state :math:`y` in :math:`Y` when starting in state :math:`x`.

    For a fixed target state :math:`y` it is given by

    .. math :: \mathbb{E}_x[T_y] = \left \{  \begin{array}{cc}
                                             0 & x=y \\
                                             1+\sum_{z} T_{x,z} \mathbb{E}_z[T_y] & x \neq y
                                             \end{array}  \right.

    For a set of target states :math:`Y` it is given by

    .. math :: \mathbb{E}_x[T_Y] = \left \{  \begin{array}{cc}
                                             0 & x \in Y \\
                                             1+\sum_{z} T_{x,z} \mathbb{E}_z[T_Y] & x \notin Y
                                             \end{array}  \right.

    References
    ----------
    .. [1] Hoel, P G and S C Port and C J Stone. 1972. Introduction to
        Stochastic Processes.

    Examples
    --------

    >>> from msmtools.analysis import mfpt

    >>> T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])
    >>> m_t = mfpt(T, 0)
    >>> m_t
    array([  0.,  12.,  22.])

    """
    dim = T.shape[0]
    A = eye(dim, dim) - T

    """Convert to DOK (dictionary of keys) matrix to enable
    row-slicing and assignement"""
    A = A.todok()
    D = A.diagonal()
    A[target, :] = 0.0
    D[target] = 1.0
    A.setdiag(D)
    """Convert back to CSR-format for fast sparse linear algebra"""
    A = A.tocsr()
    b = np.ones(dim)
    b[target] = 0.0
    m_t = spsolve(A, b)
    return m_t


def mfpt_between_sets(T, target, origin, mu=None):
    """Compute mean-first-passage time between subsets of state space.

    Parameters
    ----------
    T : scipy.sparse matrix
        Transition matrix.
    target : int or list of int
        Set of target states.
    origin : int or list of int
        Set of starting states.
    mu : (M,) ndarray (optional)
        The stationary distribution of the transition matrix T.

    Returns
    -------
    tXY : float
        Mean first passage time between set X and Y.

    Notes
    -----
    The mean first passage time :math:`\mathbf{E}_X[T_Y]` is the expected
    hitting time of one state :math:`y` in :math:`Y` when starting in a
    state :math:`x` in :math:`X`:

    .. math :: \mathbb{E}_X[T_Y] = \sum_{x \in X}
                \frac{\mu_x \mathbb{E}_x[T_Y]}{\sum_{z \in X} \mu_z}

    """
    if mu is None:
        mu = stationary_distribution(T)

    """Stationary distribution restriced on starting set X"""
    nuX = mu[origin]
    muX = nuX / np.sum(nuX)

    """Mean first-passage time to Y (for all possible starting states)"""
    tY = mfpt(T, target)

    """Mean first-passage time from X to Y"""
    tXY = np.dot(muX, tY[origin])
    return tXY
