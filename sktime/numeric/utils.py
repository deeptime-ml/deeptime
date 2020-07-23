# This file is part of scikit-time
#
# Copyright (c) 2020 AI4Science Group, Freie Universitaet Berlin (GER)
#
# scikit-time is free software: you can redistribute it and/or modify
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


import numpy as _np
from scipy.sparse import issparse


def mdot(*args):
    """Computes a matrix product of multiple ndarrays

    This is a convenience function to avoid constructs such as np.dot(A, np.dot(B, np.dot(C, D))) and instead
    use mdot(A, B, C, D).

    Parameters
    ----------
    *args : an arbitrarily long list of ndarrays that must be compatible for multiplication,
        i.e. args[i].shape[1] = args[i+1].shape[0].
    """
    if len(args) < 1:
        raise ValueError('need at least one argument')
    args = list(args)[::-1]
    x = args.pop()
    i = 0
    while len(args):
        y = args.pop()
        try:
            x = _np.dot(x, y)
            i += 1
        except ValueError as ve:
            raise ValueError(f'argument {i} and {i + 1} are not shape compatible:\n{ve}')
    return x


def is_diagonal_matrix(matrix: _np.ndarray) -> bool:
    r""" Checks whether a provided matrix is a diagonal matrix, i.e., :math:`A = \mathrm{diag}(a_1,\ldots, a_n)`.

    Parameters
    ----------
    matrix : ndarray
        The matrix for which this check is performed.

    Returns
    -------
    is_diagonal : bool
        True if the matrix is a diagonal matrix, otherwise False.
    """
    return _np.all(matrix == _np.diag(_np.diagonal(matrix)))


def is_square_matrix(arr: _np.ndarray) -> bool:
    r""" Determines whether an array is a square matrix. This means that ndim must be 2 and shape[0] must be equal
    to shape[1].

    Parameters
    ----------
    arr : ndarray or sparse array
        The array to check.

    Returns
    -------
    is_square_matrix : bool
        Whether the array is a square matrix.
    """
    return (issparse(arr) or isinstance(arr, _np.ndarray)) and arr.ndim == 2 and arr.shape[0] == arr.shape[1]
