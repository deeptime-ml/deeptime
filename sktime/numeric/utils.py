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


def allclose_sparse(A, B, rtol=1e-5, atol=1e-8):
    """
    Compares two sparse matrices in the same matter like numpy.allclose()
    Parameters
    ----------
    A : scipy.sparse matrix
        first matrix to compare
    B : scipy.sparse matrix
        second matrix to compare
    rtol : float
        relative tolerance
    atol : float
        absolute tolerance

    Returns
    -------
    True, if given matrices are equal in bounds of rtol and atol
    False, otherwise

    Notes
    -----
    If the following equation is element-wise True, then allclose returns
    True.

     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

    The above equation is not symmetric in `a` and `b`, so that
    `allclose(a, b)` might be different from `allclose(b, a)` in
    some rare cases.
    """
    A = A.tocsr()
    B = B.tocsr()

    """Shape"""
    same_shape = (A.shape == B.shape)

    """Data"""
    if same_shape:
        diff = (A - B).data
        same_data = _np.allclose(diff, 0.0, rtol=rtol, atol=atol)
        return same_data
    else:
        return False

