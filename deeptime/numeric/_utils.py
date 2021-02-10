import numpy as _np
from scipy.sparse import issparse


def is_sorted(x: _np.ndarray, order: str = 'asc'):
    r"""Check if x is sorted.

    Parameters
    ----------
    x : ndarray
        input array
    order : str, default='asc'
        One of asc (for ascending) and desc (for descending)

    Returns
    -------
    sorted : bool
        Whether array is sorted.
    """
    assert order in ('asc', 'desc')
    try:
        if _np.issubdtype(x.dtype, _np.unsignedinteger):
            # x is unsigned int array, risk of int underflow in np.diff
            x = _np.int64(x)
    except AttributeError:
        pass
    if order == 'asc':
        return (_np.diff(x) >= 0).all()
    else:
        return (_np.diff(x) <= 0).all()


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


def drop_nan_rows(arr: _np.ndarray, *args):
    r"""
    Remove rows in all inputs for which `arr` has `_np.nan` entries.

    Parameters
    ----------
    arr : numpy.ndarray
        Array whose rows are checked for nan entries.
        Any rows containing nans are removed from `arr` and all arguments
        passed via `args`.

    *args : variable length argument list of numpy.ndarray
        Additional arrays from which to remove rows.
        Each argument should have the same number of rows as `arr`.
    """
    nan_inds = _np.isnan(arr).any(axis=1)
    return (arr[~nan_inds], *[arg[~nan_inds] for arg in args])  # parenthesis not redundant in py < 3.8
