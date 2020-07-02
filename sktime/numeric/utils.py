import numpy as _np


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
    r""" Checks whether a provided matrix is a diagonal matrix, i.e., :math:`A = \mathrm{diag}(a_1,\ldots,\a_n)`.

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
    arr : ndarray
        The array to check.

    Returns
    -------
    is_square_matrix : bool
        Whether the array is a square matrix.
    """
    return arr.ndim == 2 and arr.shape[0] == arr.shape[1]
