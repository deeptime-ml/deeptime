import numbers

import numpy as np


def ensure_ndarray(arr, shape: tuple = None, ndim: int = None, dtype=None, size=None) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=dtype)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"Shape of provided array was {arr.shape} != {shape}")
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"ndim of provided array was {arr.ndim} != {ndim}")
    if size is not None and np.size(arr) != size:
        raise ValueError(f"size of provided array was {np.size(arr)} != {size}")
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype)
    return arr


def submatrix(M, sel):
    """Returns a submatrix of the quadratic matrix M, given by the selected columns and row
    Parameters
    ----------
    M : ndarray(n,n)
        symmetric matrix
    sel : int-array
        selection of rows and columns. Element i,j will be selected if both are in sel.
    Returns
    -------
    S : ndarray(m,m)
        submatrix with m=len(sel)
    """
    assert len(M.shape) == 2, 'M is not a matrix'
    assert M.shape[0] == M.shape[1], 'M is not quadratic'
    import scipy.sparse
    """Row slicing"""
    if scipy.sparse.issparse(M):
        C_cc = M.tocsr()
    else:
        C_cc = M
    C_cc = C_cc[sel, :]

    """Column slicing"""
    if scipy.sparse.issparse(M):
        C_cc = C_cc.tocsc()
    C_cc = C_cc[:, sel]

    if scipy.sparse.issparse(M):
        return C_cc.tocoo()

    return C_cc


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
    elif len(args) == 1:
        return args[0]
    elif len(args) == 2:
        return np.dot(args[0], args[1])
    else:
        return np.dot(args[0], mdot(*args[1:]))


def ensure_dtraj_list(dtrajs):
    """Makes sure that dtrajs is a list of discrete trajectories (array of int)"""
    if len(dtrajs) > 0 and isinstance(dtrajs[0], numbers.Integral):
        return [ensure_ndarray(dtrajs, dtype=np.int32)]
    return [ensure_ndarray(t, dtype=np.int32) for t in dtrajs]
