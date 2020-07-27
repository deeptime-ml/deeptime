import numpy as np


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
        same_data = np.allclose(diff, 0.0, rtol=rtol, atol=atol)

        return same_data
    else:
        return False
