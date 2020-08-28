from scipy.sparse import issparse


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
    """Row slicing"""
    if issparse(M):
        C_cc = M.tocsr()
    else:
        C_cc = M
    C_cc = C_cc[sel, :]

    """Column slicing"""
    if issparse(M):
        C_cc = C_cc.tocsc()
    C_cc = C_cc[:, sel]

    if issparse(M):
        return C_cc.tocoo()

    return C_cc
