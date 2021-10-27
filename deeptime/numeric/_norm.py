import numpy as np


def schatten_norm(mat: np.ndarray, order: float = 1, hermitian: bool = False) -> float:
    r""" Computes the r-Schatten norm

    .. math::
        \| T \|_r = \left( \sum_i \sigma_i^p (T) \right)^{1/r},

    where :math:`\sigma_i` are the singular values corresponding to matrix :math:`T`.

    Parameters
    ----------
    mat : ndarray
        input matrix
    order : float
        Order of the Schatten norm, must be :math:`\geq 1`.
    hermitian : bool, optional, default=False
        If True, the matrix is assumed to be Hermitian, which allows the norm to be computed more efficiently.

    Returns
    -------
    norm : float
        The norm.
    """
    assert order >= 1, 'Order only defined for r >= 1'
    if order == 1:
        return np.linalg.norm(mat, ord='nuc')
    elif order == 2:
        return np.linalg.norm(mat, ord='fro')
    else:
        s = np.linalg.svd(mat, compute_uv=False, hermitian=hermitian)
        return np.power(np.sum(np.power(s, order)), 1. / order)
