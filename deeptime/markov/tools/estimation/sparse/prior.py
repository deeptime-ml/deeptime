r"""This module provides functions for computation of prior count matrices

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>
"""

import numpy as np

from scipy.sparse import coo_matrix, issparse, csc_matrix


def prior_neighbor(C, alpha=0.001):
    r"""Neighbor prior of strength alpha for the given count matrix.

    Prior is defined by
        b_ij = alpha  if Z_ij+Z_ji > 0
        b_ij = 0      else

    Parameters
    ----------
    C : (M, M) scipy.sparse matrix
        Count matrix
    alpha : float (optional)
        Value of prior counts

    Returns
    -------
    B : (M, M) scipy.sparse matrix
        Prior count matrix

    """
    C_sym = C + C.transpose()
    C_sym = C_sym.tocoo()
    data = C_sym.data
    row = C_sym.row
    col = C_sym.col

    data_B = alpha * np.ones_like(data)
    B = coo_matrix((data_B, (row, col)))
    return B


def prior_const(C, alpha=0.001):
    """Constant prior of strength alpha.

    Prior is defined via

        b_ij=alpha for all i,j

    Parameters
    ----------
    C : (M, M) ndarray or scipy.sparse matrix
        Count matrix
    alpha : float (optional)
        Value of prior counts

    Returns
    -------
    B : (M, M) ndarray
        Prior count matrix

    """
    B = alpha * np.ones(C.shape)
    return B


def prior_rev(C, alpha=-1.0):
    r"""Prior counts for sampling of reversible transition
    matrices.

    Prior is defined as

    b_ij= alpha if i<=j
    b_ij=0         else

    The reversible prior adds -1 to the upper triagular part of
    the given count matrix. This prior respects the fact that
    for a reversible transition matrix the degrees of freedom
    correspond essentially to the upper, respectively the lower
    triangular part of the matrix.

    Parameters
    ----------
    C : (M, M) ndarray or scipy.sparse matrix
        Count matrix
    alpha : float (optional)
        Value of prior counts

    Returns
    -------
    B : (M, M) ndarray
        Matrix of prior counts

    """
    ind = np.triu_indices(C.shape[0])
    if issparse(C):
        alphas = np.empty(len(ind[0]), dtype=np.float64)
        alphas.fill(alpha)
        B = csc_matrix((alphas, ind), shape=C.shape)
    else:
        B = np.zeros(C.shape, dtype=np.float64)
        B[ind] = alpha
    return B
