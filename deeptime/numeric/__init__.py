r"""
.. currentmodule: deeptime.numeric

===============================================================================
General numerical tools
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst
    
    drop_nan_rows
    is_diagonal_matrix
    is_square_matrix
    schatten_norm

===============================================================================
Numerical tools for eigenvalue problems
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    eigs
    eig_corr
    sort_eigs
    spd_eig
    spd_truncated_svd
    spd_inv
    spd_inv_split
    spd_inv_sqrt
    ZeroRankError
"""
from ._utils import drop_nan_rows, is_diagonal_matrix, is_square_matrix, is_sorted, allclose_sparse
from ._eigen import eigs, eig_corr, sort_eigs, spd_eig, spd_truncated_svd, \
    spd_inv, spd_inv_split, spd_inv_sqrt, ZeroRankError
from ._norm import schatten_norm
