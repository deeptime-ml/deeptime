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
    spd_inv
    spd_inv_split
    spd_inv_sqrt
    ZeroRankError
"""
from .utils import drop_nan_rows, is_diagonal_matrix, is_square_matrix
from .eigen import eigs, eig_corr, sort_eigs, spd_eig, spd_inv, spd_inv_split, spd_inv_sqrt, ZeroRankError
