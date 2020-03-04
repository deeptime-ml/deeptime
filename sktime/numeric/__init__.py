r"""
.. currentmodule: sktime.numeric

===============================================================================
General numerical tools
===============================================================================

.. autosummary::
    :toctree: generated/

    mdot

===============================================================================
Numerical tools for eigenvalue problems
===============================================================================

.. autosummary::
    :toctree: generated/

    eig_corr
    sort_by_norm
    spd_eig
    spd_inv
    spd_inv_split
    spd_inv_sqrt
    ZeroRankError

"""

from .utils import mdot
from .eigen import eig_corr, sort_by_norm, spd_eig, spd_inv, spd_inv_split, spd_inv_sqrt, ZeroRankError

