r"""
.. currentmodule: sktime.data

===============================================================================
API
===============================================================================

.. autosummary::
    :toctree: generated/

    double_well_discrete
    ellipsoids

===============================================================================
Utilities
===============================================================================

.. autosummary::
    :toctree: generated/

    timeshifted_split

===============================================================================
Implementations
===============================================================================

.. autosummary::
    :toctree: generated/impl/

    DoubleWellDiscrete
    Ellipsoids
"""

from .util import timeshifted_split
from .double_well import DoubleWellDiscrete
from .ellipsoids import Ellipsoids
from .datasets import double_well_discrete, ellipsoids
