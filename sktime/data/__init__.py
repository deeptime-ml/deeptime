r"""
.. currentmodule: sktime.data

===============================================================================
API
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: base_nomodule.rst

    double_well_discrete
    ellipsoids
    position_based_fluids

===============================================================================
Utilities
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: base_nomodule.rst

    timeshifted_split

===============================================================================
Implementations
===============================================================================

.. autosummary::
    :toctree: generated/impl/
    :template: base_nomodule.rst

    double_well_dataset.DoubleWellDiscrete
    ellipsoids_dataset.Ellipsoids
    pbf_simulator.PBFSimulator
"""

from .util import timeshifted_split
from .datasets import double_well_discrete, ellipsoids, position_based_fluids
