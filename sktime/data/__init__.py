r"""
.. currentmodule: sktime.data

===============================================================================
API
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    double_well_discrete
    ellipsoids
    position_based_fluids
    drunkards_walk

===============================================================================
Utilities
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    timeshifted_split

===============================================================================
Implementations
===============================================================================

.. autosummary::
    :toctree: generated/impl/
    :template: class_nomodule.rst

    double_well_dataset.DoubleWellDiscrete
    ellipsoids_dataset.Ellipsoids
    pbf_simulator.PBFSimulator
    drunkards_walk_simulator.DrunkardsWalk
"""

from .util import timeshifted_split
from .datasets import double_well_discrete, ellipsoids, position_based_fluids, drunkards_walk
