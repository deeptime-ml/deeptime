r"""
.. currentmodule: deeptime.data

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
    bickley_jet
    birth_death_chain
    tmatrix_metropolis1d

===============================================================================
Utilities
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    timeshifted_split
    TimeSeriesDataset
    TimeLaggedDataset

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
    bickley_simulator.BickleyJet
    birth_death_chain_dataset.BirthDeathChain
"""

from .util import timeshifted_split, TimeSeriesDataset, TimeLaggedDataset
from .datasets import double_well_discrete, ellipsoids, position_based_fluids, drunkards_walk, bickley_jet, \
    birth_death_chain, tmatrix_metropolis1d
