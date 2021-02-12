r"""
.. currentmodule: deeptime.data

===============================================================================
API
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    double_well_discrete
    quadruple_well
    quadruple_well_asymmetric
    triple_well_2d
    triple_well_1d
    double_well_2d
    abc_flow
    ornstein_uhlenbeck
    ellipsoids
    sqrt_model
    position_based_fluids
    drunkards_walk
    bickley_jet
    birth_death_chain
    tmatrix_metropolis1d

===============================================================================
Custom systems
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    custom_sde
    custom_ode

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

    DoubleWellDiscrete
    Ellipsoids
    PBFSimulator
    DrunkardsWalk
    BickleyJet
    BirthDeathChain

"""

from ._util import timeshifted_split, TimeSeriesDataset, TimeLaggedDataset
from ._datasets import double_well_discrete, ellipsoids, position_based_fluids, drunkards_walk, bickley_jet, \
    birth_death_chain, tmatrix_metropolis1d, sqrt_model, quadruple_well, triple_well_2d, abc_flow, ornstein_uhlenbeck, \
    triple_well_1d, quadruple_well_asymmetric, double_well_2d
from ._datasets import custom_sde, custom_ode
from ._double_well import DoubleWellDiscrete
from ._ellipsoids import Ellipsoids
from ._pbf_simulator import PBFSimulator
from ._drunkards_walk_simulator import DrunkardsWalk
from ._bickley_simulator import BickleyJet
from ._birth_death_chain import BirthDeathChain
