r"""
.. currentmodule: deeptime.data

===============================================================================
Deterministic datasets
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    abc_flow
    bickley_jet
    lorenz_system
    position_based_fluids


===============================================================================
Stochastic datasets
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    tmatrix_metropolis1d
    birth_death_chain
    ornstein_uhlenbeck
    double_well_2d
    double_well_discrete
    triple_well_1d
    triple_well_2d
    quadruple_well
    quadruple_well_asymmetric
    time_dependent_quintuple_well
    prinz_potential
    ellipsoids
    sqrt_model
    swissroll_model
    drunkards_walk


===============================================================================
Custom systems
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    custom_sde
    custom_ode

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

    TimeDependentSystem
    TimeIndependentSystem
    CustomSystem
"""

from ._systems import CustomSystem, TimeIndependentSystem, TimeDependentSystem
from ._datasets import double_well_discrete, ellipsoids, position_based_fluids, drunkards_walk, bickley_jet, \
    birth_death_chain, tmatrix_metropolis1d, sqrt_model, quadruple_well, triple_well_2d, abc_flow, ornstein_uhlenbeck, \
    triple_well_1d, quadruple_well_asymmetric, double_well_2d, swissroll_model, prinz_potential, \
    time_dependent_quintuple_well, lorenz_system
from ._datasets import custom_sde, custom_ode
from ._double_well import DoubleWellDiscrete
from ._ellipsoids import Ellipsoids
from ._pbf_simulator import PBFSimulator
from ._drunkards_walk_simulator import DrunkardsWalk
from ._bickley_simulator import BickleyJet
from ._birth_death_chain import BirthDeathChain
