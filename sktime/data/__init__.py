# This file is part of scikit-time
#
# Copyright (c) 2020 AI4Science Group, Freie Universitaet Berlin (GER)
#
# scikit-time is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


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
    TimeSeriesDataSet

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

from .util import timeshifted_split, TimeSeriesDataset
from .datasets import double_well_discrete, ellipsoids, position_based_fluids, drunkards_walk
