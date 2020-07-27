
# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# MSMTools is free software: you can redistribute it and/or modify
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

===============================================================
dtraj - Discrete trajectories functions (:mod:`msmtools.dtraj`)
===============================================================

.. currentmodule:: msmtools.dtraj

Discrete trajectory io
======================

.. autosummary::
   :toctree: generated/

   read_discrete_trajectory - read microstate trajectory from ascii file
   read_dtraj
   write_discrete_trajectory - write microstate trajectory to ascii file
   write_dtraj
   load_discrete_trajectory - read microstate trajectory from biqqnary file
   load_dtraj
   save_discrete_trajectory -  write microstate trajectory to binary file
   save_dtraj

Simple statistics
=================

.. autosummary::
   :toctree: generated/

   count_states
   visited_set
   number_of_states
   index_states

Sampling trajectory indexes
===========================

.. autosummary::
   :toctree: generated/

   sample_indexes_by_distribution
   sample_indexes_by_state
   sample_indexes_by_sequence

"""

from .api import *
