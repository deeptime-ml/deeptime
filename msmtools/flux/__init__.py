
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

====================================================================
flux - Reactive flux an transition pathways (:mod:`msmtools.flux`)
====================================================================

.. currentmodule:: msmtools.flux

This module contains functions to compute reactive flux networks and
find dominant reaction pathways in such networks.

TPT-object
==========

.. autosummary::
   :toctree: generated/

   tpt
   ReactiveFlux

Reactive flux
=============

.. autosummary::
   :toctree: generated/

   flux_matrix - TPT flux network
   to_netflux - Netflux from gross flux
   flux_production - Net flux-production for all states
   flux_producers
   flux_consumers
   coarsegrain

Reaction rates and fluxes
=========================

.. autosummary::
   :toctree: generated/

   total_flux
   rate
   mfpt


Pathway decomposition
=====================

.. autosummary::
   :toctree: generated/

   pathways

"""

from .api import *
from .reactive_flux import *
