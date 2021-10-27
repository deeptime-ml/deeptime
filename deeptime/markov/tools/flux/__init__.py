r"""
============================
Transition path theory tools
============================

.. currentmodule:: deeptime.markov.tools.flux

This module (:mod:`deeptime.markov.tools.flux`) contains functions to compute reactive flux networks and
find dominant reaction pathways in such networks.

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

from .api import flux_matrix, to_netflux, flux_production, flux_producers, flux_consumers, coarsegrain
from .api import total_flux, rate, mfpt
from .api import pathways
