# This file is part of scikit-time and MSMTools.
#
# Copyright (c) 2020, 2015, 2014 AI4Science Group, Freie Universitaet Berlin (GER)
#
# scikit-time and MSMTools is free software: you can redistribute it and/or modify
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

========================
MSM estimation from data
========================

.. currentmodule:: sktime.markov.tools.estimation

This module (:mod:`sktime.markov.tools.estimation`) contains utility functions dealing with MSM estimation from data.

Countmatrix
===========

.. autosummary::
   :toctree: generated/

   count_matrix - estimate count matrix from discrete trajectories
   cmatrix - estimate count matrix from discrete trajectories

Connectivity
============

.. autosummary::
   :toctree: generated/

   connected_sets - Find connected subsets
   largest_connected_set - Find largest connected set
   largest_connected_submatrix - Count matrix on largest connected set
   connected_cmatrix
   is_connected - Test count matrix connectivity

Estimation
==========

.. autosummary::
   :toctree: generated/

   transition_matrix - Estimate transition matrix
   tmatrix
   rate_matrix
   log_likelihood
   tmatrix_cov
   error_perturbation


Sampling
========

.. autosummary::
   :toctree: generated/

   tmatrix_sampler - Random sample from transition matrix posterior

Bootstrap
=========

.. autosummary::
   :toctree: generated/

   bootstrap_counts
   bootstrap_trajectories

Priors
======

.. autosummary::
   :toctree: generated/

   prior_neighbor
   prior_const
   prior_rev


"""
from .api import *
