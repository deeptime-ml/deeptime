r"""

==============
MSM estimation
==============

.. currentmodule:: deeptime.markov.tools.estimation

This module (:mod:`deeptime.markov.tools.estimation`) contains utility functions dealing with MSM estimation from data.

Countmatrix
===========

.. autosummary::
   :toctree: generated/

   count_matrix - estimate count matrix from discrete trajectories

Connectivity
============

.. autosummary::
   :toctree: generated/

   connected_sets - Find connected subsets
   largest_connected_set - Find largest connected set
   largest_connected_submatrix - Count matrix on largest connected set
   is_connected - Test count matrix connectivity

Estimation
==========

.. autosummary::
   :toctree: generated/

   transition_matrix - Estimate transition matrix
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
