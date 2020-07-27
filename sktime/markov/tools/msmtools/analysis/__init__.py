
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

==============================================================
analysis - MSM analysis functions (:mod:`msmtools.analysis`)
==============================================================

.. currentmodule:: msmtools.analysis

This module contains functions to analyze a created Markov model, which is
specified with a transition matrix T.

Validation
==========

.. autosummary::
   :toctree: generated/

   is_transition_matrix - Positive entries and rows sum to one
   is_tmatrix
   is_rate_matrix - Nonpositive off-diagonal entries and rows sum to zero
   is_connected - Irreducible matrix
   is_reversible - Symmetric with respect to some probability vector pi

Decomposition
=============

Decomposition routines use the scipy LAPACK bindings for dense
numpy-arrays and the ARPACK bindings for scipy sparse matrices.

.. autosummary::
   :toctree: generated/

   stationary_distribution - Invariant vector from eigendecomposition
   statdist
   eigenvalues - Spectrum via eigenvalue decomposition
   eigenvectors - Right or left eigenvectors
   rdl_decomposition - Full decomposition into eigenvalues and eigenvectors
   timescales - Implied timescales from eigenvalues

Expected counts
=================

.. autosummary::
   :toctree: generated/

   expected_counts - Count matrix expected for given initial distribution
   expected_counts_stationary - Count matrix expected for equilibrium distribution

Passage times
=============

.. autosummary::
   :toctree: generated/

   mfpt - Mean first-passage time

Committors and PCCA
===================

.. autosummary::
   :toctree: generated/

   committor - Forward and backward committor
   pcca - Perron cluster center analysis

Fingerprints
============

.. autosummary::
   :toctree: generated/

   fingerprint_correlation
   fingerprint_relaxation
   expectation - Equilibrium expectation value of an observable
   correlation
   relaxation

Sensitivity analysis
====================

.. autosummary::
   :toctree: generated/

   stationary_distribution_sensitivity
   eigenvalue_sensitivity
   timescale_sensitivity
   eigenvector_sensitivity
   mfpt_sensitivity
   committor_sensitivity
   expectation_sensitivity

"""

from .api import *
