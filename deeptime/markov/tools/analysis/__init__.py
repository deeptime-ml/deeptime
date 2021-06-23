r"""

============
MSM analysis
============

.. currentmodule:: deeptime.markov.tools.analysis

This module (:mod:`deeptime.markov.tools.analysis`) contains functions to analyze a created Markov model, which is
specified with a transition matrix T.

Validation
==========

.. autosummary::
   :toctree: generated/
   :template: class_nomodule.rst

   is_transition_matrix - Positive entries and rows sum to one
   is_rate_matrix - Nonpositive off-diagonal entries and rows sum to zero
   is_connected - Irreducible matrix
   is_reversible - Symmetric with respect to some probability vector pi

Decomposition
=============

Decomposition routines use the scipy LAPACK bindings for dense
numpy-arrays and the ARPACK bindings for scipy sparse matrices.

.. autosummary::
   :toctree: generated/
   :template: class_nomodule.rst

   stationary_distribution - Invariant vector from eigendecomposition
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
   pcca_memberships - Perron cluster center analysis
   hitting_probability

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

from ._api import is_transition_matrix, is_rate_matrix, is_connected, is_reversible
from ._api import stationary_distribution, eigenvalues, eigenvectors, rdl_decomposition, timescales, \
    timescales_from_eigenvalues
from ._api import expected_counts, expected_counts_stationary
from ._api import mfpt
from ._api import committor, pcca_memberships, hitting_probability
from ._api import fingerprint_correlation, fingerprint_relaxation, expectation, correlation, relaxation
from ._api import stationary_distribution_sensitivity, eigenvalue_sensitivity, eigenvector_sensitivity, \
    timescale_sensitivity, mfpt_sensitivity, committor_sensitivity, expectation_sensitivity
