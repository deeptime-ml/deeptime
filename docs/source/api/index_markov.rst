.. _ref-markov-hmm:
.. currentmodule:: deeptime.markov

===============================================================================
deeptime.markov
===============================================================================

The *markov* package contains algorithms which can be used to estimate markov state models and apply
analysis tools like PCCA+, TPT, bayesian sampling for confidence intervals.

Estimators
==========
.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    msm.MaximumLikelihoodMSM
    msm.BayesianMSM


Models
======
.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    msm.MarkovStateModel
    msm.MarkovStateModelCollection
    msm.BayesianPosterior

Analysis tools
==============
.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    pcca
    reactive_flux
    MembershipsChapmanKolmogorovValidator

With output models

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    PCCAModel
    ReactiveFlux


Utilities and alternatives
==========================

Transition counting
-------------------
An alternative to estimating Markov state models directly from discrete timeseries is to first estimate (and
potentially subselect) a count matrix and then use that for estimation.

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    TransitionCountEstimator
    TransitionCountModel

Special MSM estimators and models
---------------------------------
.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    msm.OOMReweightedMSM
    msm.KoopmanReweightedMSM

    msm.AugmentedMSMEstimator
    msm.AugmentedMSM
