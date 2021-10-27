.. currentmodule:: deeptime.markov.hmm

===============================================================================
deeptime.markov.hmm
===============================================================================

Package containing tools for estimation and analysis of hidden Markov state models.

They consist out of a hidden state :class:`MSM <deeptime.markov.msm.MarkovStateModel>` which holds information
on how hidden states can transition between one another and an
:class:`OutputModel <deeptime.markov.hmm.OutputModel>`, which maps hidden states to discrete observable states in
case of an  :class:`DiscreteOutputModel <deeptime.markov.hmm.DiscreteOutputModel>` or to continuous observables
in case of an :class:`GaussianOutputModel <deeptime.markov.hmm.GaussianOutputModel>`.

Estimators
==========
.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    MaximumLikelihoodHMM
    BayesianHMM

Output models
=============
.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    OutputModel
    DiscreteOutputModel
    GaussianOutputModel

Initial guess
=============
Depending on the output model there are some methods that provide initial guesses for estimation.

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    init.discrete.random_guess
    init.discrete.metastable_from_data
    init.discrete.metastable_from_msm
    init.gaussian.from_data

Models
======
.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    HiddenMarkovModel
    BayesianHMMPosterior
