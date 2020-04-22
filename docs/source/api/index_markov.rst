.. _ref-markov:

sktime.markov
=============

The *markov* package contains algorithms which can be used to estimate (hidden) markov state models and apply
analysis tools like PCCA+, TPT, bayesian sampling for confidence intervals.

.. automodule:: sktime.markov

.. toctree::
   :maxdepth: 1

sktime.markov.msm
-----------------

Package containing tools for estimation and analysis of Markov state models.

.. automodule:: sktime.markov.msm

.. toctree::
   :maxdepth: 1

sktime.markov.hmm
-----------------

Package containing tools for estimation and analysis of hidden Markov state models.

They consist out of a hidden state :class:`MSM <sktime.markov.msm.MarkovStateModel>` which holds information
on how hidden states can transition between one another and an
:class:`OutputModel <sktime.markov.hmm.OutputModel>`, which maps hidden states to discrete observable states in
case of an  :class:`DiscreteOutputModel <sktime.markov.hmm.DiscreteOutputModel>` or to continuous observables
in case of an :class:`GaussianOutputModel <sktime.markov.hmm.GaussianOutputModel>`.

.. automodule:: sktime.markov.hmm

.. toctree::
   :maxdepth: 1
