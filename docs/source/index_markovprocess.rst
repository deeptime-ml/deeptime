.. _ref-markovprocess:

Markovprocess package (sktime.markovprocess)
============================================

The *markovprocess* package contains algorithms which can be used to estimate (hidden) markov state models and apply
analysis tools like PCCA+, TPT, bayesian sampling for confidence intervals.

.. automodule:: sktime.markovprocess

.. toctree::
   :maxdepth: 1

MSM subpackage (sktime.markovprocess.msm)
-----------------------------------------

Package containing tools for estimation and analysis of Markov state models.

.. automodule:: sktime.markovprocess.msm

.. toctree::
   :maxdepth: 1

HMSM subpackage (sktime.markovprocess.hmm)
------------------------------------------

Package containing tools for estimation and analysis of hidden Markov state models.

They consist out of a hidden state :class:`MSM <sktime.markovprocess.msm.MarkovStateModel>` which holds information
on how hidden states can transition between one another and an
:class:`OutputModel <sktime.markovprocess.hmm.OutputModel>`, which maps hidden states to discrete observable states in
case of an  :class:`DiscreteOutputModel <sktime.markovprocess.hmm.DiscreteOutputModel>` or to continuous observables
in case of an :class:`GaussianOutputModel <sktime.markovprocess.hmm.GaussianOutputModel>`.

.. automodule:: sktime.markovprocess.hmm

.. toctree::
   :maxdepth: 1
