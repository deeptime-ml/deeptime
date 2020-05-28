==================================
Markov state models and estimation
==================================

Here we introduce several types of markov state models as well as analysis tools related to them. At the very core,
are a stochastic model describing chains of events where the state of one particular point in time only depends on
the state prior to it, i.e., considering the chain of events :math:`(\ldots,X_{t-2}, X_{t-1}, X_t)` with a set of
possible states :math:`S`, the probability of encountering a particular state :math:`X_{t+1}\in S` is a conditional
probability on :math:`X_t\in S`.
These conditional probabilities are often described as so-called
*transition matrix* :math:`P\in \mathbb{R}^{n\times n}`, where :math:`n = |S|`, the number of states.
Assuming that :math:`S` is represented by the enumeration :math:`\{1,\ldots,n\}`, it is

.. math::  P_{ij} = \mathbb{P}(X_{t+1}=j \mid X_t = i)\quad\forall t,

i.e., the probability of transitioning to state :math:`i` given one is currently in state :math:`j`.
This also means that :math:`P` is a row-stochastic matrix

.. math:: \sum_{j=1}^n P_{ij} = 1 \quad\forall i=1,\ldots,n.

If a markov state model is available, interesting dynamical quantities can be computed, e.g., mean first passage times
and fluxes between (sets of) states :cite:`ix-msm-metzner2009transition`, timescales, metastable decompositions
of markov states :cite:`ix-msm-roblitz2013fuzzy`.

The goal of the :py:mod:`sktime.markov` package is to provide tools to estimate and analyze markov state
models from discrete-state timeseries data. If the data's domain is not discrete,
`clustering <notebooks/clustering.ipynb>`__ can be employed to assign each frame to a state.

In the following, we introduce the core object, the :class:`MarkovStateModel <sktime.markov.msm.MarkovStateModel>`,
as well as a variety of estimators. When estimating a MSM from time series data, it is important to collect statistics
over the encountered state transitions. This is covered in `transition counting <notebooks/transition-counting.ipynb>`__.

.. toctree::
    :maxdepth: 1

    notebooks/transition-counting
    notebooks/mlmsm
    notebooks/bayesian-msm
    notebooks/amm
    notebooks/oom-msm

.. rubric:: References

.. bibliography:: /references.bib
    :style: unsrt
    :filter: docname in docnames
    :keyprefix: ix-msm-
