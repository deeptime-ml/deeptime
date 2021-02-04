===================
Markov state models
===================

Here we introduce several types of markov state models as well as analysis tools related to them. At the very core,
are a stochastic model describing chains of events where the state of one particular point in time only depends on
the state prior to it, i.e., considering the chain of events :math:`(\ldots,X_{t-2}, X_{t-1}, X_t)` with a set of
possible states :math:`S`, the probability of encountering a particular state :math:`X_{t+1}\in S` is a conditional
probability on :math:`X_t\in S`.

A great deal is written about MSMs in the literature, so we omit many crucial discussions here.
The 2018 review by Husic and Pande :footcite:`husic2018markov` is a good place to start for a high-level
discussion of Markov state models
and a chronology of their development in the context of molecular kinetics. Figure 3 is particularly helpful for
understanding the many "flavors" of MSM analyses developed.
A comprehensive overview of the mathematics was presented by Prinz et al :footcite:`prinz2011markov`,
including the MLE estimator used in Maximum Likelhood MSMs. This content is also covered
in Chapter 4 of a useful book on Markov state models :footcite:`bowman2013introduction`, which is a valuable
resource for many aspects of Markov state modeling (see book Figure 1.1).

The standard formulation - which is also employed here - assumes that :math:`S` is discrete and of finite cardinality.
This means that when related back to continuous-space processes, these discrete states represent a Voronoi tessellation
of state space and can be obtained via indicator functions.

These conditional probabilities are often described as so-called
*transition matrix* :math:`P\in \mathbb{R}^{n\times n}`, where :math:`n = |S|`, the number of states.
Assuming that :math:`S` is represented by the enumeration :math:`\{1,\ldots,n\}`, it is

.. math::  P_{ij} = \mathbb{P}(X_{t+1}=j \mid X_t = i)\quad\forall t,

i.e., the probability of transitioning to state :math:`i` given one is currently in state :math:`j`.
This also means that :math:`P` is a row-stochastic matrix

.. math:: \sum_{j=1}^n P_{ij} = 1 \quad\forall i=1,\ldots,n.

If a markov state model is available, interesting dynamical quantities can be computed, e.g., mean first passage times
and fluxes between (sets of) states :footcite:`metzner2009transition`, timescales, metastable decompositions
of markov states :footcite:`roblitz2013fuzzy`.

The goal of the :py:mod:`deeptime.markov` package is to provide tools to estimate and analyze markov state
models from discrete-state timeseries data. If the data's domain is not discrete,
`clustering <notebooks/clustering.ipynb>`__ can be employed to assign each frame to a state.

In the following, we introduce the core object, the :class:`MarkovStateModel <deeptime.markov.msm.MarkovStateModel>`,
as well as a variety of estimators.

When estimating a MSM from time series data, it is important to collect statistics
over the encountered state transitions. This is covered in `transition counting <notebooks/transition-counting.ipynb>`__.

.. toctree::
    :maxdepth: 1

    notebooks/transition-counting
    notebooks/mlmsm
    notebooks/pcca
    notebooks/tpt

Furthermore, deeptime implements :class:`Augmented Markov models <deeptime.markov.msm.AugmentedMSMEstimator>`
:footcite:`olsson2017combining` which can be used when experimental data is available, as well as
:class:`Observable Operator Model MSMs <deeptime.markov.msm.OOMReweightedMSM>` :footcite:`nuske2017markov` which is
an unbiased estimator for the MSM transition matrix that corrects for the effect of starting out of equilibrium,
even when short lag times are used.

.. rubric:: References

.. footbibliography::
