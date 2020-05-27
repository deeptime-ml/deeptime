==================================
Markov state models and estimation
==================================

Here we introduce several types of markov state models as well as analysis tools related to them. At the very core,
are a stochastic model describing chains of events where the state of one particular point in time only depends on
the state prior to it, i.e., considering the chain of events :math:`(\ldots,s_{t-2}, s_{t-1}, s_t)` with a set of
possible states :math:`S`, the probability of encountering a particular state :math:`s_{t+1}\in S` is a conditional
probability on :math:`s_t\in S`.
These conditional probabilities are often described as so-called
*transition matrix* :math:`P\in \mathbb{R}^{n\times n}`, where :math:`n = |S|`, the number of states. Then it is

.. math::  P_{ij} = \mathbb{P}(s_i \mid s_j)

the probability of transitioning to state :math:`s_i` given one is currently in state :math:`s_j`. This also means that
it is a row-stochastic matrix, i.e.,

.. math:: \sum_{j=1}^n P_{ij} = 1 \quad\forall i=1,\ldots,n.

In the following, we introduce the core object, the :class:`MarkovStateModel <sktime.markov.msm.MarkovStateModel>`,
as well as a variety of estimators.

.. toctree::
    :maxdepth: 2

    notebooks/msm
