r"""
.. currentmodule: deeptime.markov.hmm

The implementations that can be found in this subpackage are based on the `BHMM <https://github.com/bhmm/bhmm>`_
software package.

Output models
-------------

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    OutputModel
    DiscreteOutputModel
    GaussianOutputModel


Maximum-likelihood estimation of HMMs
-------------------------------------

Since the maximum-likelihood estimation (via Baum-Welch :footcite:`baum1967inequality`) of
:class:`HMMs <deeptime.markov.hmm.HiddenMarkovModel>` is likely to
get stuck in local maxima, a good initial guess is important for a high-quality model. The following methods aim to
provide heuristics which can yield such an initial guess and can be found in :footcite:`noe2013projected`.

For a HMM with a :class:`discrete output model <deeptime.markov.hmm.DiscreteOutputModel>`, the following main
steps are involved:

1. A :class:`markov state model <deeptime.markov.msm.MarkovStateModel>` is estimated from given discrete
   trajectories. This step is optional if the markov model already exists.
2. The estimated MSM transition matrix :math:`P\in\mathbb{R}^{n\times n}` is coarse-grained with PCCA+ into the
   desired number of hidden states :math:`m` and memberships :math:`M\in\mathbb{R}^{n \times m}`. The transition
   matrix is projected into the hidden state space by

   .. math::
        P_\mathrm{coarse} = (M^\top M)^{-1} M^\top P M.

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    init.discrete.metastable_from_data
    init.discrete.metastable_from_msm

A non data-driven way to initialize a HMM is implemented by setting the transition matrix and initial distribution to
uniform, and drawing a random row-stochastic emission probability matrix:

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    init.discrete.random_guess

For a HMM with a :class:`gaussian output model <deeptime.markov.hmm.GaussianOutputModel>`, a Gaussian mixture
model is estimated. This particular heuristic requires an installation of `scikit-learn <https://scikit-learn.org/>`_.

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    init.gaussian.from_data

Estimation and resulting model:

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    MaximumLikelihoodHMM
    HiddenMarkovModel

Bayesian hidden markov state models
-----------------------------------
Bayesian HMMs can provide confidence estimates. They are estimated by starting from a reference HMM and then use
Gibbs sampling.

See :footcite:`chodera2011bayesian` for a manuscript describing the theory behind using Gibbs
sampling to sample from Bayesian hidden Markov model posteriors.

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    BayesianHMM
    BayesianHMMPosterior

Utilities
---------

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    viterbi

References
----------
.. footbibliography::
"""

from ._hidden_markov_model import HiddenMarkovModel, viterbi
from ._maximum_likelihood_hmm import MaximumLikelihoodHMM
from ._bayesian_hmm import BayesianHMM, BayesianHMMPosterior
from ._output_model import OutputModel, DiscreteOutputModel, GaussianOutputModel

from . import init
