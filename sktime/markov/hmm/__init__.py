r"""
.. currentmodule: sktime.markov.hmm

The implementations that can be found in this subpackage are based on the `BHMM <https://github.com/bhmm/bhmm>`_
software package.

Output models
-------------

.. autosummary::
    :toctree: generated/

    OutputModel
    DiscreteOutputModel
    GaussianOutputModel


Maximum-likelihood estimation of HMMs
-------------------------------------

Since the maximum-likelihood estimation (via Baum-Welch [1]_) of
:class:`HMMs <sktime.markov.hmm.HiddenMarkovStateModel>` is likely to
get stuck in local maxima, a good initial guess is important for a high-quality model. The following methods aim to
provide heuristics which can yield such an initial guess and can be found in [3]_.

For a HMM with a :class:`discrete output model <sktime.markov.hmm.DiscreteOutputModel>`, the following main
steps are involved:

1. A :class:`markov state model <sktime.markov.msm.MarkovStateModel>` is estimated from given discrete
   trajectories. This step is optional if the markov model already exists.
2. The estimated MSM transition matrix :math:`P\in\mathbb{R}^{n\times n}` is coarse-grained with PCCA+ into the
   desired number of hidden states :math:`m` and memberships :math:`M\in\mathbb{R}^{n \times m}`. The transition
   matrix is projected into the hidden state space by

   .. math::
        P_\mathrm{coarse} = (M^\top M)^{-1} M^\top P M.

.. autosummary::
    :toctree: generated/

    initial_guess_discrete_from_data
    initial_guess_discrete_from_msm

For a HMM with a :class:`gaussian output model <sktime.markov.hmm.GaussianOutputModel>`, a Gaussian mixture
model is estimated. This particular heuristic requires an installation of `scikit-learn <https://scikit-learn.org/>`_.

.. autosummary::
    :toctree: generated/

    initial_guess_gaussian_from_data

Estimation and resulting model:

.. autosummary::
    :toctree: generated/

    MaximumLikelihoodHMSM
    HiddenMarkovStateModel

Bayesian hidden markov state models
-----------------------------------
Bayesian HMMs can provide confidence estimates. They are estimated by starting from a reference HMM and then use
Gibbs sampling.

See [2]_ for a manuscript describing the theory behind using Gibbs sampling to sample from Bayesian hidden Markov model
posteriors.

.. autosummary::
    :toctree: generated/

    BayesianHMSM
    BayesianHMMPosterior

References
----------
.. [1] L. E. Baum and J. A. Egon, "An inequality with applications to statistical estimation for probabilistic
       functions of a Markov process and to a model for ecology,"
       Bull. Amer. Meteorol. Soc., vol. 73, pp. 360-363, 1967.
.. [2] Bayesian hidden Markov model analysis of single-molecule force spectroscopy: Characterizing kinetics under
       measurement uncertainty. John D. Chodera, Phillip Elms, Frank Noé, Bettina Keller, Christian M. Kaiser,
       Aaron Ewall-Wice, Susan Marqusee, Carlos Bustamante, Nina Singhal Hinrichs http://arxiv.org/abs/1108.1430
.. [3] F. Noe, H. Wu, J.-H. Prinz and N. Plattner: Projected and hidden Markov models for calculating kinetics and
       metastable states of complex molecules. J. Chem. Phys. 139, 184114 (2013)
"""

from .hmm import HiddenMarkovStateModel
from .maximum_likelihood_hmm import MaximumLikelihoodHMSM, initial_guess_discrete_from_data, \
    initial_guess_discrete_from_msm, initial_guess_gaussian_from_data
from .bayesian_hmm import BayesianHMSM, BayesianHMMPosterior
from .output_model import OutputModel, DiscreteOutputModel, GaussianOutputModel
