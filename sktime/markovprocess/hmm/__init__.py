r"""
.. currentmodule: sktime.markovprocess.hmm

Output models
-------------

.. autosummary::
    :toctree: generated/

    DiscreteOutputModel
    GaussianOutputModel


Maximum-likelihood estimation of HMMs
-------------------------------------

Making initial guesses:

.. autosummary::
    :toctree: generated/

    initial_guess_discrete_from_data
    initial_guess_discrete_from_msm

Estimation and resulting model:

.. autosummary::
    :toctree: generated/

    MaximumLikelihoodHMSM
    HiddenMarkovStateModel

Bayesian hidden markov state models
-----------------------------------

.. autosummary::
    :toctree: generated/

    BayesianHMSM
    BayesianHMMPosterior
"""

from .hmm import HiddenMarkovStateModel
from .maximum_likelihood_hmm import MaximumLikelihoodHMSM, initial_guess_discrete_from_data, \
    initial_guess_discrete_from_msm
from .bayesian_hmm import BayesianHMSM, BayesianHMMPosterior
from .output_model import DiscreteOutputModel, GaussianOutputModel
