r"""
.. currentmodule: deeptime.markov.msm

Maximum-likelihood MSMs (ML-MSM) and Bayesian sampling
------------------------------------------------------
.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    MaximumLikelihoodMSM
    MarkovStateModel
    MarkovStateModelCollection

    BayesianMSM
    BayesianPosterior

Observable operator model MSMs (OOMs)
-------------------------------------
.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    OOMReweightedMSM
    KoopmanReweightedMSM

Augmented markov models (AMMs)
------------------------------
.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    AugmentedMSMEstimator
    AugmentedMSM
    AMMOptimizerState
"""
import logging

from ._markov_state_model import MarkovStateModel, MarkovStateModelCollection
from ._maximum_likelihood_msm import MaximumLikelihoodMSM
from ._bayesian_msm import BayesianMSM, BayesianPosterior
from ._koopman_reweighted_msm import KoopmanReweightedMSM, OOMReweightedMSM
from ._augmented_msm import AugmentedMSMEstimator, AugmentedMSM, AMMOptimizerState

# set up null handler
logging.getLogger(__name__).addHandler(logging.NullHandler())
