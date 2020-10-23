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
    QuantityStatistics

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

from .markov_state_model import MarkovStateModel, MarkovStateModelCollection
from .maximum_likelihood_msm import MaximumLikelihoodMSM
from .bayesian_msm import BayesianMSM, BayesianPosterior
from .koopman_reweighted_msm import KoopmanReweightedMSM, OOMReweightedMSM
from .augmented_msm import AugmentedMSMEstimator, AugmentedMSM, AMMOptimizerState
from ...util.stats import QuantityStatistics

# set up null handler
logging.getLogger(__name__).addHandler(logging.NullHandler())
