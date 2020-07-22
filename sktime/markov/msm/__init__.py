# This file is part of scikit-time
#
# Copyright (c) 2020 AI4Science Group, Freie Universitaet Berlin (GER)
#
# scikit-time is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


r"""
.. currentmodule: sktime.markov.msm

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
from ...util import QuantityStatistics

# set up null handler
logging.getLogger(__name__).addHandler(logging.NullHandler())
