
# This file is part of BHMM (Bayesian Hidden Markov Models).
#
# Copyright (c) 2016 Frank Noe (Freie Universitaet Berlin)
# and John D. Chodera (Memorial Sloan-Kettering Cancer Center, New York)
#
# BHMM is free software: you can redistribute it and/or modify
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

from __future__ import absolute_import

# import API
from bhmm.api import *

# hmms
from bhmm.hmm.generic_hmm import HMM
from bhmm.hmm.gaussian_hmm import GaussianHMM
from bhmm.hmm.discrete_hmm import DiscreteHMM

from bhmm.hmm.generic_sampled_hmm import SampledHMM
from bhmm.hmm.gaussian_hmm import SampledGaussianHMM
from bhmm.hmm.discrete_hmm import SampledDiscreteHMM

# estimators
from bhmm.estimators.bayesian_sampling import BayesianHMMSampler as BHMM
from bhmm.estimators.maximum_likelihood import MaximumLikelihoodEstimator as MLHMM

# output models
from bhmm.output_models import OutputModel, GaussianOutputModel, DiscreteOutputModel

# other stuff
from bhmm.util import config
from bhmm.util import testsystems

from .version import get_versions
__version__ = get_versions()['version']
del get_versions
