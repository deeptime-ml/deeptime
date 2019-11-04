
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

import numpy as np

from bhmm.hmm.generic_hmm import HMM
from bhmm.hmm.generic_sampled_hmm import SampledHMM
from bhmm.output_models.gaussian import GaussianOutputModel
from bhmm.util import config
from bhmm.util.statistics import confidence_interval_arr


class GaussianHMM(HMM, GaussianOutputModel):
    r""" Convenience access to an HMM with a Gaussian output model.

    """

    def __init__(self, hmm):
        # superclass constructors
        if not isinstance(hmm.output_model, GaussianOutputModel):
            raise TypeError('Given hmm is not a Gaussian HMM, but has an output model of type: ' +
                            str(type(hmm.output_model)))
        GaussianOutputModel.__init__(self, hmm.nstates, means=hmm.output_model.means, sigmas=hmm.output_model.sigmas)
        HMM.__init__(self, hmm.initial_distribution, hmm.transition_matrix, self, lag=hmm.lag)


class SampledGaussianHMM(GaussianHMM, SampledHMM):
    """ Sampled Gaussian HMM with a representative single point estimate and error estimates

    Parameters
    ----------
    estimated_hmm : :class:`HMM <generic_hmm.HMM>`
        Representative HMM estimate, e.g. a maximum likelihood estimate or mean HMM.
    sampled_hmms : list of :class:`HMM <generic_hmm.HMM>`
        Sampled HMMs
    conf : float, optional, default = 0.95
        confidence interval, e.g. 0.68 for 1 sigma or 0.95 for 2 sigma.

    """
    def __init__(self, estimated_hmm, sampled_hmms, conf=0.95):
        # enforce right type
        estimated_hmm = GaussianHMM(estimated_hmm)
        for i in range(len(sampled_hmms)):
            sampled_hmms[i] = GaussianHMM(sampled_hmms[i])
        # call GaussianHMM superclass constructer with estimated_hmm
        GaussianHMM.__init__(self, estimated_hmm)
        # call SampledHMM superclass constructor
        SampledHMM.__init__(self, estimated_hmm, sampled_hmms, conf=conf)

    @property
    def means_samples(self):
        r""" Samples of the Gaussian distribution means """
        res = np.empty((self.nsamples, self.nstates, self.dimension), dtype=config.dtype)
        for i in range(self.nsamples):
            for j in range(self.nstates):
                res[i, j, :] = self._sampled_hmms[i].means[j]
        return res

    @property
    def means_mean(self):
        r""" The mean of the Gaussian distribution means """
        return np.mean(self.means_samples, axis=0)

    @property
    def means_std(self):
        r""" The standard deviation of the Gaussian distribution means """
        return np.std(self.means_samples, axis=0)

    @property
    def means_conf(self):
        r""" The standard deviation of the Gaussian distribution means """
        return confidence_interval_arr(self.means_samples, conf=self._conf)

    @property
    def sigmas_samples(self):
        r""" Samples of the Gaussian distribution standard deviations """
        res = np.empty((self.nsamples, self.nstates, self.dimension), dtype=config.dtype)
        for i in range(self.nsamples):
            for j in range(self.nstates):
                res[i, j, :] = self._sampled_hmms[i].sigmas[j]
        return res

    @property
    def sigmas_mean(self):
        r""" The mean of the Gaussian distribution standard deviations """
        return np.mean(self.sigmas_samples, axis=0)

    @property
    def sigmas_std(self):
        r""" The standard deviation of the Gaussian distribution standard deviations """
        return np.std(self.sigmas_samples, axis=0)

    @property
    def sigmas_conf(self):
        r""" The standard deviation of the Gaussian distribution standard deviations """
        return confidence_interval_arr(self.sigmas_samples, conf=self._conf)
