
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
from bhmm.util import config
from bhmm.util.statistics import confidence_interval_arr


class SampledHMM(HMM):
    """ Sampled HMM with a representative single point estimate and error estimates

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
        # call superclass constructer with estimated_hmm
        HMM.__init__(self, estimated_hmm.initial_distribution, estimated_hmm.transition_matrix,
                     estimated_hmm.output_model, lag=estimated_hmm.lag)
        # save sampled HMMs to calculate statistical moments.
        self._sampled_hmms = sampled_hmms
        self._nsamples = len(sampled_hmms)
        # save confindence interval
        self._conf = conf

    def set_confidence(self, conf):
        r""" Set confidence interval """
        self._conf = conf

    @property
    def nsamples(self):
        r""" Number of samples """
        return self._nsamples

    @property
    def sampled_hmms(self):
        r""" List of sampled HMMs """
        return self._sampled_hmms

    @property
    def confidence_interval(self):
        r""" Confidence interval used """
        return self._conf

    @property
    def initial_distribution_samples(self):
        r""" Samples of the initial distribution """
        res = np.empty((self.nsamples, self.nstates), dtype=config.dtype)
        for i in range(self.nsamples):
            res[i, :] = self._sampled_hmms[i].stationary_distribution
        return res

    @property
    def initial_distribution_mean(self):
        r""" The mean of the initial distribution of the hidden states """
        return np.mean(self.initial_distribution_samples, axis=0)

    @property
    def initial_distribution_std(self):
        r""" The standard deviation of the initial distribution of the hidden states """
        return np.std(self.initial_distribution_samples, axis=0)

    @property
    def initial_distribution_conf(self):
        r""" The standard deviation of the initial distribution of the hidden states """
        return confidence_interval_arr(self.initial_distribution_samples, conf=self._conf)

    @property
    def stationary_distribution_samples(self):
        r""" Samples of the stationary distribution """
        if self.is_stationary:
            return self.initial_distribution_samples
        else:
            raise ValueError('HMM is not stationary')

    @property
    def stationary_distribution_mean(self):
        r""" The mean of the stationary distribution of the hidden states """
        return np.mean(self.stationary_distribution_samples, axis=0)

    @property
    def stationary_distribution_std(self):
        r""" The standard deviation of the stationary distribution of the hidden states """
        return np.std(self.stationary_distribution_samples, axis=0)

    @property
    def stationary_distribution_conf(self):
        r""" The standard deviation of the stationary distribution of the hidden states """
        return confidence_interval_arr(self.stationary_distribution_samples, conf=self._conf)

    @property
    def transition_matrix_samples(self):
        r""" Samples of the transition matrix """
        res = np.empty((self.nsamples, self.nstates, self.nstates), dtype=config.dtype)
        for i in range(self.nsamples):
            res[i, :, :] = self._sampled_hmms[i].transition_matrix
        return res

    @property
    def transition_matrix_mean(self):
        r""" The mean of the transition_matrix of the hidden states """
        return np.mean(self.transition_matrix_samples, axis=0)

    @property
    def transition_matrix_std(self):
        r""" The standard deviation of the transition_matrix of the hidden states """
        return np.std(self.transition_matrix_samples, axis=0)

    @property
    def transition_matrix_conf(self):
        r""" The standard deviation of the transition_matrix of the hidden states """
        return confidence_interval_arr(self.transition_matrix_samples, conf=self._conf)

    @property
    def eigenvalues_samples(self):
        r""" Samples of the eigenvalues """
        res = np.empty((self.nsamples, self.nstates), dtype=config.dtype)
        for i in range(self.nsamples):
            res[i, :] = self._sampled_hmms[i].eigenvalues
        return res

    @property
    def eigenvalues_mean(self):
        r""" The mean of the eigenvalues of the hidden states """
        return np.mean(self.eigenvalues_samples, axis=0)

    @property
    def eigenvalues_std(self):
        r""" The standard deviation of the eigenvalues of the hidden states """
        return np.std(self.eigenvalues_samples, axis=0)

    @property
    def eigenvalues_conf(self):
        r""" The standard deviation of the eigenvalues of the hidden states """
        return confidence_interval_arr(self.eigenvalues_samples, conf=self._conf)

    @property
    def eigenvectors_left_samples(self):
        r""" Samples of the left eigenvectors of the hidden transition matrix """
        res = np.empty((self.nsamples, self.nstates, self.nstates), dtype=config.dtype)
        for i in range(self.nsamples):
            res[i, :, :] = self._sampled_hmms[i].eigenvectors_left
        return res

    @property
    def eigenvectors_left_mean(self):
        r""" The mean of the left eigenvectors of the hidden transition matrix """
        return np.mean(self.eigenvectors_left_samples, axis=0)

    @property
    def eigenvectors_left_std(self):
        r""" The standard deviation of the left eigenvectors of the hidden transition matrix """
        return np.std(self.eigenvectors_left_samples, axis=0)

    @property
    def eigenvectors_left_conf(self):
        r""" The standard deviation of the left eigenvectors of the hidden transition matrix """
        return confidence_interval_arr(self.eigenvectors_left_samples, conf=self._conf)

    @property
    def eigenvectors_right_samples(self):
        r""" Samples of the right eigenvectors of the hidden transition matrix """
        res = np.empty((self.nsamples, self.nstates, self.nstates), dtype=config.dtype)
        for i in range(self.nsamples):
            res[i, :, :] = self._sampled_hmms[i].eigenvectors_right
        return res

    @property
    def eigenvectors_right_mean(self):
        r""" The mean of the right eigenvectors of the hidden transition matrix """
        return np.mean(self.eigenvectors_right_samples, axis=0)

    @property
    def eigenvectors_right_std(self):
        r""" The standard deviation of the right eigenvectors of the hidden transition matrix """
        return np.std(self.eigenvectors_right_samples, axis=0)

    @property
    def eigenvectors_right_conf(self):
        r""" The standard deviation of the right eigenvectors of the hidden transition matrix """
        return confidence_interval_arr(self.eigenvectors_right_samples, conf=self._conf)

    @property
    def timescales_samples(self):
        r""" Samples of the timescales """
        res = np.empty((self.nsamples, self.nstates-1), dtype=config.dtype)
        for i in range(self.nsamples):
            res[i, :] = self._sampled_hmms[i].timescales
        return res

    @property
    def timescales_mean(self):
        r""" The mean of the timescales of the hidden states """
        return np.mean(self.timescales_samples, axis=0)

    @property
    def timescales_std(self):
        r""" The standard deviation of the timescales of the hidden states """
        return np.std(self.timescales_samples, axis=0)

    @property
    def timescales_conf(self):
        r""" The standard deviation of the timescales of the hidden states """
        return confidence_interval_arr(self.timescales_samples, conf=self._conf)

    @property
    def lifetimes_samples(self):
        r""" Samples of the timescales """
        res = np.empty((self.nsamples, self.nstates), dtype=config.dtype)
        for i in range(self.nsamples):
            res[i, :] = self._sampled_hmms[i].lifetimes
        return res

    @property
    def lifetimes_mean(self):
        r""" The mean of the lifetimes of the hidden states """
        return np.mean(self.lifetimes_samples, axis=0)

    @property
    def lifetimes_std(self):
        r""" The standard deviation of the lifetimes of the hidden states """
        return np.std(self.lifetimes_samples, axis=0)

    @property
    def lifetimes_conf(self):
        r""" The standard deviation of the lifetimes of the hidden states """
        return confidence_interval_arr(self.lifetimes_samples, conf=self._conf)
