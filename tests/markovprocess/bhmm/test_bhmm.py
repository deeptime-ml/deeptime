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

import unittest
from os import pardir
from os.path import abspath, join

import numpy as np

from sktime.markovprocess import bhmm
from sktime.markovprocess.bhmm.output_models.discrete import DiscreteOutputModel
from sktime.util import confidence_interval


class TestBHMM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # load observations
        testfile = abspath(join(abspath(__file__), pardir))
        testfile = join(testfile, 'data')
        testfile = join(testfile, '2well_traj_100K.dat')
        obs = np.loadtxt(testfile, dtype=np.int32)

        # hidden states
        cls.n_states = 2
        # samples
        cls.nsamples = 100

        # EM with lag 10
        lag = 10
        cls.hmm_lag10 = bhmm.estimate_hmm([obs], cls.n_states, lag=lag, output='discrete')
        # BHMM
        cls.sampled_hmm_lag10 = bhmm.bayesian_hmm([obs[::lag]], cls.hmm_lag10, nsample=cls.nsamples).fetch_model()

    def test_output_model(self):
        assert isinstance(self.sampled_hmm_lag10.prior.output_model, DiscreteOutputModel)
        assert all(isinstance(s.output_model, DiscreteOutputModel) for s in self.sampled_hmm_lag10)

    def test_reversible(self):
        assert self.sampled_hmm_lag10.prior.is_reversible
        assert all(s.is_reversible for s in self.sampled_hmm_lag10)

    def test_stationary(self):
        assert not self.sampled_hmm_lag10.prior.is_stationary
        assert all(not s.is_stationary for s in self.sampled_hmm_lag10)

    def test_lag(self):
        assert self.sampled_hmm_lag10.prior.lag == 10
        assert all(s.lag == 10 for s in self.sampled_hmm_lag10)

    def test_n_states(self):
        assert self.sampled_hmm_lag10.prior.n_states == 2
        assert all(s.n_states == 2 for s in self.sampled_hmm_lag10)

    def test_transition_matrix_samples(self):
        Psamples = np.array([s.transition_matrix for s in self.sampled_hmm_lag10])
        # shape
        assert np.array_equal(Psamples.shape, (self.nsamples, self.n_states, self.n_states))
        # consistency
        import msmtools.analysis as msmana
        for P in Psamples:
            assert msmana.is_transition_matrix(P)
            assert msmana.is_reversible(P)

    def test_transition_matrix_stats(self):
        import msmtools.analysis as msmana
        Psamples = np.array([s.transition_matrix for s in self.sampled_hmm_lag10])

        # mean
        Pmean = Psamples.mean(axis=0)
        # test shape and consistency
        assert np.array_equal(Pmean.shape, (self.n_states, self.n_states))
        assert msmana.is_transition_matrix(Pmean)
        # std
        Pstd = Psamples.std(axis=0)
        # test shape
        assert np.array_equal(Pstd.shape, (self.n_states, self.n_states))
        # conf
        L, R = confidence_interval(Psamples)
        # test shape
        assert np.array_equal(L.shape, (self.n_states, self.n_states))
        assert np.array_equal(R.shape, (self.n_states, self.n_states))
        # test consistency
        assert np.all(L <= Pmean)
        assert np.all(R >= Pmean)

    def test_eigenvalues_samples(self):
        samples = np.array([s.eigenvalues for s in self.sampled_hmm_lag10])
        # shape
        assert np.array_equal(samples.shape, (self.nsamples, self.n_states))
        # consistency
        for ev in samples:
            assert np.isclose(ev[0], 1)
            assert np.all(ev[1:] < 1.0)

    def test_eigenvalues_stats(self):
        samples = np.array([s.eigenvalues for s in self.sampled_hmm_lag10])
        # mean
        mean = samples.mean(axis=0)
        # test shape and consistency
        assert np.array_equal(mean.shape, (self.n_states,))
        assert np.isclose(mean[0], 1)
        assert np.all(mean[1:] < 1.0)
        # std
        std = samples.std(axis=0)
        # test shape
        assert np.array_equal(std.shape, (self.n_states,))
        # conf
        L, R = confidence_interval(samples)
        # test shape
        assert np.array_equal(L.shape, (self.n_states,))
        assert np.array_equal(R.shape, (self.n_states,))
        # test consistency
        assert np.all(L <= mean)
        assert np.all(R >= mean)

    def test_eigenvectors_left_samples(self):
        samples = np.array([s.eigenvectors_left for s in self.sampled_hmm_lag10])
        # shape
        assert np.array_equal(samples.shape, (self.nsamples, self.n_states, self.n_states))
        # consistency
        for evec in samples:
            assert np.sign(evec[0, 0]) == np.sign(evec[0, 1])
            assert np.sign(evec[1, 0]) != np.sign(evec[1, 1])

    def test_eigenvectors_left_stats(self):
        samples = np.array([s.eigenvectors_left for s in self.sampled_hmm_lag10])

        # mean
        mean = samples.mean(axis=0)
        # test shape and consistency
        assert np.array_equal(mean.shape, (self.n_states, self.n_states))
        assert np.sign(mean[0, 0]) == np.sign(mean[0, 1])
        assert np.sign(mean[1, 0]) != np.sign(mean[1, 1])
        # std
        std = samples.std(axis=0)
        # test shape
        assert np.array_equal(std.shape, (self.n_states, self.n_states))
        # conf
        L, R = confidence_interval(samples)
        # test shape
        assert np.array_equal(L.shape, (self.n_states, self.n_states))
        assert np.array_equal(R.shape, (self.n_states, self.n_states))
        # test consistency
        assert np.all(L <= mean)
        assert np.all(R >= mean)

    def test_eigenvectors_right_samples(self):
        samples = np.array([s.eigenvectors_right for s in self.sampled_hmm_lag10])
        # shape
        assert np.array_equal(samples.shape, (self.nsamples, self.n_states, self.n_states))
        # consistency
        for evec in samples:
            assert np.sign(evec[0, 0]) == np.sign(evec[1, 0])
            assert np.sign(evec[0, 1]) != np.sign(evec[1, 1])

    def test_eigenvectors_right_stats(self):
        samples = np.array([s.eigenvectors_right for s in self.sampled_hmm_lag10])

        # mean
        mean = samples.mean(axis=0)
        # test shape and consistency
        assert np.array_equal(mean.shape, (self.n_states, self.n_states))
        assert np.sign(mean[0, 0]) == np.sign(mean[1, 0])
        assert np.sign(mean[0, 1]) != np.sign(mean[1, 1])
        # std
        std = samples.std(axis=0)
        # test shape
        assert np.array_equal(std.shape, (self.n_states, self.n_states))
        # conf
        L, R = confidence_interval(samples)
        # test shape
        assert np.array_equal(L.shape, (self.n_states, self.n_states))
        assert np.array_equal(R.shape, (self.n_states, self.n_states))
        # test consistency
        assert np.all(L <= mean)
        assert np.all(R >= mean)

    def test_initial_distribution_samples(self):
        samples = np.array([s.stationary_distribution for s in self.sampled_hmm_lag10])
        # shape
        assert np.array_equal(samples.shape, (self.nsamples, self.n_states))
        # consistency
        for mu in samples:
            assert np.isclose(mu.sum(), 1.0)
            assert np.all(mu > 0.0)

    def test_initial_distribution_stats(self):
        samples = np.array([s.stationary_distribution for s in self.sampled_hmm_lag10])

        # mean
        mean = samples.mean(axis=0)
        # test shape and consistency
        assert np.array_equal(mean.shape, (self.n_states,))
        assert np.isclose(mean.sum(), 1.0)
        assert np.all(mean > 0.0)
        assert np.max(np.abs(mean[0] - mean[1])) < 0.05
        # std
        std = samples.std(axis=0)
        # test shape
        assert np.array_equal(std.shape, (self.n_states,))
        # conf
        L, R = confidence_interval(samples)
        # test shape
        assert np.array_equal(L.shape, (self.n_states,))
        assert np.array_equal(R.shape, (self.n_states,))
        # test consistency
        assert np.all(L <= mean)
        assert np.all(R >= mean)

    def test_lifetimes_samples(self):
        samples = np.array([s.lifetimes for s in self.sampled_hmm_lag10])
        # shape
        assert np.array_equal(samples.shape, (self.nsamples, self.n_states))
        # consistency
        for l in samples:
            assert np.all(l > 0.0)

    def test_lifetimes_stats(self):
        samples = np.array([s.lifetimes for s in self.sampled_hmm_lag10])
        # mean
        mean = np.mean(samples, axis=0)
        # test shape and consistency
        assert np.array_equal(mean.shape, (self.n_states,))
        assert np.all(mean > 0.0)
        # std
        std = np.std(samples, axis=0)
        # test shape
        assert np.array_equal(std.shape, (self.n_states,))
        # conf
        L, R = confidence_interval(samples)
        # test shape
        assert np.array_equal(L.shape, (self.n_states,))
        assert np.array_equal(R.shape, (self.n_states,))
        # test consistency
        assert np.all(L <= mean)
        assert np.all(R >= mean)

    def test_timescales_samples(self):
        samples = np.array([s.timescales for s in self.sampled_hmm_lag10.samples])
        # shape
        assert np.array_equal(samples.shape, (self.nsamples, self.n_states - 1))
        # consistency
        for l in samples:
            assert np.all(l > 0.0)

    def test_timescales_stats(self):
        samples = np.array([s.timescales for s in self.sampled_hmm_lag10])
        # mean
        mean = samples.mean(axis=0)
        # test shape and consistency
        assert np.array_equal(mean.shape, (self.n_states - 1,))
        assert np.all(mean > 0.0)
        # std
        std = samples.std(axis=0)
        # test shape
        assert np.array_equal(std.shape, (self.n_states - 1,))
        # conf
        L, R = confidence_interval(samples)
        # test shape
        assert np.array_equal(L.shape, (self.n_states - 1,))
        assert np.array_equal(R.shape, (self.n_states - 1,))
        # test consistency
        assert np.all(L <= mean)
        assert np.all(R >= mean)

    # TODO: these tests can be made compact because they are almost the same. can define general functions for testing
    # TODO: samples and stats, only need to implement consistency check individually.


class TestCornerCase(unittest.TestCase):
    def test_no_except(self):
        obs = [np.array([0, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0], dtype=int),
               np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0], dtype=int)
               ]
        lag = 1
        n_states = 2
        nsamples = 2
        hmm_lag10 = bhmm.estimate_hmm(obs, n_states, lag=lag, output='discrete')
        # BHMM
        sampled_hmm_lag10 = bhmm.bayesian_hmm(obs[::lag], hmm_lag10, nsample=nsamples)


if __name__ == "__main__":
    unittest.main()
