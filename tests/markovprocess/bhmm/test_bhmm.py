
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
import numpy as np
import bhmm
from os.path import abspath, join
from os import pardir


class TestBHMM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # load observations
        testfile = abspath(join(abspath(__file__), pardir))
        testfile = join(testfile, 'data')
        testfile = join(testfile, '2well_traj_100K.dat')
        obs = np.loadtxt(testfile, dtype=int)

        # don't print
        bhmm.config.verbose = False
        # hidden states
        cls.nstates = 2
        # samples
        cls.nsamples = 100

        # EM with lag 10
        lag = 10
        cls.hmm_lag10 = bhmm.estimate_hmm([obs], cls.nstates, lag=lag, output='discrete')
        # BHMM
        cls.sampled_hmm_lag10 = bhmm.bayesian_hmm([obs[::lag]], cls.hmm_lag10, nsample=cls.nsamples)

    def test_output_model(self):
        from bhmm import DiscreteOutputModel
        assert isinstance(self.sampled_hmm_lag10.output_model, DiscreteOutputModel)

    def test_reversible(self):
        assert self.sampled_hmm_lag10.is_reversible

    def test_stationary(self):
        assert not self.sampled_hmm_lag10.is_stationary

    def test_lag(self):
        assert self.sampled_hmm_lag10.lag == 10

    def test_nstates(self):
        assert self.sampled_hmm_lag10.nstates == 2

    def test_transition_matrix_samples(self):
        Psamples = self.sampled_hmm_lag10.transition_matrix_samples
        # shape
        assert np.array_equal(Psamples.shape, (self.nsamples, self.nstates, self.nstates))
        # consistency
        import msmtools.analysis as msmana
        for P in Psamples:
            assert msmana.is_transition_matrix(P)
            assert msmana.is_reversible(P)

    def test_transition_matrix_stats(self):
        import msmtools.analysis as msmana
        # mean
        Pmean = self.sampled_hmm_lag10.transition_matrix_mean
        # test shape and consistency
        assert np.array_equal(Pmean.shape, (self.nstates, self.nstates))
        assert msmana.is_transition_matrix(Pmean)
        # std
        Pstd = self.sampled_hmm_lag10.transition_matrix_std
        # test shape
        assert np.array_equal(Pstd.shape, (self.nstates, self.nstates))
        # conf
        L, R = self.sampled_hmm_lag10.transition_matrix_conf
        # test shape
        assert np.array_equal(L.shape, (self.nstates, self.nstates))
        assert np.array_equal(R.shape, (self.nstates, self.nstates))
        # test consistency
        assert np.all(L <= Pmean)
        assert np.all(R >= Pmean)

    def test_eigenvalues_samples(self):
        samples = self.sampled_hmm_lag10.eigenvalues_samples
        # shape
        assert np.array_equal(samples.shape, (self.nsamples, self.nstates))
        # consistency
        for ev in samples:
            assert np.isclose(ev[0], 1)
            assert np.all(ev[1:] < 1.0)

    def test_eigenvalues_stats(self):
        # mean
        mean = self.sampled_hmm_lag10.eigenvalues_mean
        # test shape and consistency
        assert np.array_equal(mean.shape, (self.nstates,))
        assert np.isclose(mean[0], 1)
        assert np.all(mean[1:] < 1.0)
        # std
        std = self.sampled_hmm_lag10.eigenvalues_std
        # test shape
        assert np.array_equal(std.shape, (self.nstates,))
        # conf
        L, R = self.sampled_hmm_lag10.eigenvalues_conf
        # test shape
        assert np.array_equal(L.shape, (self.nstates,))
        assert np.array_equal(R.shape, (self.nstates,))
        # test consistency
        assert np.all(L <= mean)
        assert np.all(R >= mean)

    def test_eigenvectors_left_samples(self):
        samples = self.sampled_hmm_lag10.eigenvectors_left_samples
        # shape
        assert np.array_equal(samples.shape, (self.nsamples, self.nstates, self.nstates))
        # consistency
        for evec in samples:
            assert np.sign(evec[0, 0]) == np.sign(evec[0, 1])
            assert np.sign(evec[1, 0]) != np.sign(evec[1, 1])

    def test_eigenvectors_left_stats(self):
        # mean
        mean = self.sampled_hmm_lag10.eigenvectors_left_mean
        # test shape and consistency
        assert np.array_equal(mean.shape, (self.nstates, self.nstates))
        assert np.sign(mean[0, 0]) == np.sign(mean[0, 1])
        assert np.sign(mean[1, 0]) != np.sign(mean[1, 1])
        # std
        std = self.sampled_hmm_lag10.eigenvectors_left_std
        # test shape
        assert np.array_equal(std.shape, (self.nstates, self.nstates))
        # conf
        L, R = self.sampled_hmm_lag10.eigenvectors_left_conf
        # test shape
        assert np.array_equal(L.shape, (self.nstates, self.nstates))
        assert np.array_equal(R.shape, (self.nstates, self.nstates))
        # test consistency
        assert np.all(L <= mean)
        assert np.all(R >= mean)

    def test_eigenvectors_right_samples(self):
        samples = self.sampled_hmm_lag10.eigenvectors_right_samples
        # shape
        assert np.array_equal(samples.shape, (self.nsamples, self.nstates, self.nstates))
        # consistency
        for evec in samples:
            assert np.sign(evec[0, 0]) == np.sign(evec[1, 0])
            assert np.sign(evec[0, 1]) != np.sign(evec[1, 1])

    def test_eigenvectors_right_stats(self):
        # mean
        mean = self.sampled_hmm_lag10.eigenvectors_right_mean
        # test shape and consistency
        assert np.array_equal(mean.shape, (self.nstates, self.nstates))
        assert np.sign(mean[0, 0]) == np.sign(mean[1, 0])
        assert np.sign(mean[0, 1]) != np.sign(mean[1, 1])
        # std
        std = self.sampled_hmm_lag10.eigenvectors_right_std
        # test shape
        assert np.array_equal(std.shape, (self.nstates, self.nstates))
        # conf
        L, R = self.sampled_hmm_lag10.eigenvectors_right_conf
        # test shape
        assert np.array_equal(L.shape, (self.nstates, self.nstates))
        assert np.array_equal(R.shape, (self.nstates, self.nstates))
        # test consistency
        assert np.all(L <= mean)
        assert np.all(R >= mean)

    def test_initial_distribution_samples(self):
        samples = self.sampled_hmm_lag10.initial_distribution_samples
        # shape
        assert np.array_equal(samples.shape, (self.nsamples, self.nstates))
        # consistency
        for mu in samples:
            assert np.isclose(mu.sum(), 1.0)
            assert np.all(mu > 0.0)

    def test_initial_distribution_stats(self):
        # mean
        mean = self.sampled_hmm_lag10.initial_distribution_mean
        # test shape and consistency
        assert np.array_equal(mean.shape, (self.nstates, ))
        assert np.isclose(mean.sum(), 1.0)
        assert np.all(mean > 0.0)
        assert np.max(np.abs(mean[0]-mean[1])) < 0.05
        # std
        std = self.sampled_hmm_lag10.initial_distribution_std
        # test shape
        assert np.array_equal(std.shape, (self.nstates, ))
        # conf
        L, R = self.sampled_hmm_lag10.initial_distribution_conf
        # test shape
        assert np.array_equal(L.shape, (self.nstates, ))
        assert np.array_equal(R.shape, (self.nstates, ))
        # test consistency
        assert np.all(L <= mean)
        assert np.all(R >= mean)

    def test_lifetimes_samples(self):
        samples = self.sampled_hmm_lag10.lifetimes_samples
        # shape
        assert np.array_equal(samples.shape, (self.nsamples, self.nstates))
        # consistency
        for l in samples:
            assert np.all(l > 0.0)

    def test_lifetimes_stats(self):
        # mean
        mean = self.sampled_hmm_lag10.lifetimes_mean
        # test shape and consistency
        assert np.array_equal(mean.shape, (self.nstates, ))
        assert np.all(mean > 0.0)
        # std
        std = self.sampled_hmm_lag10.lifetimes_std
        # test shape
        assert np.array_equal(std.shape, (self.nstates, ))
        # conf
        L, R = self.sampled_hmm_lag10.lifetimes_conf
        # test shape
        assert np.array_equal(L.shape, (self.nstates, ))
        assert np.array_equal(R.shape, (self.nstates, ))
        # test consistency
        assert np.all(L <= mean)
        assert np.all(R >= mean)

    def test_timescales_samples(self):
        samples = self.sampled_hmm_lag10.timescales_samples
        # shape
        assert np.array_equal(samples.shape, (self.nsamples, self.nstates-1))
        # consistency
        for l in samples:
            assert np.all(l > 0.0)

    def test_timescales_stats(self):
        # mean
        mean = self.sampled_hmm_lag10.timescales_mean
        # test shape and consistency
        assert np.array_equal(mean.shape, (self.nstates-1, ))
        assert np.all(mean > 0.0)
        # std
        std = self.sampled_hmm_lag10.timescales_std
        # test shape
        assert np.array_equal(std.shape, (self.nstates-1, ))
        # conf
        L, R = self.sampled_hmm_lag10.timescales_conf
        # test shape
        assert np.array_equal(L.shape, (self.nstates-1, ))
        assert np.array_equal(R.shape, (self.nstates-1, ))
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
        nstates = 2
        nsamples = 2
        hmm_lag10 = bhmm.estimate_hmm(obs, nstates, lag=lag, output='discrete')
        # BHMM
        sampled_hmm_lag10 = bhmm.bayesian_hmm(obs[::lag], hmm_lag10, nsample=nsamples)


if __name__=="__main__":
    unittest.main()
