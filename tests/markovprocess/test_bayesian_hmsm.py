# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
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

from sktime import datasets
from sktime.datasets import double_well_discrete
from sktime.markovprocess.hmm.bayesian_hmsm import BayesianHMSM, BayesianHMMPosterior
from sktime.util import confidence_interval
from tests.markovprocess.test_hmsm import estimate_hidden_markov_model


class TestBHMM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # load observations
        obs = double_well_discrete().dtraj

        # hidden states
        cls.n_states = 2
        # samples
        cls.n_samples = 100

        cls.lag = 10
        cls.est = BayesianHMSM.default(
            dtrajs=obs, n_states=cls.n_states, lagtime=cls.lag, reversible=True, n_samples=cls.n_samples
        ).fit(obs)
        # cls.est = bayesian_hidden_markov_model([obs], cls.n_states, cls.lag, reversible=True, n_samples=cls.n_samples)
        cls.bhmm = cls.est.fetch_model()
        assert isinstance(cls.bhmm, BayesianHMMPosterior)

    def test_reversible(self):
        assert self.bhmm.prior.reversible
        assert all(s.reversible for s in self.bhmm)

    def test_lag(self):
        assert self.bhmm.prior.lagtime == self.lag
        assert all(s.lagtime == self.lag for s in self.bhmm)

    def test_n_states(self):
        assert self.bhmm.prior.n_states == self.n_states
        assert all(s.n_states == self.n_states for s in self.bhmm)

    def test_transition_matrix_samples(self):
        Psamples = np.array([m.transition_matrix for m in self.bhmm])
        # shape
        assert np.array_equal(np.shape(Psamples), (self.n_samples, self.n_states, self.n_states))
        # consistency
        import msmtools.analysis as msmana
        for P in Psamples:
            assert msmana.is_transition_matrix(P)
            assert msmana.is_reversible(P)

    def test_transition_matrix_stats(self):
        stats = self.bhmm.gather_stats('transition_matrix')
        import msmtools.analysis as msmana
        # mean
        Pmean = stats.mean
        # test shape and consistency
        assert np.array_equal(Pmean.shape, (self.n_states, self.n_states))
        assert msmana.is_transition_matrix(Pmean)
        # std
        Pstd = stats.std
        # test shape
        assert np.array_equal(Pstd.shape, (self.n_states, self.n_states))
        # conf
        L, R = stats.L, stats.R
        # test shape
        assert np.array_equal(L.shape, (self.n_states, self.n_states))
        assert np.array_equal(R.shape, (self.n_states, self.n_states))
        # test consistency
        assert np.all(L <= Pmean)
        assert np.all(R >= Pmean)

    def test_eigenvalues_samples(self):
        samples = np.array([m.eigenvalues() for m in self.bhmm])
        # shape
        self.assertEqual(np.shape(samples), (self.n_samples, self.n_states))
        # consistency
        for ev in samples:
            assert np.isclose(ev[0], 1)
            assert np.all(ev[1:] < 1.0)

    def test_eigenvalues_stats(self):
        stats = self.bhmm.gather_stats('eigenvalues', k=None)
        # mean
        mean = stats.mean
        # test shape and consistency
        assert np.array_equal(mean.shape, (self.n_states,))
        assert np.isclose(mean[0], 1)
        assert np.all(mean[1:] < 1.0)
        # std
        std = stats.std
        # test shape
        assert np.array_equal(std.shape, (self.n_states,))
        # conf
        L, R = stats.L, stats.R
        # test shape
        assert np.array_equal(L.shape, (self.n_states,))
        assert np.array_equal(R.shape, (self.n_states,))
        # test consistency
        tol = 1e-12
        assert np.all(L - tol <= mean)
        assert np.all(R + tol >= mean)

    def test_eigenvectors_left_samples(self):
        samples = np.array([m.eigenvectors_left() for m in self.bhmm])
        # shape
        np.testing.assert_equal(np.shape(samples), (self.n_samples, self.n_states, self.n_states))
        # consistency
        for evec in samples:
            assert np.sign(evec[0, 0]) == np.sign(evec[0, 1])
            assert np.sign(evec[1, 0]) != np.sign(evec[1, 1])

    def test_eigenvectors_left_stats(self):
        samples = np.array([m.eigenvectors_left() for m in self.bhmm])
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
        tol = 1e-12
        assert np.all(L - tol <= mean)
        assert np.all(R + tol >= mean)

    def test_eigenvectors_right_samples(self):
        samples = np.array([m.eigenvectors_right() for m in self.bhmm])
        # shape
        np.testing.assert_equal(np.shape(samples), (self.n_samples, self.n_states, self.n_states))
        # consistency
        for evec in samples:
            assert np.sign(evec[0, 0]) == np.sign(evec[1, 0])
            assert np.sign(evec[0, 1]) != np.sign(evec[1, 1])

        # mean
        mean = samples.mean(axis=0)
        # test shape and consistency
        np.testing.assert_equal(mean.shape, (self.n_states, self.n_states))
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
        tol = 1e-12
        assert np.all(L - tol <= mean)
        assert np.all(R + tol >= mean)

    def test_stationary_distribution_samples(self):
        samples = np.array([m.stationary_distribution for m in self.bhmm])
        # shape
        assert np.array_equal(np.shape(samples), (self.n_samples, self.n_states))
        # consistency
        for mu in samples:
            assert np.isclose(mu.sum(), 1.0)
            assert np.all(mu > 0.0)

    def test_stationary_distribution_stats(self):
        samples = np.array([m.stationary_distribution for m in self.bhmm])
        tol = 1e-12
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
        assert np.all(L - tol <= mean)
        assert np.all(R + tol >= mean)

    def test_timescales_samples(self):
        samples = np.array([m.timescales() for m in self.bhmm])
        # shape
        np.testing.assert_equal(np.shape(samples), (self.n_samples, self.n_states - 1))
        # consistency
        for l in samples:
            assert np.all(l > 0.0)

    def test_timescales_stats(self):
        stats = self.bhmm.gather_stats('timescales')
        mean = stats.mean
        # test shape and consistency
        assert np.array_equal(mean.shape, (self.n_states - 1,))
        assert np.all(mean > 0.0)
        # std
        std = stats.std
        # test shape
        assert np.array_equal(std.shape, (self.n_states - 1,))
        # conf
        L, R = stats.L, stats.R
        # test shape
        assert np.array_equal(L.shape, (self.n_states - 1,))
        assert np.array_equal(R.shape, (self.n_states - 1,))
        # test consistency
        assert np.all(L <= mean)
        assert np.all(R >= mean)

    def test_lifetimes_samples(self):
        samples = np.array([m.lifetimes for m in self.bhmm])
        # shape
        assert np.array_equal(np.shape(samples), (self.n_samples, self.n_states))
        # consistency
        for l in samples:
            assert np.all(l > 0.0)

    def test_lifetimes_stats(self):
        samples = np.array([m.lifetimes for m in self.bhmm])
        # mean
        mean = samples.mean(axis=0)
        # test shape and consistency
        assert np.array_equal(mean.shape, (self.n_states,))
        assert np.all(mean > 0.0)
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

    def test_submodel_simple(self):
        # sanity check for submodel;
        dtrj = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0,
                0, 2, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0,
                1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 2, 0, 0, 1, 1, 2, 0, 1, 1, 1,
                0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0])

        h = BayesianHMSM.default(dtrj, n_states=3, lagtime=2).fit(dtrj).fetch_model()
        hs = h.submodel_largest(strong=True, connectivity_threshold=5, observe_nonempty=True, dtrajs=dtrj)

        models_to_check = [hs.prior] + hs.samples
        for i, m in enumerate(models_to_check):
            self.assertEqual(m.timescales().shape[0], 1, msg=i)
            self.assertEqual(m.stationary_distribution.shape[0], 2, msg=i)
            self.assertEqual(m.transition_matrix.shape, (2, 2), msg=i)

    # TODO: these tests can be made compact because they are almost the same. can define general functions for testing
    # TODO: samples and stats, only need to implement consistency check individually.


class TestBHMMSpecialCases(unittest.TestCase):

    def test_separate_states(self):
        dtrajs = [np.array([0, 1, 1, 1, 1, 1, 0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1]),
                  np.array([2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2]), ]
        hmm_bayes = BayesianHMSM.default(dtrajs, n_states=3, lagtime=1, separate=[0], n_samples=100)\
            .fit(dtrajs).fetch_model()
        # we expect zeros in all samples at the following indexes:
        pobs_zeros = ((0, 1, 2, 2, 2), (0, 0, 1, 2, 3))
        for i, s in enumerate(hmm_bayes.samples):
            np.testing.assert_allclose(s.observation_probabilities[pobs_zeros], 0, err_msg=i)
        for strajs in hmm_bayes.hidden_state_trajectories_samples:
            assert strajs[0][0] == 2
            assert strajs[0][6] == 2

    @unittest.skip("This tests if bhmm prior was estimated with same data as fit is called, "
                   "this is no longer checked.")
    def test_initialized_bhmm(self):
        obs = datasets.double_well_discrete().dtraj

        est, init_hmm = estimate_hidden_markov_model(obs, 2, 10, return_estimator=True)
        bay_hmm_est = BayesianHMSM(n_states=est.n_states, lagtime=init_hmm.lagtime,
                                   stride=est.stride, init_hmsm=init_hmm)
        bay_hmm_est.fit(obs)
        bay_hmm = bay_hmm_est.fetch_model()
        bay_hmm = bay_hmm.submodel_largest(dtrajs=obs)

        assert np.isclose(bay_hmm.prior.stationary_distribution.sum(), 1)
        assert all(np.isclose(m.stationary_distribution.sum(), 1) for m in bay_hmm)

        with self.assertRaises(NotImplementedError) as ctx:
            obs = np.copy(obs)
            assert obs[0] != np.min(obs)
            obs[0] = np.min(obs)
            bay_hmm_est.fit(obs)
            self.assertIn('same data', ctx.exception.message)

    def test_initialized_bhmm_newstride(self):
        obs = np.random.randint(0, 2, size=1000)

        est, init_hmm = estimate_hidden_markov_model(obs, 2, 10, return_estimator=True)
        bay_hmm = BayesianHMSM(n_states=init_hmm.n_states, lagtime=est.lagtime,
                               stride='effective', init_hmsm=init_hmm)
        bay_hmm.fit(obs)

        assert np.isclose(bay_hmm.fetch_model().prior.stationary_distribution.sum(), 1)
        assert all(np.isclose(m.stationary_distribution.sum(), 1) for m in bay_hmm.fetch_model())


if __name__ == "__main__":
    unittest.main()
