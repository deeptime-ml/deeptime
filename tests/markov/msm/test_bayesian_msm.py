import unittest

import numpy as np

from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MarkovStateModel, BayesianPosterior, BayesianMSM
from deeptime.util.stats import confidence_interval
from tests.markov.factory import bmsm_double_well


class TestBMSMBasic(unittest.TestCase):

    def test_estimator_params(self):
        estimator = BayesianMSM(n_samples=13, n_steps=55, reversible=False,
                                stationary_distribution_constraint=np.array([0.5, 0.5]), sparse=True, confidence=0.9,
                                maxiter=5000, maxerr=1e-12)
        np.testing.assert_equal(estimator.n_samples, 13)
        np.testing.assert_equal(estimator.n_steps, 55)
        np.testing.assert_equal(estimator.reversible, False)
        np.testing.assert_equal(estimator.stationary_distribution_constraint, [0.5, 0.5])
        np.testing.assert_equal(estimator.sparse, True)
        np.testing.assert_equal(estimator.confidence, 0.9)
        np.testing.assert_equal(estimator.maxiter, 5000)
        np.testing.assert_equal(estimator.maxerr, 1e-12)
        with self.assertRaises(ValueError):
            estimator.stationary_distribution_constraint = np.array([1.1, .5])
        with self.assertRaises(ValueError):
            estimator.stationary_distribution_constraint = np.array([.5, -.1])

    def test_with_count_matrix(self):
        count_matrix = np.ones((5, 5), dtype=np.float32)
        posterior = BayesianMSM(n_samples=33).fit(count_matrix).fetch_model()
        np.testing.assert_equal(len(posterior.samples), 33)

    def test_with_count_model(self):
        dtraj = np.random.randint(0, 10, size=(10000,))
        with self.assertRaises(ValueError):
            counts = TransitionCountEstimator(lagtime=1, count_mode="sliding").fit(dtraj).fetch_model()
            BayesianMSM().fit(counts)  # fails because its not effective or sliding-effective
        counts = TransitionCountEstimator(lagtime=1, count_mode="effective").fit(dtraj).fetch_model()
        bmsm = BayesianMSM(n_samples=44).fit(counts).fetch_model()
        np.testing.assert_equal(len(bmsm.samples), 44)

        bmsm = bmsm.submodel(np.array([3, 4, 5]))
        np.testing.assert_equal(bmsm.prior.count_model.state_symbols, [3, 4, 5])
        for sample in bmsm:
            np.testing.assert_equal(sample.count_model.state_symbols, [3, 4, 5])


class TestBMSM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # hidden states
        cls.n_states = 2
        # samples
        cls.nsamples = 100

        cls.lag = 100
        cls.bmsm_rev = bmsm_double_well(lagtime=cls.lag, nsamples=cls.nsamples, reversible=True).fetch_model()
        cls.bmsm_revpi = bmsm_double_well(lagtime=cls.lag, reversible=True, constrain_to_coarse_pi=True,
                                          nsamples=cls.nsamples).fetch_model()

        assert isinstance(cls.bmsm_rev, BayesianPosterior)
        assert isinstance(cls.bmsm_revpi, BayesianPosterior)

    def test_class_hierarchy(self):
        """ensure that the composite model has the right types """
        assert isinstance(self.bmsm_rev.prior, MarkovStateModel)
        assert all(isinstance(s, MarkovStateModel) for s in self.bmsm_rev.samples)

    def test_reversible(self):
        self._reversible(self.bmsm_rev)
        self._reversible(self.bmsm_revpi)

    def _reversible(self, msm):
        assert msm.prior.reversible
        assert all(s.reversible for s in msm.samples)

    def test_lag(self):
        self._lag(self.bmsm_rev)
        self._lag(self.bmsm_revpi)

    def _lag(self, msm):
        assert msm.prior.lagtime == self.lag
        assert all(s.lagtime == self.lag for s in msm.samples)

    def test_n_states(self):
        self._n_states(self.bmsm_rev)
        self._n_states(self.bmsm_revpi)

    def _n_states(self, msm):
        assert msm.prior.n_states == self.n_states
        assert all(s.n_states == self.n_states for s in msm.samples)

    def test_transition_matrix_samples(self):
        self._transition_matrix_samples(self.bmsm_rev, given_pi=False)
        self._transition_matrix_samples(self.bmsm_revpi, given_pi=True)

    def _transition_matrix_samples(self, msm, given_pi):
        Psamples = [s.transition_matrix for s in msm.samples]
        # shape
        assert np.array_equal(np.shape(Psamples), (self.nsamples, self.n_states, self.n_states))
        # consistency
        import deeptime.markov.tools.analysis as msmana
        for P in Psamples:
            assert msmana.is_transition_matrix(P)
            try:
                assert msmana.is_reversible(P)
            except AssertionError:
                # re-do calculation msmtools just performed to get details
                from deeptime.markov.tools.analysis import stationary_distribution
                mu = stationary_distribution(P)
                X = mu[:, np.newaxis] * P
                np.testing.assert_allclose(X, np.transpose(X), atol=1e-12,
                                           err_msg="P not reversible, given_pi={}".format(given_pi))

    def test_transition_matrix_stats(self):
        self._transition_matrix_stats(self.bmsm_rev)
        self._transition_matrix_stats(self.bmsm_revpi)

    def _transition_matrix_stats(self, msm):
        import deeptime.markov.tools.analysis as msmana
        # mean
        Ps = np.array([s.transition_matrix for s in msm.samples])
        Pmean = Ps.mean(axis=0)
        # test shape and consistency
        assert np.array_equal(Pmean.shape, (self.n_states, self.n_states))
        assert msmana.is_transition_matrix(Pmean)
        # std
        Pstd = Ps.std(axis=0)
        # test shape
        assert np.array_equal(Pstd.shape, (self.n_states, self.n_states))
        # conf
        L, R = confidence_interval(Ps)
        # test shape
        assert np.array_equal(L.shape, (self.n_states, self.n_states))
        assert np.array_equal(R.shape, (self.n_states, self.n_states))
        # test consistency
        assert np.all(L <= Pmean)
        assert np.all(R >= Pmean)

    def test_eigenvalues_samples(self):
        self._eigenvalues_samples(self.bmsm_rev)
        self._eigenvalues_samples(self.bmsm_revpi)

    def _eigenvalues_samples(self, msm):
        samples = np.array([s.eigenvalues() for s in msm.samples])
        # shape
        self.assertEqual(np.shape(samples), (self.nsamples, self.n_states))
        # consistency
        for ev in samples:
            assert np.isclose(ev[0], 1)
            assert np.all(ev[1:] < 1.0)

    def test_eigenvalues_stats(self):
        self._eigenvalues_stats(self.bmsm_rev)
        self._eigenvalues_stats(self.bmsm_revpi)

    def _eigenvalues_stats(self, msm, tol=1e-12):
        # mean
        samples = np.array([s.eigenvalues() for s in msm.samples])
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
        assert np.all(L - tol <= mean)
        assert np.all(R + tol >= mean)

    def test_eigenvectors_left_samples(self):
        self._eigenvectors_left_samples(self.bmsm_rev)
        self._eigenvectors_left_samples(self.bmsm_revpi)

    def _eigenvectors_left_samples(self, msm):
        samples = np.array([s.eigenvectors_left() for s in msm.samples])
        # shape
        np.testing.assert_equal(np.shape(samples), (self.nsamples, self.n_states, self.n_states))
        # consistency
        for evec in samples:
            assert np.sign(evec[0, 0]) == np.sign(evec[0, 1])
            assert np.sign(evec[1, 0]) != np.sign(evec[1, 1])

    def test_eigenvectors_left_stats(self):
        self._eigenvectors_left_stats(self.bmsm_rev)
        self._eigenvectors_left_stats(self.bmsm_revpi)

    def _eigenvectors_left_stats(self, msm, tol=1e-12):
        # mean
        samples = np.array([s.eigenvectors_left() for s in msm.samples])
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
        assert np.all(L - tol <= mean)
        assert np.all(R + tol >= mean)

    def test_eigenvectors_right_samples(self):
        self._eigenvectors_right_samples(self.bmsm_rev)
        self._eigenvectors_right_samples(self.bmsm_revpi)

    def _eigenvectors_right_samples(self, msm):
        samples = np.array([s.eigenvectors_right() for s in msm.samples])
        # shape
        np.testing.assert_equal(np.shape(samples), (self.nsamples, self.n_states, self.n_states))
        # consistency
        for evec in samples:
            assert np.sign(evec[0, 0]) == np.sign(evec[1, 0])
            assert np.sign(evec[0, 1]) != np.sign(evec[1, 1])

    def test_eigenvectors_right_stats(self):
        self._eigenvectors_right_stats(self.bmsm_rev)
        self._eigenvectors_right_stats(self.bmsm_revpi)

    def _eigenvectors_right_stats(self, msm, tol=1e-12):
        samples = np.array([s.eigenvectors_right() for s in msm.samples])
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
        assert np.all(L - tol <= mean)
        assert np.all(R + tol >= mean)

    def test_stationary_distribution_samples(self):
        self._stationary_distribution_samples(self.bmsm_rev)

    def _stationary_distribution_samples(self, msm):
        samples = np.array([s.stationary_distribution for s in msm.samples])
        # shape
        assert np.array_equal(np.shape(samples), (self.nsamples, self.n_states))
        # consistency
        for mu in samples:
            assert np.isclose(mu.sum(), 1.0)
            assert np.all(mu > 0.0)

    def test_stationary_distribution_stats(self):
        self._stationary_distribution_stats(self.bmsm_rev)
        self._stationary_distribution_stats(self.bmsm_revpi)

    def _stationary_distribution_stats(self, msm, tol=1e-12):
        samples = np.array([s.stationary_distribution for s in msm.samples])
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
        self._timescales_samples(self.bmsm_rev)
        self._timescales_samples(self.bmsm_revpi)

    def _timescales_samples(self, msm):
        stats = msm.gather_stats(quantity='timescales', store_samples=True)
        samples = stats.samples
        # shape
        np.testing.assert_equal(np.shape(samples), (self.nsamples, self.n_states - 1))
        # consistency
        lag = msm.prior.count_model.lagtime
        for l in samples:
            assert np.all(l > 0.0)

    def test_timescales_stats(self):
        self._timescales_stats(self.bmsm_rev)
        self._timescales_stats(self.bmsm_revpi)

    def _timescales_stats(self, msm):
        stats = msm.gather_stats('timescales')
        # mean
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
