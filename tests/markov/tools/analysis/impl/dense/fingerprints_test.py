r"""Unit test for the fingerprint module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest

import numpy as np

from deeptime.data import birth_death_chain
from tests.markov.tools.numeric import assert_allclose

from deeptime.markov.tools.analysis._decomposition import rdl_decomposition, timescales


from deeptime.markov.tools.analysis._fingerprints import fingerprint, fingerprint_correlation, fingerprint_relaxation
from deeptime.markov.tools.analysis._fingerprints import correlation_decomp, correlation_matvec, correlation
from deeptime.markov.tools.analysis._fingerprints import relaxation_decomp, relaxation_matvec, relaxation
from deeptime.markov.tools.analysis import expectation
from deeptime.markov.tools.analysis._fingerprints import propagate

################################################################################
# Fingerprints
################################################################################

class TestFingerprint(unittest.TestCase):
    def setUp(self):
        p = np.zeros(10)
        q = np.zeros(10)
        p[0:-1] = 0.5
        q[1:] = 0.5
        p[4] = 0.01
        q[6] = 0.1

        self.bdc = birth_death_chain(q, p)

        self.mu = self.bdc.stationary_distribution
        self.T = self.bdc.transition_matrix
        R, D, L = rdl_decomposition(self.T)
        self.L = L
        self.R = R
        self.ts = timescales(self.T)
        self.times = np.array([1, 5, 10, 20])

        ev = np.diagonal(D)
        self.ev_t = ev[np.newaxis, :] ** self.times[:, np.newaxis]

        self.k = 4
        self.tau = 7.5

        """Observables"""
        obs1 = np.zeros(10)
        obs1[0] = 1
        obs1[1] = 1
        obs2 = np.zeros(10)
        obs2[8] = 1
        obs2[9] = 1

        self.obs1 = obs1
        self.obs2 = obs2

        """Initial vector for relaxation"""
        w0 = np.zeros(10)
        w0[0:4] = 0.25
        self.p0 = w0

    def test_fingerprint_correlation(self):
        """Autocorrelation"""

        """k=None, tau=1"""
        acorr_amp = np.dot(self.mu * self.obs1, self.R) * np.dot(self.L, self.obs1)
        tsn, acorr_ampn = fingerprint_correlation(self.T, self.obs1)
        assert_allclose(tsn, self.ts)
        assert_allclose(acorr_ampn, acorr_amp)

        """k=None, tau=7.5"""
        tau = self.tau
        tsn, acorr_ampn = fingerprint_correlation(self.T, self.obs1, tau=tau)
        assert_allclose(tsn, tau * self.ts)
        assert_allclose(acorr_ampn, acorr_amp)

        """k=4, tau=1"""
        k = self.k
        acorr_amp = np.dot(self.mu * self.obs1, self.R[:, 0:k]) * np.dot(self.L[0:k, :], self.obs1)
        tsn, acorr_ampn = fingerprint_correlation(self.T, self.obs1, k=k)
        assert_allclose(tsn, self.ts[0:k])
        assert_allclose(acorr_ampn, acorr_amp)

        """k=4, tau=7.5"""
        tau = self.tau
        tsn, acorr_ampn = fingerprint_correlation(self.T, self.obs1, k=k, tau=tau)
        assert_allclose(tsn, tau * self.ts[0:k])
        assert_allclose(acorr_ampn, acorr_amp)

        """Cross-correlation"""

        """k=None, tau=1"""
        corr_amp = np.dot(self.mu * self.obs1, self.R) * np.dot(self.L, self.obs2)
        tsn, corr_ampn = fingerprint_correlation(self.T, self.obs1, obs2=self.obs2)
        assert_allclose(tsn, self.ts)
        assert_allclose(corr_ampn, corr_amp)

        """k=None, tau=7.5"""
        tau = self.tau
        tsn, corr_ampn = fingerprint_correlation(self.T, self.obs1, obs2=self.obs2, tau=tau)
        assert_allclose(tsn, tau * self.ts)
        assert_allclose(corr_ampn, corr_amp)

        """k=4, tau=1"""
        k = self.k
        corr_amp = np.dot(self.mu * self.obs1, self.R[:, 0:k]) * np.dot(self.L[0:k, :], self.obs2)
        tsn, corr_ampn = fingerprint_correlation(self.T, self.obs1, obs2=self.obs2, k=k)
        assert_allclose(tsn, self.ts[0:k])
        assert_allclose(corr_ampn, corr_amp)

        """k=4, tau=7.5"""
        tau = self.tau
        tsn, corr_ampn = fingerprint_correlation(self.T, self.obs1, obs2=self.obs2, k=k, tau=tau)
        assert_allclose(tsn, tau * self.ts[0:k])
        assert_allclose(corr_ampn, corr_amp)

    def test_fingerprint_relaxation(self):
        one_vec = np.ones(self.T.shape[0])

        """k=None"""
        relax_amp = np.dot(self.p0, self.R) * np.dot(self.L, self.obs1)
        tsn, relax_ampn = fingerprint_relaxation(self.T, self.p0, self.obs1)
        assert_allclose(tsn, self.ts)
        assert_allclose(relax_ampn, relax_amp)

        """k=4"""
        k = self.k
        relax_amp = np.dot(self.p0, self.R[:, 0:k]) * np.dot(self.L[0:k, :], self.obs1)
        tsn, relax_ampn = fingerprint_relaxation(self.T, self.p0, self.obs1, k=k)
        assert_allclose(tsn, self.ts[0:k])
        assert_allclose(relax_ampn, relax_amp)

    def test_fingerprint(self):
        """k=None"""
        amp = np.dot(self.p0 * self.obs1, self.R) * np.dot(self.L, self.obs2)
        tsn, ampn = fingerprint(self.T, self.obs1, obs2=self.obs2, p0=self.p0)
        assert_allclose(tsn, self.ts)
        assert_allclose(ampn, amp)

        """k=4"""
        k = self.k
        amp = np.dot(self.p0 * self.obs1, self.R[:, 0:k]) * np.dot(self.L[0:k, :], self.obs2)
        tsn, ampn = fingerprint(self.T, self.obs1, obs2=self.obs2, p0=self.p0, k=k)
        assert_allclose(tsn, self.ts[0:k])
        assert_allclose(ampn, amp)

# ==============================
# Expectation
# ==============================

class TestExpectation(unittest.TestCase):
    def setUp(self):
        p = np.zeros(10)
        q = np.zeros(10)
        p[0:-1] = 0.5
        q[1:] = 0.5
        p[4] = 0.01
        q[6] = 0.1

        self.bdc = birth_death_chain(q, p)

        self.mu = self.bdc.stationary_distribution
        self.T = self.bdc.transition_matrix

        obs1 = np.zeros(10)
        obs1[0] = 1
        obs1[1] = 1

        self.obs1 = obs1

    def test_expectation(self):
        exp = np.dot(self.mu, self.obs1)
        expn = expectation(self.T, self.obs1)
        assert_allclose(exp, expn)


################################################################################
# Correlation
################################################################################

class TestCorrelation(unittest.TestCase):
    def setUp(self):
        p = np.zeros(10)
        q = np.zeros(10)
        p[0:-1] = 0.5
        q[1:] = 0.5
        p[4] = 0.01
        q[6] = 0.1

        self.bdc = birth_death_chain(q, p)

        self.mu = self.bdc.stationary_distribution
        self.T = self.bdc.transition_matrix
        R, D, L = rdl_decomposition(self.T, norm='reversible')
        self.L = L
        self.R = R
        self.ts = timescales(self.T)
        self.times = np.array([1, 5, 10, 20, 100])

        ev = np.diagonal(D)
        self.ev_t = ev[np.newaxis, :] ** self.times[:, np.newaxis]

        self.k = 4

        obs1 = np.zeros(10)
        obs1[0] = 1
        obs1[1] = 1
        obs2 = np.zeros(10)
        obs2[8] = 1
        obs2[9] = 1

        self.obs1 = obs1
        self.obs2 = obs2
        self.one_vec = np.ones(10)

    def test_correlation_decomp(self):
        """Auto-correlation"""

        """k=None"""
        acorr_amp = np.dot(self.mu * self.obs1, self.R) * np.dot(self.L, self.obs1)
        acorr = np.dot(self.ev_t, acorr_amp)
        acorrn = correlation_decomp(self.T, self.obs1, times=self.times)
        assert_allclose(acorrn, acorr)

        """k=4"""
        k = self.k
        acorr_amp = np.dot(self.mu * self.obs1, self.R[:, 0:k]) * np.dot(self.L[0:k, :], self.obs1)
        acorr = np.dot(self.ev_t[:, 0:k], acorr_amp)
        acorrn = correlation_decomp(self.T, self.obs1, times=self.times, k=k)
        assert_allclose(acorrn, acorr)

        """Cross-correlation"""

        """k=None"""
        corr_amp = np.dot(self.mu * self.obs1, self.R) * np.dot(self.L, self.obs2)
        corr = np.dot(self.ev_t, corr_amp)
        corrn = correlation_decomp(self.T, self.obs1, obs2=self.obs2, times=self.times)
        assert_allclose(corrn, corr)

        """k=4"""
        k = self.k
        corr_amp = np.dot(self.mu * self.obs1, self.R[:, 0:k]) * np.dot(self.L[0:k, :], self.obs2)
        corr = np.dot(self.ev_t[:, 0:k], corr_amp)
        corrn = correlation_decomp(self.T, self.obs1, obs2=self.obs2, times=self.times, k=k)
        assert_allclose(corrn, corr)

    def test_correlation_matvec(self):
        """Auto-correlation"""
        acorr_amp = np.dot(self.mu * self.obs1, self.R) * np.dot(self.L, self.obs1)
        acorr = np.dot(self.ev_t, acorr_amp)
        acorrn = correlation_matvec(self.T, self.obs1, times=self.times)
        assert_allclose(acorrn, acorr)

        """Cross-correlation"""
        corr_amp = np.dot(self.mu * self.obs1, self.R) * np.dot(self.L, self.obs2)
        corr = np.dot(self.ev_t, corr_amp)
        corrn = correlation_matvec(self.T, self.obs1, obs2=self.obs2, times=self.times)
        assert_allclose(corrn, corr)

    def test_correlation(self):
        """Auto-correlation"""

        """k=None"""
        acorr_amp = np.dot(self.mu * self.obs1, self.R) * np.dot(self.L, self.obs1)
        acorr = np.dot(self.ev_t, acorr_amp)
        acorrn = correlation(self.T, self.obs1, times=self.times)
        assert_allclose(acorrn, acorr)

        """k=4"""
        k = self.k
        acorr_amp = np.dot(self.mu * self.obs1, self.R[:, 0:k]) * np.dot(self.L[0:k, :], self.obs1)
        acorr = np.dot(self.ev_t[:, 0:k], acorr_amp)
        acorrn = correlation(self.T, self.obs1, times=self.times, k=k)
        assert_allclose(acorrn, acorr)

        """Cross-correlation"""

        """k=None"""
        corr_amp = np.dot(self.mu * self.obs1, self.R) * np.dot(self.L, self.obs2)
        corr = np.dot(self.ev_t, corr_amp)
        corrn = correlation(self.T, self.obs1, obs2=self.obs2, times=self.times)
        assert_allclose(corrn, corr)

        """k=4"""
        k = self.k
        corr_amp = np.dot(self.mu * self.obs1, self.R[:, 0:k]) * np.dot(self.L[0:k, :], self.obs2)
        corr = np.dot(self.ev_t[:, 0:k], corr_amp)
        corrn = correlation(self.T, self.obs1, obs2=self.obs2, times=self.times, k=k)
        assert_allclose(corrn, corr)


################################################################################
# Relaxation
################################################################################

class TestRelaxation(unittest.TestCase):
    def setUp(self):
        p = np.zeros(10)
        q = np.zeros(10)
        p[0:-1] = 0.5
        q[1:] = 0.5
        p[4] = 0.01
        q[6] = 0.1

        self.bdc = birth_death_chain(q, p)

        self.mu = self.bdc.stationary_distribution
        self.T = self.bdc.transition_matrix

        """Test matrix-vector product against spectral decomposition"""
        R, D, L = rdl_decomposition(self.T)
        self.L = L
        self.R = R
        self.ts = timescales(self.T)
        self.times = np.array([1, 5, 10, 20, 100])

        ev = np.diagonal(D)
        self.ev_t = ev[np.newaxis, :] ** self.times[:, np.newaxis]

        self.k = 4

        """Observable"""
        obs1 = np.zeros(10)
        obs1[0] = 1
        obs1[1] = 1
        self.obs = obs1

        """Initial distribution"""
        w0 = np.zeros(10)
        w0[0:4] = 0.25
        self.p0 = w0

    def test_relaxation_decomp(self):
        """k=None"""
        relax_amp = np.dot(self.p0, self.R) * np.dot(self.L, self.obs)
        relax = np.dot(self.ev_t, relax_amp)
        relaxn = relaxation_decomp(self.T, self.p0, self.obs, times=self.times)
        assert_allclose(relaxn, relax)

        """k=4"""
        k = self.k
        relax_amp = np.dot(self.p0, self.R[:, 0:k]) * np.dot(self.L[0:k, :], self.obs)
        relax = np.dot(self.ev_t[:, 0:k], relax_amp)
        relaxn = relaxation_decomp(self.T, self.p0, self.obs, k=k, times=self.times)
        assert_allclose(relaxn, relax)

    def test_relaxation_matvec(self):
        """k=None"""
        relax_amp = np.dot(self.p0, self.R) * np.dot(self.L, self.obs)
        relax = np.dot(self.ev_t, relax_amp)
        relaxn = relaxation_matvec(self.T, self.p0, self.obs, times=self.times)
        assert_allclose(relaxn, relax)

    def test_relaxation(self):
        """k=None"""
        relax_amp = np.dot(self.p0, self.R) * np.dot(self.L, self.obs)
        relax = np.dot(self.ev_t, relax_amp)
        relaxn = relaxation(self.T, self.p0, self.obs, times=self.times)
        assert_allclose(relaxn, relax)

        """k=4"""
        k = self.k
        relax_amp = np.dot(self.p0, self.R[:, 0:k]) * np.dot(self.L[0:k, :], self.obs)
        relax = np.dot(self.ev_t[:, 0:k], relax_amp)
        relaxn = relaxation(self.T, self.p0, self.obs, k=k, times=self.times)
        assert_allclose(relaxn, relax)


################################################################################
# Helper functions
################################################################################

class TestPropagate(unittest.TestCase):
    def setUp(self):
        self.A = np.array([[0.2, 0.5, 0.3], [0.4, 0.2, 0.4], [0.0, 0.1, 0.9]])
        self.x = np.array([0.2, 0.2, 0.2])

    def test_propagate(self):
        A = self.A
        x = self.x

        yn = propagate(A, x, 1)
        y = np.dot(A, x)
        assert_allclose(yn, y)

        yn = propagate(A, x, 2)
        y = np.dot(A, np.dot(A, x))
        assert_allclose(yn, y)

        yn = propagate(A, x, 100)
        y = np.dot(np.linalg.matrix_power(A, 100), x)
        assert_allclose(yn, y)
