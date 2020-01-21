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
from msmtools import analysis as msmana

from sktime.markovprocess import cktest
from sktime.markovprocess.maximum_likelihood_hmsm import MaximumLikelihoodHMSM
from sktime.markovprocess.util import count_states
from tests.markovprocess.test_msm import estimate_markov_model


class TestMLHMM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # load observations
        from sktime.data.double_well import DoubleWellDiscrete
        obs = DoubleWellDiscrete().dtraj.copy()
        obs -= np.min(obs)  # remove empty states
        cls.obs = obs

        # hidden states
        n_states = 2

        # run with lag 1 and 10
        cls.msm_lag1 = estimate_markov_model(obs, 1, reversible=True)
        cls.hmsm_lag1 = estimate_hidden_markov_model(obs, n_states, 1, reversible=True, observe_nonempty=True)
        cls.msm_lag10 = estimate_markov_model(obs, 10, reversible=True)
        cls.hmsm_lag10 = estimate_hidden_markov_model(obs, n_states, 10, reversible=True, observe_nonempty=True)

    # =============================================================================
    # Test basic HMM properties
    # =============================================================================
    def test_reversible(self):
        assert self.hmsm_lag1.reversible
        assert self.hmsm_lag10.reversible

    def test_lag(self):
        assert self.hmsm_lag1.lagtime == 1
        assert self.hmsm_lag10.lagtime == 10

    def test_n_states(self):
        assert self.hmsm_lag1.n_states == 2
        assert self.hmsm_lag10.n_states == 2

    def test_transition_matrix(self):
        for P in [self.hmsm_lag1.transition_matrix, self.hmsm_lag1.transition_matrix]:
            assert msmana.is_transition_matrix(P)
            assert msmana.is_reversible(P)

    def test_eigenvalues(self):
        for ev in [self.hmsm_lag1.eigenvalues(), self.hmsm_lag10.eigenvalues()]:
            assert len(ev) == 2
            assert np.isclose(ev[0], 1)
            assert ev[1] < 1.0

    def test_eigenvectors_left(self):
        for evec in [self.hmsm_lag1.eigenvectors_left(), self.hmsm_lag10.eigenvectors_left()]:
            assert np.array_equal(evec.shape, (2, 2))
            assert np.sign(evec[0, 0]) == np.sign(evec[0, 1])
            assert np.sign(evec[1, 0]) != np.sign(evec[1, 1])

    def test_eigenvectors_right(self):
        for evec in [self.hmsm_lag1.eigenvectors_right(), self.hmsm_lag10.eigenvectors_right()]:
            assert np.array_equal(evec.shape, (2, 2))
            assert np.isclose(evec[0, 0], evec[1, 0])
            assert np.sign(evec[0, 1]) != np.sign(evec[1, 1])

    def test_stationary_distribution(self):
        for mu in [self.hmsm_lag1.stationary_distribution, self.hmsm_lag10.stationary_distribution]:
            # normalization
            assert np.isclose(mu.sum(), 1.0)
            # positivity
            assert np.all(mu > 0.0)
            # this data: approximately equal probability
            assert np.max(np.abs(mu[0] - mu[1])) < 0.05

    # def test_lifetimes(self):
    #     for l in [self.hmm_lag1.lifetimes, self.hmm_lag10.lifetimes]:
    #         assert len(l) == 2
    #         assert np.all(l > 0.0)
    #     # this data: lifetimes about 680
    #     assert np.max(np.abs(self.hmm_lag10.lifetimes - 680)) < 20.0

    def test_timescales(self):
        for l in [self.hmsm_lag1.timescales(), self.hmsm_lag10.timescales()]:
            assert len(l) == 1
            assert np.all(l > 0.0)
        # this data: lifetimes about 680
        assert np.abs(self.hmsm_lag10.timescales()[0] - 340) < 20.0

    # =============================================================================
    # Hidden transition matrix first passage problems
    # =============================================================================

    def test_committor(self):
        hmsm = self.hmsm_lag10
        a = 0
        b = 1
        q_forward = hmsm.committor_forward(a, b)
        assert q_forward[a] == 0
        assert q_forward[b] == 1
        q_backward = hmsm.committor_backward(a, b)
        assert q_backward[a] == 1
        assert q_backward[b] == 0
        # REVERSIBLE:
        assert np.allclose(q_forward + q_backward, np.ones(hmsm.n_states))

    def test_mfpt(self):
        hmsm = self.hmsm_lag10
        a = 0
        b = 1
        tab = hmsm.mfpt(a, b)
        tba = hmsm.mfpt(b, a)
        assert tab > 0
        assert tba > 0
        # HERE:
        err = np.minimum(np.abs(tab - 680.708752214), np.abs(tba - 699.560589099))
        assert err < 1e-6

    # =============================================================================
    # Test HMSM observable spectral properties
    # =============================================================================

    def test_n_states_obs(self):
        assert self.hmsm_lag1.n_states_obs == self.msm_lag1.n_states
        assert self.hmsm_lag10.n_states_obs == self.msm_lag10.n_states

    def test_observation_probabilities(self):
        assert np.array_equal(self.hmsm_lag1.observation_probabilities.shape, (2, self.hmsm_lag1.n_states_obs))
        assert np.allclose(self.hmsm_lag1.observation_probabilities.sum(axis=1), np.ones(2))
        assert np.array_equal(self.hmsm_lag10.observation_probabilities.shape, (2, self.hmsm_lag10.n_states_obs))
        assert np.allclose(self.hmsm_lag10.observation_probabilities.sum(axis=1), np.ones(2))

    def test_transition_matrix_obs(self):
        assert np.array_equal(self.hmsm_lag1.transition_matrix_obs().shape,
                              (self.hmsm_lag1.n_states_obs, self.hmsm_lag1.n_states_obs))
        assert np.array_equal(self.hmsm_lag10.transition_matrix_obs().shape,
                              (self.hmsm_lag10.n_states_obs, self.hmsm_lag10.n_states_obs))
        for T in [self.hmsm_lag1.transition_matrix_obs(),
                  self.hmsm_lag1.transition_matrix_obs(k=2),
                  self.hmsm_lag10.transition_matrix_obs(),
                  self.hmsm_lag10.transition_matrix_obs(k=4)]:
            assert msmana.is_transition_matrix(T)
            assert msmana.is_reversible(T)

    def test_stationary_distribution_obs(self):
        for hmsm in [self.hmsm_lag1, self.hmsm_lag10]:
            # lag 1
            sd = hmsm.stationary_distribution_obs
            assert len(sd) == hmsm.n_states_obs
            assert np.allclose(sd.sum(), 1.0)
            assert np.allclose(sd, np.dot(sd, hmsm.transition_matrix_obs()))

    def test_eigenvectors_left_obs(self):
        for hmsm in [self.hmsm_lag1, self.hmsm_lag10]:
            L = hmsm.eigenvectors_left_obs
            # shape should be right
            assert np.array_equal(L.shape, (hmsm.n_states, hmsm.n_states_obs))
            # first one should be identical to stat.dist
            l1 = L[0, :]
            err = hmsm.stationary_distribution_obs - l1
            assert np.max(np.abs(err)) < 1e-10
            # sums should be 1, 0, 0, ...
            assert np.allclose(np.sum(L[1:, :], axis=1), np.zeros(hmsm.n_states_obs - 1))
            # REVERSIBLE:
            if hmsm.reversible:
                assert np.all(np.isreal(L))

    def test_eigenvectors_right_obs(self):
        for hmsm in [self.hmsm_lag1, self.hmsm_lag10]:
            R = hmsm.eigenvectors_right_obs
            # shape should be right
            assert np.array_equal(R.shape, (hmsm.n_states_obs, hmsm.n_states))
            # should be all ones
            r1 = R[:, 0]
            assert np.allclose(r1, np.ones(hmsm.n_states_obs))
            # REVERSIBLE:
            if hmsm.reversible:
                assert np.all(np.isreal(R))

    # =============================================================================
    # Test HMSM kinetic observables
    # =============================================================================

    def test_expectation(self):
        hmsm = self.hmsm_lag10
        e = hmsm.expectation(np.arange(hmsm.n_states_obs))
        # approximately equal for both
        assert np.abs(e - 31.73) < 0.01

        # test error case of incompatible vector size
        self.assertRaises(ValueError, hmsm.expectation, np.arange(hmsm.n_states_obs - 1))

    def test_correlation(self):
        hmsm = self.hmsm_lag10
        maxtime = 1000
        a = [1, 2, 3]
        with self.assertRaises(ValueError):
            hmsm.correlation(a, 1) # raise assertion error because size is wrong:

        # should decrease
        a = np.arange(hmsm.n_states_obs)
        times, corr1 = hmsm.correlation(a, maxtime=maxtime)
        assert len(corr1) == maxtime / hmsm.lagtime
        assert len(times) == maxtime / hmsm.lagtime
        assert corr1[0] > corr1[-1]
        a = np.arange(hmsm.n_states_obs)
        times, corr2 = hmsm.correlation(a, a, maxtime=maxtime)
        # should be identical to autocorr
        np.testing.assert_allclose(corr1, corr2)
        # Test: should be increasing in time
        b = np.arange(hmsm.n_states_obs)[::-1]
        times, corr3 = hmsm.correlation(a, b, maxtime=maxtime)
        assert len(times) == maxtime / hmsm.lagtime
        assert len(corr3) == maxtime / hmsm.lagtime
        assert corr3[0] < corr3[-1]

        # test error case of incompatible vector size
        self.assertRaises(ValueError, hmsm.correlation, np.arange(hmsm.n_states + hmsm.n_states_obs)), None

    def test_relaxation(self):
        hmsm = self.hmsm_lag10
        a = np.arange(hmsm.n_states)
        maxtime = 1000
        times, rel1 = hmsm.relaxation(hmsm.stationary_distribution, a, maxtime=maxtime)
        # should be constant because we are in equilibrium
        assert np.allclose(rel1 - rel1[0], np.zeros((np.shape(rel1)[0])))
        pi_perturbed = [1, 0]
        times, rel2 = hmsm.relaxation(pi_perturbed, a, maxtime=maxtime)
        # should relax
        assert len(times) == maxtime / hmsm.lagtime
        assert len(rel2) == maxtime / hmsm.lagtime
        assert rel2[0] < rel2[-1]

        # test error case of incompatible vector size
        with self.assertRaises(ValueError):
            hmsm.relaxation(np.arange(hmsm.n_states + 1), np.arange(hmsm.n_states + 1))

    def test_fingerprint_correlation(self):
        hmsm = self.hmsm_lag10
        # raise assertion error because size is wrong:
        a = [1, 2, 3]
        with self.assertRaises(ValueError):
            hmsm.fingerprint_correlation(a, 1)
        # should decrease
        a = np.arange(hmsm.n_states_obs)
        fp1 = hmsm.fingerprint_correlation(a)
        # first timescale is infinite
        assert fp1[0][0] == np.inf
        # next timescales are identical to timescales:
        assert np.allclose(fp1[0][1:], hmsm.timescales().magnitude)
        # all amplitudes nonnegative (for autocorrelation)
        assert np.all(fp1[1][:] >= 0)
        # identical call
        b = np.arange(hmsm.n_states_obs)
        fp2 = hmsm.fingerprint_correlation(a, b)
        assert np.allclose(fp1[0], fp2[0])
        assert np.allclose(fp1[1], fp2[1])
        # should be - of the above, apart from the first
        b = np.arange(hmsm.n_states_obs)[::-1]
        fp3 = hmsm.fingerprint_correlation(a, b)
        assert np.allclose(fp1[0], fp3[0])
        assert np.allclose(fp1[1][1:], -fp3[1][1:])

        # test error case of incompatible vector size
        self.assertRaises(ValueError, hmsm.fingerprint_correlation, np.arange(hmsm.n_states + hmsm.n_states_obs)), None

    def test_fingerprint_relaxation(self):
        hmsm = self.hmsm_lag10
        # raise assertion error because size is wrong:
        a = [1, 2, 3]
        with self.assertRaises(ValueError):
            hmsm.fingerprint_relaxation(hmsm.stationary_distribution, a)
        # equilibrium relaxation should be constant
        a = np.arange(hmsm.n_states)
        fp1 = hmsm.fingerprint_relaxation(hmsm.stationary_distribution, a)
        # first timescale is infinite
        assert fp1[0][0] == np.inf
        # next timescales are identical to timescales:
        assert np.allclose(fp1[0][1:], hmsm.timescales().magnitude)
        # dynamical amplitudes should be near 0 because we are in equilibrium
        assert np.max(np.abs(fp1[1][1:])) < 1e-10
        # off-equilibrium relaxation
        pi_perturbed = [0, 1]
        fp2 = hmsm.fingerprint_relaxation(pi_perturbed, a)
        # first timescale is infinite
        assert fp2[0][0] == np.inf
        # next timescales are identical to timescales:
        assert np.allclose(fp2[0][1:], hmsm.timescales().magnitude)
        # dynamical amplitudes should be significant because we are not in equilibrium
        assert np.max(np.abs(fp2[1][1:])) > 0.1

        # test error case of incompatible vector size
        with self.assertRaises(ValueError):
            hmsm.fingerprint_relaxation(np.arange(hmsm.n_states + 1), np.arange(hmsm.n_states + 1))

    # ================================================================================================================
    # Metastable state stuff
    # ================================================================================================================

    def test_metastable_memberships(self):
        hmsm = self.hmsm_lag10
        M = hmsm.metastable_memberships
        # should be right size
        assert np.all(M.shape == (hmsm.n_states_obs, 2))
        # should be nonnegative
        assert np.all(M >= 0)
        # should add up to one:
        assert np.allclose(np.sum(M, axis=1), np.ones(hmsm.n_states_obs))

    def test_metastable_distributions(self):
        hmsm = self.hmsm_lag10
        pccadist = hmsm.metastable_distributions
        # should be right size
        assert np.all(pccadist.shape == (2, hmsm.n_states_obs))
        # should be nonnegative
        assert np.all(pccadist >= 0)
        # should roughly add up to stationary:
        ds = pccadist[0] + pccadist[1]
        ds /= ds.sum()
        assert np.max(np.abs(ds - hmsm.stationary_distribution_obs)) < 0.001

    def test_metastable_sets(self):
        hmsm = self.hmsm_lag10
        S = hmsm.metastable_sets
        assignment = hmsm.metastable_assignments
        # should coincide with assignment
        for i, s in enumerate(S):
            for j in range(len(s)):
                assert assignment[s[j]] == i

    def test_metastable_assignment(self):
        hmsm = self.hmsm_lag10
        ass = hmsm.metastable_assignments
        # test: number of states
        assert len(ass) == hmsm.n_states_obs
        # test: should be 0 or 1
        assert np.all(ass >= 0)
        assert np.all(ass <= 1)
        # should be equal (zero variance) within metastable sets
        assert np.std(ass[:30]) == 0
        assert np.std(ass[40:]) == 0

    # ---------------------------------
    # STATISTICS, SAMPLING
    # ---------------------------------
    def test_observable_state_indexes(self):
        from sktime.markovprocess.sample import compute_index_states

        hmsm = self.hmsm_lag10
        I = compute_index_states(self.obs, subset=hmsm.count_model.observable_set)
        # I = hmsm.observable_state_indexes
        assert len(I) == hmsm.n_states_obs
        # compare to histogram
        hist = count_states(self.obs)
        # number of frames should match on active subset
        A = hmsm.count_model.observable_set
        for i in range(A.shape[0]):
            assert I[i].shape[0] == hist[A[i]]
            assert I[i].shape[1] == 2

    @unittest.skip('not yet impled, we do not store dtrajs anymore.')
    def test_sample_by_observation_probabilities(self):
        hmsm = self.hmsm_lag10
        nsample = 100
        ss = hmsm.sample_by_observation_probabilities(nsample)
        # must have the right size
        assert len(ss) == hmsm.n_states
        # must be correctly assigned
        for i, samples in enumerate(ss):
            # right shape
            assert np.all(samples.shape == (nsample, 2))
            for row in samples:
                assert row[0] == 0  # right trajectory

    def test_simulate_HMSM(self):
        hmsm = self.hmsm_lag10
        N = 400
        start = 1
        traj, obs = hmsm.simulate(N=N, start=start)
        assert len(traj) <= N
        assert len(np.unique(traj)) <= len(hmsm.transition_matrix)

    def test_dt_model(self):
        self.assertEqual(self.hmsm_lag10.dt_model.magnitude, 10)
        self.assertEqual(self.hmsm_lag10.dt_model.units, '1 step')

    # ----------------------------------
    # MORE COMPLEX TESTS / SANITY CHECKS
    # ----------------------------------

    def test_two_state_kinetics(self):
        # sanity check: k_forward + k_backward = 1.0/t2 for the two-state process
        hmsm = self.hmsm_lag10
        # transition time from left to right and vice versa
        t12 = hmsm.mfpt(0, 1)
        t21 = hmsm.mfpt(1, 0)
        # relaxation time
        t2 = hmsm.timescales()[0]
        # the following should hold: k12 + k21 = k2.
        # sum of forward/backward rates can be a bit smaller because we are using small cores and
        # therefore underestimate rates
        ksum = 1.0 / t12 + 1.0 / t21
        k2 = 1.0 / t2
        assert np.abs(k2 - ksum) < 1e-4

    def test_cktest_simple(self):
        dtraj = np.random.randint(0, 10, 100)
        oom = estimate_markov_model(dtraj, 1)
        nsets = 2
        est, hmm = oom.hmm(dtraj, nsets, return_estimator=True)
        cktest(test_estimator=est, test_model=hmm, dtrajs=dtraj, nsets=nsets)

    def test_submodel_simple(self):
        # sanity check for submodel;
        dtrj = [np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0,
                0, 2, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0,
                1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 2, 0, 0, 1, 1, 2, 0, 1, 1, 1,
                0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0])]

        h = estimate_hidden_markov_model(dtrj, 3, 2)
        hs = h.submodel_largest(mincount_connectivity=5, dtrajs=dtrj)

        self.assertEqual(hs.timescales().shape[0], 1)
        self.assertEqual(hs.stationary_distribution.shape[0], 2)
        self.assertEqual(hs.transition_matrix.shape, (2, 2))


def estimate_hidden_markov_model(dtrajs, n_states, lag, return_estimator=False, **kwargs):
    est = MaximumLikelihoodHMSM(n_states=n_states, lagtime=lag, **kwargs)
    est.fit(dtrajs, )

    model = est.fetch_model().submodel_largest(dtrajs=dtrajs)
    if not return_estimator:
        return model

    return est, model


class TestHMMSpecialCases(unittest.TestCase):

    def test_separate_states(self):
        dtrajs = [np.array([0, 1, 1, 1, 1, 1, 0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1]),
                  np.array([2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2]), ]
        hmm = estimate_hidden_markov_model(dtrajs, 3, lag=1, separate=[0])
        # we expect zeros in all samples at the following indexes:
        pobs_zeros = ((0, 1, 2, 2, 2), (0, 0, 1, 2, 3))
        assert np.allclose(hmm.observation_probabilities[pobs_zeros], 0)


if __name__ == "__main__":
    unittest.main()
