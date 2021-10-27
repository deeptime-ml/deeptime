import unittest

import deeptime.markov.hmm._hmm_bindings as _bindings
import numpy as np
import pytest
from numpy.testing import assert_raises, assert_equal, assert_array_almost_equal, assert_

import deeptime.markov.tools
from deeptime.data import DoubleWellDiscrete
from deeptime.markov import count_states
from deeptime.data import prinz_potential
from deeptime.markov.hmm import DiscreteOutputModel, MaximumLikelihoodHMM, init, BayesianHMM, viterbi, HiddenMarkovModel
from deeptime.markov.msm import MarkovStateModel, MaximumLikelihoodMSM
from tests.markov.msm.test_mlmsm import estimate_markov_model
from tests.testing_utilities import assert_array_not_equal


@pytest.mark.parametrize('mode', ['maximum-likelihood', 'bayesian'], ids=lambda mode: f"mode={mode}")
@pytest.mark.parametrize('reversible', [True, False], ids=lambda rev: f"reversible={rev}")
def test_disconnected_dtraj_sanity(mode, reversible):
    msm1 = MarkovStateModel([[.8, .2], [.3, .7]])
    msm2 = MarkovStateModel([[.9, .05, .05], [.3, .6, .1], [.1, .1, .8]])
    dtrajs = [msm1.simulate(10000), 2 + msm2.simulate(10000), np.array([5]*100)]
    init_hmm = init.discrete.random_guess(6, 3)
    hmm = MaximumLikelihoodHMM(init_hmm, lagtime=1, reversible=reversible) \
        .fit(dtrajs).fetch_model()
    if mode == 'bayesian':
        BayesianHMM(hmm.submodel_largest(dtrajs=dtrajs), reversible=reversible).fit(dtrajs)


class TestAlgorithmsAgainstReference(unittest.TestCase):
    """ Tests against example from Wikipedia: http://en.wikipedia.org/wiki/Forward-backward_algorithm#Example """

    def setUp(self) -> None:
        # weather transition probabilities: 1=rain and 2=no rain
        self.transition_probabilities = np.array([
            [0.7, 0.3],
            [0.3, 0.7]
        ])
        # discrete traj: 1 = umbrella, 2 = no umbrella
        self.dtraj = np.array([0, 0, 1, 0, 0])
        # conditional probabilities
        self.conditional_probabilities = np.array([
            [0.9, 0.1], [0.2, 0.8]
        ])
        self.state_probabilities = np.array([
            [0.9, 0.2],
            [0.9, 0.2],
            [0.1, 0.8],
            [0.9, 0.2],
            [0.9, 0.2]
        ])

    def test_model_likelihood(self):
        hmm = HiddenMarkovModel(self.transition_probabilities, self.conditional_probabilities)
        loglik = hmm.compute_observation_likelihood(self.dtraj)
        ref_logprob = -3.3725
        np.testing.assert_array_almost_equal(loglik, ref_logprob, decimal=4)

    def test_forward(self):
        alpha_out = np.zeros_like(self.state_probabilities)
        logprob = _bindings.util.forward(self.transition_probabilities, self.state_probabilities, np.array([0.5, 0.5]),
                                         alpha_out=alpha_out)
        ref_logprob = -3.3725
        ref_alpha = np.array([
            [0.8182, 0.1818],
            [0.8834, 0.1166],
            [0.1907, 0.8093],
            [0.7308, 0.2692],
            [0.8673, 0.1327]
        ])
        np.testing.assert_array_almost_equal(logprob, ref_logprob, decimal=4)
        np.testing.assert_array_almost_equal(alpha_out, ref_alpha, decimal=4)

    def test_backward(self):
        beta_out = np.zeros_like(self.state_probabilities)
        _bindings.util.backward(self.transition_probabilities, self.state_probabilities, beta_out=beta_out)

        ref_beta = np.array([
            [0.5923, 0.4077],
            [0.3763, 0.6237],
            [0.6533, 0.3467],
            [0.6273, 0.3727],
            [.5, .5]
        ])
        np.testing.assert_array_almost_equal(beta_out, ref_beta, decimal=4)

    def test_state_probabilities(self):
        ref_alpha = np.array([
            [0.8182, 0.1818],
            [0.8834, 0.1166],
            [0.1907, 0.8093],
            [0.7308, 0.2692],
            [0.8673, 0.1327]
        ])
        ref_beta = np.array([
            [0.5923, 0.4077],
            [0.3763, 0.6237],
            [0.6533, 0.3467],
            [0.6273, 0.3727],
            [.5, .5]
        ])
        gamma = np.zeros((len(self.dtraj), self.transition_probabilities.shape[0]))
        _bindings.util.state_probabilities(ref_alpha, ref_beta, gamma_out=gamma)

        gamma_ref = np.array([
            [0.8673, 0.1327],
            [0.8204, 0.1796],
            [0.3075, 0.6925],
            [0.8204, 0.1796],
            [0.8673, 0.1327]
        ])
        np.testing.assert_array_almost_equal(gamma, gamma_ref, decimal=4)

    def test_viterbi(self):
        path = viterbi(self.transition_probabilities, self.state_probabilities, np.array([0.5, 0.5]))
        np.testing.assert_array_equal(path, self.dtraj)


class TestMLHMM(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        dtraj = DoubleWellDiscrete().dtraj
        initial_hmm_10 = init.discrete.metastable_from_data(dtraj, n_hidden_states=2, lagtime=10)
        cls.hmm_lag10 = MaximumLikelihoodHMM(initial_hmm_10, lagtime=10).fit(dtraj).fetch_model()
        cls.hmm_lag10_largest = cls.hmm_lag10.submodel_largest(dtrajs=dtraj)
        cls.msm_lag10 = estimate_markov_model(dtraj, 10, reversible=True)
        initial_hmm_1 = init.discrete.metastable_from_data(dtraj, n_hidden_states=2, lagtime=1)
        cls.hmm_lag1 = MaximumLikelihoodHMM(initial_hmm_1, lagtime=1).fit(dtraj).fetch_model()
        cls.hmm_lag1_largest = cls.hmm_lag1.submodel_largest(dtrajs=dtraj)
        cls.msm_lag1 = estimate_markov_model(dtraj, 1, reversible=True)
        cls.dtrajs = dtraj

    # =============================================================================
    # Test basic HMM properties
    # =============================================================================

    def test_likelihood(self):
        ref_loglik = self.hmm_lag10.likelihood
        new_loglik = self.hmm_lag10.compute_observation_likelihood(self.dtrajs)
        np.testing.assert_allclose(new_loglik, ref_loglik, rtol=1e-1)

    def test_collect_observations_in_state_sanity(self):
        self.hmm_lag1.collect_observations_in_state(self.dtrajs, 1)

    def test_output_model(self):
        assert isinstance(self.hmm_lag1.output_model, DiscreteOutputModel)
        assert isinstance(self.hmm_lag10.output_model, DiscreteOutputModel)

    def test_reversible(self):
        np.testing.assert_(self.hmm_lag1.transition_model.reversible)
        np.testing.assert_(self.hmm_lag10.transition_model.reversible)

    def test_stationary(self):
        np.testing.assert_(self.hmm_lag1.transition_model.stationary)
        np.testing.assert_(self.hmm_lag10.transition_model.stationary)

    def test_lag(self):
        assert self.hmm_lag1.transition_model.lagtime == 1
        assert self.hmm_lag10.transition_model.lagtime == 10

    def test_n_states(self):
        np.testing.assert_equal(self.hmm_lag1.n_hidden_states, 2)
        np.testing.assert_equal(self.hmm_lag1.transition_model.n_states, 2)
        np.testing.assert_equal(self.hmm_lag10.n_hidden_states, 2)
        np.testing.assert_equal(self.hmm_lag10.transition_model.n_states, 2)

    def test_transition_matrix(self):
        import deeptime.markov.tools.analysis as msmana
        for P in [self.hmm_lag1.transition_model.transition_matrix, self.hmm_lag10.transition_model.transition_matrix]:
            np.testing.assert_(msmana.is_transition_matrix(P))
            np.testing.assert_(msmana.is_reversible(P))

    def test_eigenvalues(self):
        for ev in [self.hmm_lag1.transition_model.eigenvalues(2), self.hmm_lag10.transition_model.eigenvalues(2)]:
            np.testing.assert_equal(len(ev), 2)
            np.testing.assert_allclose(ev[0], 1)
            np.testing.assert_(ev[1] < 1.)

    def test_eigenvectors_left(self):
        for evec in [self.hmm_lag1.transition_model.eigenvectors_left(2),
                     self.hmm_lag10.transition_model.eigenvectors_left(2)]:
            np.testing.assert_equal(evec.shape, (2, 2))
            np.testing.assert_equal(np.sign(evec[0, 0]), np.sign(evec[0, 1]))
            assert_array_not_equal(np.sign(evec[1, 0]), np.sign(evec[1, 1]))

    def test_eigenvectors_right(self):
        for evec in [self.hmm_lag1.transition_model.eigenvectors_right(),
                     self.hmm_lag10.transition_model.eigenvectors_right()]:
            np.testing.assert_equal(evec.shape, (2, 2))
            np.testing.assert_allclose(evec[0, 0], evec[1, 0])
            assert_array_not_equal(np.sign(evec[0, 1]), np.sign(evec[1, 1]))

    def test_initial_distribution(self):
        for mu in [self.hmm_lag1.initial_distribution, self.hmm_lag10.initial_distribution]:
            # normalization
            assert np.isclose(mu.sum(), 1.0)
            # should be on one side
            assert np.isclose(mu[0], 1.0) or np.isclose(mu[0], 0.0)

    def test_stationary_distribution(self):
        for mu in [self.hmm_lag1.transition_model.stationary_distribution,
                   self.hmm_lag10.transition_model.stationary_distribution]:
            # normalization
            assert np.isclose(mu.sum(), 1.0)
            # positivity
            assert np.all(mu > 0.0)
            # this data: approximately equal probability
            assert np.max(np.abs(mu[0] - mu[1])) < 0.05

    def test_lifetimes(self):
        for l in [self.hmm_lag1.lifetimes, self.hmm_lag10.lifetimes]:
            assert len(l) == 2
            assert np.all(l > 0.0)
        # this data: lifetimes about 680
        np.testing.assert_(np.max(np.abs(self.hmm_lag10.lifetimes - 680)) < 20.0)

    def test_timescales(self):
        for l in [self.hmm_lag1.transition_model.timescales(2), self.hmm_lag10.transition_model.timescales(2)]:
            np.testing.assert_equal(len(l), 1)
            np.testing.assert_(l > 0.)
        # this data: lifetimes about 680
        np.testing.assert_(np.abs(self.hmm_lag10.transition_model.timescales(2)[0] - 340) < 20.0)

    # =============================================================================
    # Hidden transition matrix first passage problems
    # =============================================================================

    def test_committor(self):
        hmsm = self.hmm_lag10
        a = 0
        b = 1
        q_forward = hmsm.transition_model.committor_forward(a, b)
        np.testing.assert_equal(q_forward[a], 0)
        np.testing.assert_equal(q_forward[b], 1)
        q_backward = hmsm.transition_model.committor_backward(a, b)
        np.testing.assert_equal(q_backward[a], 1)
        np.testing.assert_equal(q_backward[b], 0)
        # REVERSIBLE:
        np.testing.assert_allclose(q_forward + q_backward, np.ones(hmsm.n_hidden_states))

    def test_mfpt(self):
        hmsm = self.hmm_lag10
        a = 0
        b = 1
        tab = hmsm.transition_model.mfpt(a, b)
        tba = hmsm.transition_model.mfpt(b, a)
        np.testing.assert_(tab > 0)
        np.testing.assert_(tba > 0)
        # HERE:
        err = np.minimum(np.abs(tab - 680.708752214), np.abs(tba - 699.560589099))
        np.testing.assert_(err < 1e-3, msg="err was {}".format(err))

    # =============================================================================
    # Test HMSM observable spectral properties
    # =============================================================================

    def test_n_states_obs(self):
        np.testing.assert_equal(self.hmm_lag1_largest.n_observation_states, self.msm_lag1.n_states)
        np.testing.assert_equal(self.hmm_lag10_largest.n_observation_states, self.msm_lag10.n_states)

    def test_observation_probabilities(self):
        np.testing.assert_array_equal(self.hmm_lag1.output_probabilities.shape, (2, self.hmm_lag1.n_observation_states))
        np.testing.assert_allclose(self.hmm_lag1.output_probabilities.sum(axis=1), np.ones(2))
        np.testing.assert_array_equal(self.hmm_lag10.output_probabilities.shape,
                                      (2, self.hmm_lag10.n_observation_states))
        np.testing.assert_allclose(self.hmm_lag10.output_probabilities.sum(axis=1), np.ones(2))

    def test_transition_matrix_obs(self):
        assert np.array_equal(self.hmm_lag1_largest.transition_matrix_obs().shape,
                              (self.hmm_lag1_largest.n_observation_states, self.hmm_lag1_largest.n_observation_states))
        assert np.array_equal(self.hmm_lag10_largest.transition_matrix_obs().shape,
                              (
                              self.hmm_lag10_largest.n_observation_states, self.hmm_lag10_largest.n_observation_states))
        for T in [self.hmm_lag1_largest.transition_matrix_obs(),
                  self.hmm_lag1_largest.transition_matrix_obs(k=2),
                  self.hmm_lag10_largest.transition_matrix_obs(),
                  self.hmm_lag10_largest.transition_matrix_obs(k=4)]:
            np.testing.assert_(deeptime.markov.tools.analysis.is_transition_matrix(T))
            np.testing.assert_(deeptime.markov.tools.analysis.is_reversible(T))

    def test_stationary_distribution_obs(self):
        for hmsm in [self.hmm_lag1_largest, self.hmm_lag10_largest]:
            sd = hmsm.stationary_distribution_obs
            np.testing.assert_equal(len(sd), hmsm.n_observation_states)
            np.testing.assert_allclose(sd.sum(), 1.0)
            np.testing.assert_allclose(sd, np.dot(sd, hmsm.transition_matrix_obs()))

    def test_eigenvectors_left_obs(self):
        for hmsm in [self.hmm_lag1_largest, self.hmm_lag10_largest]:
            L = hmsm.eigenvectors_left_obs
            # shape should be right
            np.testing.assert_array_equal(L.shape, (hmsm.n_hidden_states, hmsm.n_observation_states))
            # first one should be identical to stat.dist
            l1 = L[0, :]
            err = hmsm.stationary_distribution_obs - l1
            np.testing.assert_almost_equal(np.max(np.abs(err)), 0, decimal=9)
            # sums should be 1, 0, 0, ...
            np.testing.assert_array_almost_equal(L.sum(axis=1), np.array([1., 0.]))
            # REVERSIBLE:
            if hmsm.transition_model.reversible:
                np.testing.assert_(np.all(np.isreal(L)))

    def test_eigenvectors_right_obs(self):
        for hmsm in [self.hmm_lag1_largest, self.hmm_lag10_largest]:
            R = hmsm.eigenvectors_right_obs
            # shape should be right
            np.testing.assert_array_equal(R.shape, (hmsm.n_observation_states, hmsm.n_hidden_states))
            # should be all ones
            r1 = R[:, 0]
            np.testing.assert_allclose(r1, np.ones(hmsm.n_observation_states))
            # REVERSIBLE:
            if hmsm.transition_model.reversible:
                np.testing.assert_(np.all(np.isreal(R)))

    # =============================================================================
    # Test HMSM kinetic observables
    # =============================================================================

    def test_expectation(self):
        hmsm = self.hmm_lag10_largest
        e = hmsm.expectation_obs(np.arange(hmsm.n_observation_states))
        # approximately equal for both
        np.testing.assert_almost_equal(e, 31.73, decimal=2)
        # test error case of incompatible vector size
        np.testing.assert_raises(ValueError, hmsm.expectation_obs, np.arange(hmsm.n_observation_states - 1))

    def test_correlation(self):
        hmsm = self.hmm_lag10_largest
        maxtime = 1000
        a = [1, 2, 3]

        # raise assertion error because size is wrong
        np.testing.assert_raises(ValueError, hmsm.correlation_obs, a, 1)

        # should decrease
        a = np.arange(hmsm.n_observation_states)
        times, corr1 = hmsm.correlation_obs(a, maxtime=maxtime)
        np.testing.assert_equal(len(corr1), maxtime / hmsm.transition_model.lagtime)
        np.testing.assert_equal(len(times), maxtime / hmsm.transition_model.lagtime)
        np.testing.assert_(corr1[0] > corr1[-1])
        a = np.arange(hmsm.n_observation_states)
        times, corr2 = hmsm.correlation_obs(a, a, maxtime=maxtime)
        # should be identical to autocorr
        np.testing.assert_allclose(corr1, corr2)
        # Test: should be increasing in time
        b = np.arange(hmsm.n_observation_states)[::-1]
        times, corr3 = hmsm.correlation_obs(a, b, maxtime=maxtime)
        np.testing.assert_equal(len(times), maxtime / hmsm.transition_model.lagtime)
        np.testing.assert_equal(len(corr3), maxtime / hmsm.transition_model.lagtime)
        np.testing.assert_(corr3[0] < corr3[-1])

        # test error case of incompatible vector size
        np.testing.assert_raises(ValueError, hmsm.correlation_obs,
                                 np.arange(hmsm.n_hidden_states + hmsm.n_observation_states))

    def test_relaxation(self):
        # todo this only really tests the hidden msm relaxation
        hmsm = self.hmm_lag10_largest
        a = np.arange(hmsm.n_hidden_states)
        maxtime = 1000
        times, rel1 = hmsm.transition_model.relaxation(hmsm.transition_model.stationary_distribution, a,
                                                       maxtime=maxtime)
        # should be constant because we are in equilibrium
        assert np.allclose(rel1 - rel1[0], np.zeros((np.shape(rel1)[0])))
        pi_perturbed = [1, 0]
        times, rel2 = hmsm.transition_model.relaxation(pi_perturbed, a, maxtime=maxtime)
        # should relax
        assert len(times) == maxtime / hmsm.transition_model.lagtime
        assert len(rel2) == maxtime / hmsm.transition_model.lagtime
        assert rel2[0] < rel2[-1]

        # test error case of incompatible vector size
        with self.assertRaises(ValueError):
            hmsm.relaxation_obs(np.arange(hmsm.n_hidden_states + 1), np.arange(hmsm.n_hidden_states + 1))

    def test_fingerprint_correlation(self):
        hmsm = self.hmm_lag10_largest
        # raise assertion error because size is wrong:
        a = [1, 2, 3]
        np.testing.assert_raises(ValueError, hmsm.fingerprint_correlation_obs, a, 1)
        # should decrease
        a = np.arange(hmsm.n_observation_states)
        fp1 = hmsm.fingerprint_correlation_obs(a)
        # first timescale is infinite
        np.testing.assert_equal(fp1[0][0], np.inf)
        # next timescales are identical to timescales:
        np.testing.assert_allclose(fp1[0][1:], hmsm.transition_model.timescales())
        # all amplitudes nonnegative (for autocorrelation)
        np.testing.assert_(np.all(fp1[1][:] >= 0))
        # identical call
        b = np.arange(hmsm.n_observation_states)
        fp2 = hmsm.fingerprint_correlation_obs(a, b)
        np.testing.assert_allclose(fp1[0], fp2[0])
        np.testing.assert_allclose(fp1[1], fp2[1])
        # should be - of the above, apart from the first
        b = np.arange(hmsm.n_observation_states)[::-1]
        fp3 = hmsm.fingerprint_correlation_obs(a, b)
        np.testing.assert_allclose(fp1[0], fp3[0])
        np.testing.assert_allclose(fp1[1][1:], -fp3[1][1:])

        # test error case of incompatible vector size
        self.assertRaises(ValueError, hmsm.fingerprint_correlation_obs,
                          np.arange(hmsm.n_hidden_states + hmsm.n_observation_states))

    def test_fingerprint_relaxation(self):
        hmsm = self.hmm_lag10_largest
        # raise assertion error because size is wrong:
        a = [1, 2, 3]
        np.testing.assert_raises(ValueError, hmsm.fingerprint_relaxation_obs, hmsm.stationary_distribution_obs, a)
        # equilibrium relaxation should be constant
        a = np.arange(hmsm.n_hidden_states)
        fp1 = hmsm.transition_model.fingerprint_relaxation(hmsm.transition_model.stationary_distribution, a)
        # first timescale is infinite
        np.testing.assert_equal(fp1[0][0], np.inf)
        # next timescales are identical to timescales:
        np.testing.assert_allclose(fp1[0][1:], hmsm.transition_model.timescales())
        # dynamical amplitudes should be near 0 because we are in equilibrium
        np.testing.assert_almost_equal(np.max(np.abs(fp1[1][1:])), 0, decimal=9)
        # off-equilibrium relaxation
        pi_perturbed = [0, 1]
        fp2 = hmsm.transition_model.fingerprint_relaxation(pi_perturbed, a)
        # first timescale is infinite
        np.testing.assert_equal(fp2[0][0], np.inf)
        # next timescales are identical to timescales:
        np.testing.assert_allclose(fp2[0][1:], hmsm.transition_model.timescales())
        # dynamical amplitudes should be significant because we are not in equilibrium
        np.testing.assert_(np.max(np.abs(fp2[1][1:])) > 0.1)
        # test error case of incompatible vector size
        np.testing.assert_raises(ValueError, hmsm.fingerprint_relaxation_obs, np.arange(hmsm.n_hidden_states + 1),
                                 np.arange(hmsm.n_hidden_states + 1))

    # ================================================================================================================
    # Metastable state stuff
    # ================================================================================================================

    def test_metastable_memberships(self):
        hmsm = self.hmm_lag10_largest
        M = hmsm.metastable_memberships
        # should be right size
        np.testing.assert_equal(M.shape, (hmsm.n_observation_states, hmsm.n_hidden_states))
        # should be nonnegative
        np.testing.assert_(np.all(M >= 0))
        # should add up to one:
        np.testing.assert_allclose(np.sum(M, axis=1), np.ones(hmsm.n_observation_states))

    def test_metastable_distributions(self):
        hmsm = self.hmm_lag10_largest
        pccadist = hmsm.metastable_distributions
        # should be right size
        np.testing.assert_equal(pccadist.shape, (hmsm.n_hidden_states, hmsm.n_observation_states))
        # should be nonnegative
        np.testing.assert_(np.all(pccadist >= 0))
        # should roughly add up to stationary:
        ds = pccadist[0] + pccadist[1]
        ds /= ds.sum()
        np.testing.assert_array_almost_equal(ds, hmsm.stationary_distribution_obs, decimal=3)

    def test_metastable_sets(self):
        hmsm = self.hmm_lag10_largest
        S = hmsm.metastable_sets
        assignment = hmsm.metastable_assignments
        # should coincide with assignment
        for i, s in enumerate(S):
            for j in range(len(s)):
                assert assignment[s[j]] == i

    def test_metastable_assignment(self):
        hmsm = self.hmm_lag10_largest
        ass = hmsm.metastable_assignments
        # test: number of states
        np.testing.assert_equal(len(ass), hmsm.n_observation_states)
        # test: should be in [0, 1]
        np.testing.assert_(np.all(ass >= 0))
        np.testing.assert_(np.all(ass <= 1))
        # should be equal (zero variance) within metastable sets
        np.testing.assert_equal(np.std(ass[:30]), 0)
        np.testing.assert_equal(np.std(ass[40:]), 0)

    # ---------------------------------
    # STATISTICS, SAMPLING
    # ---------------------------------
    def test_observable_state_indices(self):
        from deeptime.markov import sample

        hmsm = self.hmm_lag10_largest
        I = sample.compute_index_states(self.dtrajs, subset=hmsm.observation_symbols)
        # I = hmsm.observable_state_indexes
        np.testing.assert_equal(len(I), hmsm.n_observation_states)
        # compare to histogram
        hist = count_states(self.dtrajs)
        # number of frames should match on active subset
        A = hmsm.observation_symbols
        for i in range(A.shape[0]):
            np.testing.assert_equal(I[i].shape[0], hist[A[i]])
            np.testing.assert_equal(I[i].shape[1], 2)

    def test_transform_to_observed_symbols(self):
        hmsm = self.hmm_lag10_largest
        dtraj = np.concatenate((hmsm.observation_symbols_full, [500000]))
        mapped = hmsm.transform_discrete_trajectories_to_observed_symbols(dtraj)[0]
        for i in range(len(dtraj)):
            state = dtraj[i]
            if state in hmsm.observation_symbols:
                assert_equal(mapped[i], state)
            else:
                assert_equal(mapped[i], -1)

    def test_sample_by_observation_probabilities_out_of_sample(self):
        hmsm = self.hmm_lag10_largest
        nsample = 50
        with assert_raises(ValueError):  # symbols not in obs set
            hmsm.sample_by_observation_probabilities([0, 1, 2], nsample)
        # sanity check subset of observation symbols
        hmsm.sample_by_observation_probabilities(np.arange(66), nsample)
        dtraj2 = np.concatenate((hmsm.observation_symbols, [10000]))
        # sanity check too large state in there
        hmsm.sample_by_observation_probabilities(dtraj2, 50)

    def test_sample_by_observation_probabilities(self):
        hmsm = self.hmm_lag10_largest
        nsample = 100
        ss = hmsm.sample_by_observation_probabilities(self.dtrajs, nsample)
        # must have the right size
        np.testing.assert_equal(len(ss), hmsm.n_hidden_states)
        # must be correctly assigned
        for i, samples in enumerate(ss):
            # right shape
            np.testing.assert_equal(samples.shape, (nsample, 2))
            for row in samples:
                np.testing.assert_equal(row[0], 0)  # right trajectory

    def test_sample_by_observation_probabilities_mapping(self):
        tmat = np.array([[0.9, .1], [.1, .9]])
        # hidden states correspond to observable states
        obs = np.eye(2)
        hmm = HiddenMarkovModel(tmat, obs)
        # dtraj halfway-split between states 0 and 1
        dtrajs = np.repeat([0, 1], 10)
        samples = hmm.sample_by_observation_probabilities(dtrajs, 10)
        # test that all trajectory indices are 0 (only 1 traj)
        np.testing.assert_array_equal(np.unique(np.concatenate(samples)[:, 0]), [0])
        # test that both hidden states map to correct parts of dtraj
        np.testing.assert_(np.all(samples[0][:, 1] < 10))
        np.testing.assert_(np.all(samples[1][:, 1] >= 10))

    def test_sample_by_noncrisp_observation_probabilities_mapping(self):
        tmat = np.array([[0.9, .1], [.1, .9]])
        # hidden states correspond to observable states
        obs = np.array([[.9, .1], [.4, .6]])
        hmm = HiddenMarkovModel(tmat, obs)
        # dtraj halfway-split between states 0 and 1
        dtrajs = np.repeat([0, 1], 10)
        n_samples = 500000
        samples = hmm.sample_by_observation_probabilities(dtrajs, n_samples)
        # test that both hidden states map to correct distributions
        probs_hidden1 = np.histogram(dtrajs[samples[0][:, 1]], bins=2)[0] / n_samples
        probs_hidden2 = np.histogram(dtrajs[samples[1][:, 1]], bins=2)[0] / n_samples
        assert_array_almost_equal(probs_hidden1, [.9, .1], decimal=2)
        assert_array_almost_equal(probs_hidden2, [.4, .6], decimal=2)

    def test_simulate_HMSM(self):
        hmsm = self.hmm_lag10_largest
        N = 400
        start = 1
        traj, obs = hmsm.simulate(n_steps=N, start=start)
        assert len(traj) <= N
        assert len(np.unique(traj)) <= len(hmsm.transition_model.transition_matrix)

    # ----------------------------------
    # MORE COMPLEX TESTS / SANITY CHECKS
    # ----------------------------------

    def test_two_state_kinetics(self):
        # sanity check: k_forward + k_backward = 1.0/t2 for the two-state process
        hmsm = self.hmm_lag10_largest
        # transition time from left to right and vice versa
        t12 = hmsm.transition_model.mfpt(0, 1)
        t21 = hmsm.transition_model.mfpt(1, 0)
        # relaxation time
        t2 = hmsm.transition_model.timescales()[0]
        # the following should hold: k12 + k21 = k2.
        # sum of forward/backward rates can be a bit smaller because we are using small cores and
        # therefore underestimate rates
        ksum = 1.0 / t12 + 1.0 / t21
        k2 = 1.0 / t2
        np.testing.assert_almost_equal(k2, ksum, decimal=4)

    def test_submodel_simple(self):
        # sanity check for submodel;
        dtraj = [np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0,
                           0, 2, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0,
                           1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 2, 0, 0, 1, 1, 2, 0, 1, 1, 1,
                           0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0])]
        init_hmm = init.discrete.metastable_from_data(dtraj, n_hidden_states=3, lagtime=2)
        hmm = MaximumLikelihoodHMM(init_hmm, lagtime=2).fit(dtraj).fetch_model()
        hmm_sub = hmm.submodel_largest(connectivity_threshold=5, dtrajs=dtraj)

        self.assertEqual(hmm_sub.transition_model.timescales().shape[0], 1)
        self.assertEqual(hmm_sub.transition_model.stationary_distribution.shape[0], 2)
        self.assertEqual(hmm_sub.transition_model.transition_matrix.shape, (2, 2))

    def test_separate_states(self):
        dtrajs = [np.array([0, 1, 1, 1, 1, 1, 0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1]),
                  np.array([2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2]), ]
        init_hmm = init.discrete.metastable_from_data(dtrajs, n_hidden_states=3, lagtime=1, separate_symbols=[0])
        hmm = MaximumLikelihoodHMM(init_hmm, lagtime=1).fit(dtrajs).fetch_model().submodel_largest(dtrajs=dtrajs)
        # we expect zeros in all samples at the following indices:
        pobs_zeros = ((0, 1, 2, 2, 2), (0, 0, 1, 2, 3))
        assert np.allclose(hmm.output_probabilities[pobs_zeros], 0)


class TestMLHMMPathologicalCases(unittest.TestCase):

    def test_1state(self):
        obs = np.array([0, 0, 0, 0, 0], dtype=int)
        init_hmm = init.discrete.metastable_from_data(obs, n_hidden_states=1, lagtime=1)
        hmm = MaximumLikelihoodHMM(init_hmm, lagtime=1).fit(obs).fetch_model()
        # hmm = bhmm.estimate_hmm([obs], n_states=1, lag=1, accuracy=1e-6)
        p0_ref = np.array([1.0])
        A_ref = np.array([[1.0]])
        B_ref = np.array([[1.0]])
        assert np.allclose(hmm.initial_distribution, p0_ref)
        assert np.allclose(hmm.transition_model.transition_matrix, A_ref)
        assert np.allclose(hmm.output_probabilities, B_ref)

    def test_1state_fail(self):
        obs = np.array([0, 0, 0, 0, 0], dtype=int)
        with self.assertRaises(ValueError):
            _ = init.discrete.metastable_from_data(obs, n_hidden_states=2, lagtime=1)

    def test_2state_step(self):
        obs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
        init_hmm = init.discrete.metastable_from_data(obs, n_hidden_states=2, lagtime=1)
        hmm = MaximumLikelihoodHMM(init_hmm, accuracy=1e-6, lagtime=1).fit(obs).fetch_model()
        p0_ref = np.array([1, 0])
        A_ref = np.array([[0.8, 0.2],
                          [0.0, 1.0]])
        B_ref = np.array([[1, 0],
                          [0, 1]])
        perm = [1, 0]  # permutation
        assert np.allclose(hmm.initial_distribution, p0_ref, atol=1e-5) \
               or np.allclose(hmm.initial_distribution, p0_ref[perm], atol=1e-5)
        assert np.allclose(hmm.transition_model.transition_matrix, A_ref, atol=1e-5) \
               or np.allclose(hmm.transition_model.transition_matrix, A_ref[np.ix_(perm, perm)], atol=1e-5)
        assert np.allclose(hmm.output_probabilities, B_ref, atol=1e-5) \
               or np.allclose(hmm.output_probabilities, B_ref[[perm]], atol=1e-5)

    def test_2state_2step(self):
        obs = np.array([0, 1, 0], dtype=int)
        init_hmm = init.discrete.metastable_from_data(obs, n_hidden_states=2, lagtime=1)
        hmm = MaximumLikelihoodHMM(init_hmm, lagtime=1).fit(obs).fetch_model()
        p0_ref = np.array([1, 0])
        A_ref = np.array([[0.0, 1.0],
                          [1.0, 0.0]])
        B_ref = np.array([[1, 0],
                          [0, 1]])
        perm = [1, 0]  # permutation
        assert np.allclose(hmm.initial_distribution, p0_ref, atol=1e-5) \
               or np.allclose(hmm.initial_distribution, p0_ref[perm], atol=1e-5)
        assert np.allclose(hmm.transition_model.transition_matrix, A_ref, atol=1e-5) \
               or np.allclose(hmm.transition_model.transition_matrix, A_ref[np.ix_(perm, perm)], atol=1e-5)
        assert np.allclose(hmm.output_probabilities, B_ref, atol=1e-5) \
               or np.allclose(hmm.output_probabilities, B_ref[[perm]], atol=1e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gaussian_prinz(dtype):
    system = prinz_potential()
    trajs = system.trajectory(np.zeros((5, 1)), length=5000).astype(dtype)
    init_ghmm = init.gaussian.from_data(trajs, 4, reversible=True)
    ghmm = MaximumLikelihoodHMM(init_ghmm, lagtime=1).fit_fetch(trajs)
    means = ghmm.output_model.means

    for minimum in system.minima:
        assert_(np.any(np.abs(means - minimum) < 0.1))
