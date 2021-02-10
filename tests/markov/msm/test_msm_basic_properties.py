import pytest
import numpy as np
import scipy
from numpy.testing import assert_equal, assert_, assert_array_almost_equal, assert_raises, assert_almost_equal, \
    assert_allclose
from scipy.sparse import issparse

from tests.markov.msm.util import MLMSM_PARAMS, AMM_PARAMS, MLMSM_IDS, AMM_IDS, make_double_well
from deeptime.markov.msm import MaximumLikelihoodMSM, AugmentedMSM


@pytest.mark.parametrize("setting", MLMSM_PARAMS + AMM_PARAMS, ids=MLMSM_IDS + AMM_IDS)
class TestMSMBasicProperties(object):

    def test_reversible_property(self, setting):
        scenario = make_double_well(setting)
        assert_equal(scenario.msm_estimator.reversible, scenario.msm.reversible)

    def test_sparse_property(self, setting):
        scenario = make_double_well(setting)
        assert_equal(scenario.msm_estimator.sparse, scenario.msm.sparse)

    def test_lagtime_property(self, setting):
        scenario = make_double_well(setting)
        assert_equal(scenario.msm.lagtime, scenario.lagtime)

    def test_state_symbols(self, setting):
        scenario = make_double_well(setting)
        # should always be <= full set
        assert_(len(scenario.msm.count_model.state_symbols) <= scenario.msm.count_model.n_states_full)
        # should be length of n_states
        assert_equal(len(scenario.msm.count_model.state_symbols), scenario.msm.count_model.n_states)

    def test_propagate(self, setting):
        scenario = make_double_well(setting)
        sd = scenario.msm.propagate(scenario.msm.stationary_distribution, 10)
        assert_array_almost_equal(sd, scenario.msm.stationary_distribution)

    def test_compute_state_indices(self, setting):
        scenario = make_double_well(setting)
        dtrajs = [np.array([1, 0]), np.array([18, 19, 18, 33, 1, 0]), np.array([18, 19, 18, 1, 0])]
        ix = scenario.msm.compute_state_indices(dtrajs)
        for state, indices in enumerate(ix):
            for traj, frame in indices:
                assert_equal(dtrajs[traj][frame], scenario.msm.count_model.state_symbols[state])

        with assert_raises(ValueError):
            scenario.msm.compute_state_indices(np.array([0] * 22))  # state 0 is not represented

    def test_hmm_coarse_graining_sanity(self, setting):
        scenario = make_double_well(setting)
        reversible = scenario.msm.reversible
        sparse = scenario.msm.sparse
        if isinstance(scenario.msm, AugmentedMSM):
            assert_equal(scenario.msm.hmm, None)  # not supported for AMMs
        elif reversible and not sparse:
            hmm = scenario.msm.hmm(scenario.data.dtraj, nhidden=2)
            assert_equal(hmm.n_hidden_states, 2)
            assert_equal(hmm.n_observation_states, scenario.msm.count_model.n_states_full)
        else:
            with assert_raises(ValueError):
                scenario.msm.hmm(scenario.data.dtraj, nhidden=2)

    def test_n_states_property(self, setting):
        scenario = make_double_well(setting)
        assert_(scenario.msm.n_states <= scenario.msm.count_model.n_states_full)
        assert_equal(scenario.msm.n_states, scenario.n_states)

    def test_connected_sets(self, setting):
        scenario = make_double_well(setting)
        cs = scenario.msm.count_model.connected_sets()
        assert_equal(len(cs), 1)
        # mode largest: re-evaluating connected_sets should yield one connected set with exactly as many states as
        # contained in the count model
        assert_equal(cs[0], np.arange(scenario.msm.count_model.n_states))

    def test_count_matrix(self, setting):
        scenario = make_double_well(setting)
        count_matrix_full = scenario.msm.count_model.count_matrix_full
        n = np.max(scenario.data.dtraj) + 1
        assert_equal(count_matrix_full.shape, (n, n))

        count_matrix = scenario.msm.count_model.count_matrix
        assert_equal(count_matrix.shape, (scenario.msm.n_states, scenario.msm.n_states))
        if hasattr(scenario.msm, 'state_fraction'):  # msm collection
            assert_equal(scenario.msm.state_fraction, scenario.msm.count_model.selected_state_fraction)
            assert_equal(scenario.msm.count_fraction, scenario.msm.count_model.selected_count_fraction)

    def test_discrete_trajectories_active(self, setting):
        scenario = make_double_well(setting)
        dta = scenario.msm.count_model.transform_discrete_trajectories_to_submodel(scenario.data.dtraj)
        assert_equal(len(dta), 1)
        # HERE: states are shifted down from the beginning, because early states are missing
        assert_(dta[0][0] < scenario.data.dtraj[0])

    def test_transition_matrix(self, setting):
        scenario = make_double_well(setting)
        msm = scenario.msm
        P = msm.transition_matrix
        # should be ndarray by default
        # assert (isinstance(P, np.ndarray))
        assert_(isinstance(P, np.ndarray) or isinstance(P, scipy.sparse.csr_matrix))
        # shape
        assert_equal(P.shape, (msm.n_states, msm.n_states))
        # test transition matrix properties
        import deeptime.markov.tools.analysis as msmana

        assert_(msmana.is_transition_matrix(P))
        assert_(msmana.is_connected(P))
        # REVERSIBLE
        if msm.reversible:
            assert_(msmana.is_reversible(P))

    def test_selected_count_fraction(self, setting):
        scenario = make_double_well(setting)
        # should always be a fraction
        assert_(0.0 <= scenario.msm.count_model.selected_count_fraction <= 1.0)
        # special case for this data set:
        assert_equal(scenario.msm.count_model.selected_count_fraction, scenario.selected_count_fraction)

    def test_selected_state_fraction(self, setting):
        scenario = make_double_well(setting)
        # should always be a fraction
        assert_(0.0 <= scenario.msm.count_model.selected_state_fraction <= 1.0)

    def test_statdist(self, setting):
        scenario = make_double_well(setting)
        mu = scenario.msm.stationary_distribution
        # should strictly positive (irreversibility)
        assert_(np.all(mu > 0))
        # should sum to one
        assert_almost_equal(np.sum(mu), 1., decimal=10)

        # in case it was an ML estimate with fixed stationary distribution it should be reproduced
        if isinstance(scenario.msm_estimator, MaximumLikelihoodMSM) \
                and scenario.msm_estimator.stationary_distribution_constraint is not None:
            assert_array_almost_equal(
                scenario.msm.stationary_distribution,
                scenario.stationary_distribution[scenario.msm.count_model.state_symbols]
            )

    def test_eigenvalues(self, setting):
        scenario = make_double_well(setting)
        # use n_states-2 because sparse eigenvalue problem can only be solved by scipy for k < N-1
        ev = scenario.msm.eigenvalues(scenario.msm.n_states - 2)
        # stochasticity
        assert_(np.max(np.abs(ev)) <= 1 + 1e-12)
        # irreducible
        assert_(np.max(np.abs(ev[1:])) < 1)
        # ordered?
        evabs = np.abs(ev)
        for i in range(0, len(evabs) - 1):
            assert_(evabs[i] >= evabs[i + 1])
        # REVERSIBLE:
        if scenario.msm.reversible:
            assert_(np.all(np.isreal(ev)))

    def test_eigenvectors(self, setting):
        scenario = make_double_well(setting)
        msm = scenario.msm
        if not msm.sparse:
            k = msm.n_states
            L = msm.eigenvectors_left()
            D = np.diag(msm.eigenvalues())
            R = msm.eigenvectors_right()
        else:
            k = 4  # maximum scipy can handle for sparse matrices
            L = msm.eigenvectors_left(k)
            D = np.diag(msm.eigenvalues(k))
            R = msm.eigenvectors_right(k)
        # shape should be right
        assert_equal(L.shape, (k, msm.n_states))
        assert_equal(R.shape, (msm.n_states, k))
        # eigenvector properties
        assert_array_almost_equal(L[0, :], msm.stationary_distribution, err_msg="should be identical to stat. dist")
        assert_array_almost_equal(R[:, 0], np.ones(msm.n_states), err_msg="should be all ones")
        assert_array_almost_equal(np.sum(L[1:, :], axis=1), np.zeros(k - 1), err_msg="sums should be 1, 0, 0, ...")
        if msm.sparse:
            eye = np.real_if_close(np.dot(L, R), tol=10000)
            assert_array_almost_equal(eye, np.eye(k), decimal=1, err_msg="orthogonality constraint")
        else:
            assert_array_almost_equal(np.dot(L, R), np.eye(k), err_msg="orthogonality constraint")
        # recover transition matrix
        transition_matrix = msm.transition_matrix
        if msm.sparse:
            transition_matrix = transition_matrix.toarray()
            assert_array_almost_equal(np.dot(R, np.dot(D, L)), transition_matrix, decimal=0)
        else:
            assert_array_almost_equal(np.dot(R, np.dot(D, L)), transition_matrix)
        # REVERSIBLE:
        if msm.reversible:
            assert_(np.all(np.isreal(L)))
            assert_(np.all(np.isreal(R)))
            mu = msm.stationary_distribution
            L_mu = mu[:, np.newaxis] * R
            assert_array_almost_equal(np.dot(L_mu.T, R), np.eye(k))

    def test_timescales(self, setting):
        scenario = make_double_well(setting)
        if scenario.statdist_constraint:
            pytest.skip("timescales reference values only valid without constrained stationary distribution")
        msm = scenario.msm
        if not msm.sparse:
            ts = msm.timescales()
        else:
            k = 4
            ts = msm.timescales(k)

        # should be all positive
        assert_(np.all(ts > 0))
        if msm.reversible:
            # REVERSIBLE: should be all real
            assert_(np.all(np.isreal(ts)))
        assert_almost_equal(ts[:len(scenario.timescales)], scenario.timescales, decimal=2)

    def test_score(self, setting):
        scenario = make_double_well(setting)
        if isinstance(scenario.msm, AugmentedMSM):
            pytest.skip("scoring not implemented for augmented MSMs.")
        dtrajs_test = scenario.data.dtraj[80000:]
        scenario.msm.score(dtrajs_test)
        s1 = scenario.msm.score(dtrajs_test, r=1, dim=2)
        assert_(1.0 <= s1 <= 2.0)

        s2 = scenario.msm.score(dtrajs_test, r=2, dim=2)
        assert_(1.0 <= s2 <= 2.0)

    # ---------------------------------
    # FIRST PASSAGE PROBLEMS
    # ---------------------------------

    def test_committor(self, setting):
        scenario = make_double_well(setting)
        a = 16
        b = 48
        q_forward = scenario.msm.committor_forward(a, b)
        assert_equal(q_forward[a], 0)
        assert_equal(q_forward[b], 1)
        assert_(np.all(q_forward[:30] < 0.5))
        assert_(np.all(q_forward[40:] > 0.5))
        q_backward = scenario.msm.committor_backward(a, b)
        assert_equal(q_backward[a], 1)
        assert_equal(q_backward[b], 0)
        assert_(np.all(q_backward[:30] > 0.5))
        assert_(np.all(q_backward[40:] < 0.5))
        # REVERSIBLE:
        if scenario.msm.reversible:
            assert_(np.allclose(q_forward + q_backward, np.ones(scenario.msm.n_states)))

    def test_mfpt(self, setting):
        scenario = make_double_well(setting)
        if scenario.statdist_constraint:
            pytest.skip("timescales reference values only valid without constrained stationary distribution")
        a = 16
        b = 48
        t = scenario.msm.mfpt(a, b)
        assert_(t > 0)
        if isinstance(scenario.msm, AugmentedMSM):
            assert_allclose(t, 546.81, rtol=1e-3, atol=1e-6)
        else:
            if scenario.msm.reversible:
                assert_allclose(t, 872.69, rtol=1e-3, atol=1e-6)
            else:
                assert_allclose(t, 872.07, rtol=1e-3, atol=1e-6)

    def test_pcca(self, setting):
        scenario = make_double_well(setting)
        msm = scenario.msm
        if msm.reversible:
            pcca = msm.pcca(2)
            assignments = pcca.assignments
            # test: number of states
            assert_equal(len(assignments), msm.n_states)
            assert_equal(pcca.n_metastable, 2)
            # test: should be 0 or 1
            assert_(np.all(assignments >= 0))
            assert_(np.all(assignments <= 1))
            # should be equal (zero variance) within metastable sets
            assert_(np.std(assignments[:30]) == 0)
            assert_(np.std(assignments[40:]) == 0)

            pccadist = pcca.metastable_distributions
            # should be right size
            assert_equal(pccadist.shape, (2, msm.n_states))
            # should be nonnegative
            assert_(np.all(pccadist >= 0))
            # should roughly add up to stationary:
            cgdist = np.array([msm.stationary_distribution[pcca.sets[0]].sum(),
                               msm.stationary_distribution[pcca.sets[1]].sum()])
            ds = cgdist[0] * pccadist[0] + cgdist[1] * pccadist[1]
            ds /= ds.sum()
            assert_array_almost_equal(ds, msm.stationary_distribution, decimal=3)

            memberships = pcca.memberships
            # should be right size
            assert_equal(memberships.shape, (msm.n_states, 2))
            # should be nonnegative
            assert_(np.all(memberships >= 0))
            # should add up to one:
            assert_array_almost_equal(memberships.sum(axis=1), np.ones(msm.n_states))

            sets = pcca.sets
            assignment = pcca.assignments
            # should coincide with assignment
            for i, s in enumerate(sets):
                for j in range(len(s)):
                    assert (assignment[s[j]] == i)
        else:
            with assert_raises(ValueError):
                msm.pcca(2)

    def test_expectation(self, setting):
        scenario = make_double_well(setting)
        if scenario.statdist_constraint:
            pytest.skip("no reference value for statdist constraint case.")
        assert_almost_equal(scenario.msm.expectation(list(range(scenario.msm.n_states))), scenario.expectation,
                            decimal=2)

    def test_correlation(self, setting):
        scenario = make_double_well(setting)
        msm = scenario.msm
        k = msm.n_states if not msm.sparse else 4
        # raise assertion error because size is wrong:
        a = [1, 2, 3]
        with assert_raises(ValueError):
            msm.correlation(a, 1)
        maxtime = 100000
        # should decrease
        a = list(range(msm.n_states))
        times, corr1 = msm.correlation(a, maxtime=maxtime, k=k)
        assert_equal(len(corr1), maxtime / msm.lagtime)
        assert_equal(len(times), maxtime / msm.lagtime)
        assert_(corr1[0] > corr1[-1])
        a = list(range(msm.n_states))
        times, corr2 = msm.correlation(a, a, maxtime=maxtime, k=k)
        # should be identical to autocorr
        assert_almost_equal(corr1, corr2)
        # Test: should be increasing in time
        b = list(range(msm.n_states))[::-1]
        times, corr3 = msm.correlation(a, b, maxtime=maxtime, k=k)
        assert_equal(len(times), maxtime / msm.lagtime)
        assert_equal(len(corr3), maxtime / msm.lagtime)
        assert_(corr3[0] < corr3[-1])

    def test_relaxation(self, setting):
        scenario = make_double_well(setting)
        msm = scenario.msm
        if msm.sparse:
            k = 4
        else:
            k = msm.n_states
        pi_perturbed = (msm.stationary_distribution ** 2)
        pi_perturbed /= pi_perturbed.sum()
        a = list(range(msm.n_states))
        if isinstance(msm, AugmentedMSM):
            a = a[::-1]
        maxtime = 100000
        times, rel1 = msm.relaxation(msm.stationary_distribution, a, maxtime=maxtime, k=k)
        # should be constant because we are in equilibrium
        assert_array_almost_equal(rel1 - rel1[0], np.zeros((np.shape(rel1)[0])))
        times, rel2 = msm.relaxation(pi_perturbed, a, maxtime=maxtime, k=k)
        # should relax
        assert_equal(len(times), maxtime / msm.count_model.lagtime)
        assert_equal(len(rel2), maxtime / msm.count_model.lagtime)
        assert_(rel2[0] < rel2[-1])

    def test_fingerprint_correlation(self, setting):
        scenario = make_double_well(setting)
        msm = scenario.msm
        if msm.sparse:
            k = 4
        else:
            k = msm.n_states

        if msm.reversible:
            # raise assertion error because size is wrong:
            a = [1, 2, 3]
            with assert_raises(ValueError):
                msm.fingerprint_correlation(a, 1, k=k)
            # should decrease
            a = list(range(msm.n_states))
            fp1 = msm.fingerprint_correlation(a, k=k)
            # first timescale is infinite
            assert_equal(fp1[0][0], np.inf)
            # next timescales are identical to timescales:
            assert_array_almost_equal(fp1[0][1:], msm.timescales(k - 1))
            # all amplitudes nonnegative (for autocorrelation)
            assert_(np.all(fp1[1][:] >= 0))
            # identical call
            b = list(range(msm.n_states))
            fp2 = msm.fingerprint_correlation(a, b, k=k)
            assert_almost_equal(fp1[0], fp2[0])
            assert_almost_equal(fp1[1], fp2[1])
            # should be - of the above, apart from the first
            b = list(range(msm.n_states))[::-1]
            fp3 = msm.fingerprint_correlation(a, b, k=k)
            assert_almost_equal(fp1[0], fp3[0])
            assert_almost_equal(fp1[1][1:], -fp3[1][1:])
        else:  # raise ValueError, because fingerprints are not defined for nonreversible
            with assert_raises(ValueError):
                a = list(range(msm.n_states))
                msm.fingerprint_correlation(a, k=k)
            with assert_raises(ValueError):
                a = list(range(msm.n_states))
                b = list(range(msm.n_states))
                msm.fingerprint_correlation(a, b, k=k)

    def test_fingerprint_relaxation(self, setting):
        scenario = make_double_well(setting)
        msm = scenario.msm
        if msm.sparse:
            k = 4
        else:
            k = msm.n_states

        if msm.reversible:
            # raise assertion error because size is wrong:
            a = [1, 2, 3]
            with assert_raises(ValueError):
                msm.fingerprint_relaxation(msm.stationary_distribution, a, k=k)
            # equilibrium relaxation should be constant
            a = list(range(msm.n_states))
            fp1 = msm.fingerprint_relaxation(msm.stationary_distribution, a, k=k)
            # first timescale is infinite
            assert_equal(fp1[0][0], np.inf)
            # next timescales are identical to timescales:
            assert_array_almost_equal(fp1[0][1:], msm.timescales(k - 1))
            # dynamical amplitudes should be near 0 because we are in equilibrium
            assert_(np.max(np.abs(fp1[1][1:])) < 1e-10)
            # off-equilibrium relaxation
            pi_perturbed = (msm.stationary_distribution ** 2)
            pi_perturbed /= pi_perturbed.sum()
            fp2 = msm.fingerprint_relaxation(pi_perturbed, a, k=k)
            # first timescale is infinite
            assert_equal(fp2[0][0], np.inf)
            # next timescales are identical to timescales:
            assert_array_almost_equal(fp2[0][1:], msm.timescales(k - 1))
            # dynamical amplitudes should be significant because we are not in equilibrium
            assert_(np.max(np.abs(fp2[1][1:])) > 0.1)
        else:  # raise ValueError, because fingerprints are not defined for nonreversible
            with assert_raises(ValueError):
                a = list(range(msm.n_states))
                msm.fingerprint_relaxation(msm.stationary_distribution, a, k=k)
            with assert_raises(ValueError):
                pi_perturbed = (msm.stationary_distribution ** 2)
                pi_perturbed /= pi_perturbed.sum()
                a = list(range(msm.n_states))
                msm.fingerprint_relaxation(pi_perturbed, a)

    def test_active_state_indices(self, setting):
        scenario = make_double_well(setting)
        from deeptime.markov import sample
        I = sample.compute_index_states(scenario.data.dtraj, subset=scenario.msm.count_model.state_symbols)
        assert (len(I) == scenario.msm.n_states)
        # compare to histogram
        from deeptime.markov import count_states
        hist = count_states(scenario.data.dtraj)
        # number of frames should match on active subset
        A = scenario.msm.count_model.state_symbols
        for i in range(A.shape[0]):
            assert I[i].shape[0] == hist[A[i]]
            assert I[i].shape[1] == 2

    def test_trajectory_weights(self, setting):
        scenario = make_double_well(setting)
        weights = scenario.msm.compute_trajectory_weights(scenario.data.dtraj)
        assert_almost_equal(weights[0].sum(), 1., decimal=6, err_msg="Weights should sum up to 1")

    def test_simulate(self, setting):
        msm = make_double_well(setting).msm
        N = 400
        start = 1
        traj = msm.simulate(n_steps=N, start=start)
        assert_(len(traj) <= N)
        assert_(len(np.unique(traj)) <= msm.n_states)
        assert_equal(start, traj[0])

    # ----------------------------------
    # MORE COMPLEX TESTS / SANITY CHECKS
    # ----------------------------------
    def test_two_state_kinetics(self, setting):
        msm = make_double_well(setting).msm
        if msm.sparse:
            k = 4
        else:
            k = msm.n_states
        # sanity check: k_forward + k_backward = 1.0/t2 for the two-state process
        l2 = msm.eigenvectors_left(k)[1, :]
        core1 = np.argmin(l2)
        core2 = np.argmax(l2)
        # transition time from left to right and vice versa
        t12 = msm.mfpt(core1, core2)
        t21 = msm.mfpt(core2, core1)
        # relaxation time
        t2 = msm.timescales(k)[0]
        # the following should hold roughly = k12 + k21 = k2.
        # sum of forward/backward rates can be a bit smaller because we are using small cores and
        # therefore underestimate rates
        ksum = 1.0 / t12 + 1.0 / t21
        k2 = 1.0 / t2
        assert_almost_equal(k2, ksum, decimal=3)
