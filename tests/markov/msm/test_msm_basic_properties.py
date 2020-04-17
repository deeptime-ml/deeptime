import pytest
import numpy as np
import scipy
from numpy.testing import *

from markov.msm.util import MLMSM_PARAMS, AMM_PARAMS, MLMSM_IDS, AMM_IDS, make_double_well
from sktime.markov.msm import MaximumLikelihoodMSM
from sktime.markov.msm.augmented_msm import AugmentedMSM


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

    def test_discrete_trajectories_active(self, setting):
        scenario = make_double_well(setting)
        dta = scenario.msm.count_model.transform_discrete_trajectories_to_submodel(scenario.data.dtraj)
        assert_equal(len(dta), 1)
        # HERE: states are shifted down from the beginning, because early states are missing
        assert_(dta[0][0] < scenario.data.dtraj[0])

    def test_physical_time(self, setting):
        scenario = make_double_well(setting)
        assert_(str(scenario.msm.count_model.physical_time).startswith('1'))
        assert_(str(scenario.msm.count_model.physical_time).endswith('step'))

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
        import msmtools.analysis as msmana

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
        s1 = scenario.msm.score(dtrajs_test, score_method='VAMP1', score_k=2)
        assert_(1.0 <= s1 <= 2.0)

        s2 = scenario.msm.score(dtrajs_test, score_method='VAMP2', score_k=2)
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