import pytest
import numpy as np
import scipy
from numpy.testing import *

from sktime.markov.msm import MaximumLikelihoodMSM


def test_reversible_property(double_well_all):
    assert_equal(double_well_all.msm_estimator.reversible, double_well_all.msm.reversible)


def test_sparse_property(double_well_all):
    assert_equal(double_well_all.msm_estimator.sparse, double_well_all.msm.sparse)


def test_lagtime_property(double_well_all):
    assert_equal(double_well_all.msm.lagtime, double_well_all.lagtime)


def test_state_symbols(double_well_all):
    # should always be <= full set
    assert_(len(double_well_all.msm.count_model.state_symbols) <= double_well_all.msm.count_model.n_states_full)
    # should be length of n_states
    assert_equal(len(double_well_all.msm.count_model.state_symbols), double_well_all.msm.count_model.n_states)


def test_n_states_property(double_well_all):
    assert_(double_well_all.msm.n_states <= double_well_all.msm.count_model.n_states_full)
    assert_equal(double_well_all.msm.n_states, double_well_all.n_states)


def test_connected_sets(double_well_all):
    cs = double_well_all.msm.count_model.connected_sets()
    assert_equal(len(cs), 1)
    # mode largest: re-evaluating connected_sets should yield one connected set with exactly as many states as
    # contained in the count model
    assert_equal(cs[0], np.arange(double_well_all.msm.count_model.n_states))


def test_count_matrix(double_well_all):
    count_matrix_full = double_well_all.msm.count_model.count_matrix_full
    n = np.max(double_well_all.data.dtraj) + 1
    assert_equal(count_matrix_full.shape, (n, n))

    count_matrix = double_well_all.msm.count_model.count_matrix
    assert_equal(count_matrix.shape, (double_well_all.msm.n_states, double_well_all.msm.n_states))


def test_discrete_trajectories_active(double_well_all):
    dta = double_well_all.msm.count_model.transform_discrete_trajectories_to_submodel(double_well_all.data.dtraj)
    assert_equal(len(dta), 1)
    # HERE: states are shifted down from the beginning, because early states are missing
    assert_(dta[0][0] < double_well_all.data.dtraj[0])


def test_physical_time(double_well_all):
    assert_(str(double_well_all.msm.count_model.physical_time).startswith('1'))
    assert_(str(double_well_all.msm.count_model.physical_time).endswith('step'))


def test_transition_matrix(double_well_all):
    msm = double_well_all.msm
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


def test_selected_count_fraction(double_well_all):
    # should always be a fraction
    assert_(0.0 <= double_well_all.msm.count_model.selected_count_fraction <= 1.0)
    # special case for this data set:
    assert_equal(double_well_all.msm.count_model.selected_count_fraction, double_well_all.selected_count_fraction)


def test_selected_state_fraction(double_well_all):
    # should always be a fraction
    assert_(0.0 <= double_well_all.msm.count_model.selected_state_fraction <= 1.0)


def test_statdist(double_well_all):
    mu = double_well_all.msm.stationary_distribution
    # should strictly positive (irreversibility)
    assert_(np.all(mu > 0))
    # should sum to one
    assert_almost_equal(np.sum(mu), 1., decimal=10)

    # in case it was an ML estimate with fixed stationary distribution it should be reproduced
    if isinstance(double_well_all.msm_estimator, MaximumLikelihoodMSM) \
            and double_well_all.msm_estimator.stationary_distribution_constraint is not None:
        assert_array_almost_equal(
            double_well_all.msm.stationary_distribution,
            double_well_all.stationary_distribution[double_well_all.msm.count_model.state_symbols]
        )


def test_eigenvalues(double_well_all):
    # use n_states-2 because sparse eigenvalue problem can only be solved by scipy for k < N-1
    ev = double_well_all.msm.eigenvalues(double_well_all.msm.n_states - 2)
    # stochasticity
    assert_(np.max(np.abs(ev)) <= 1 + 1e-12)
    # irreducible
    assert_(np.max(np.abs(ev[1:])) < 1)
    # ordered?
    evabs = np.abs(ev)
    for i in range(0, len(evabs) - 1):
        assert_(evabs[i] >= evabs[i + 1])
    # REVERSIBLE:
    if double_well_all.msm.reversible:
        assert_(np.all(np.isreal(ev)))


def test_eigenvectors(double_well_all):
    msm = double_well_all.msm
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


@pytest.mark.parametrize("msm_type", ["MLMSM", "AMM"])
@pytest.mark.parametrize("reversible", [True, False])
@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("count_mode", ["sliding", "sample"])
def test_timescales(make_double_well_data, msm_type, reversible, sparse, count_mode):
    scenario = make_double_well_data(msm_type, reversible=reversible, statdist_constraint=False, sparse=sparse,
                                     count_mode=count_mode)
    est = scenario.msm_estimator
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
