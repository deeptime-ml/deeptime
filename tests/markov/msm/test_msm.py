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


r"""Unit test for the MSM module

.. moduleauthor:: F. Noe <frank DOT noe AT fu-berlin DOT de>
.. moduleauthor:: B. Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import itertools
import unittest
import warnings

import msmtools.analysis as msmana
import msmtools.estimation as msmest
import numpy as np
import pytest
import scipy.sparse
from msmtools.generation import generate_traj
from msmtools.util.birth_death_chain import BirthDeathChain

from sktime.markov._base import score_cv
from sktime.markov.msm import BayesianMSM, MaximumLikelihoodMSM, MarkovStateModel
from sktime.markov.transition_counting import TransitionCountEstimator
from sktime.markov.util import count_states
from tests.util import GenerateTestMatrix
from numpy.testing import *


def estimate_markov_model(dtrajs, lag, return_estimator=False, **kw) -> MarkovStateModel:
    statdist_constraint = kw.pop('statdist', None)
    connectivity = kw.pop('connectivity_threshold', 0.)
    sparse = kw.pop('sparse', False)
    count_model = TransitionCountEstimator(lagtime=lag, count_mode="sliding", sparse=sparse).fit(dtrajs).fetch_model()
    count_model = count_model.submodel_largest(probability_constraint=statdist_constraint,
                                               connectivity_threshold=connectivity)
    est = MaximumLikelihoodMSM(stationary_distribution_constraint=statdist_constraint, sparse=sparse, **kw)
    est.fit(count_model)
    if return_estimator:
        return est, est.fetch_model()
    return est.fetch_model()


@pytest.mark.parametrize("reversible", [True, False])
@pytest.mark.parametrize("statdist", [None, np.array([0.5, 0.5]), np.array([1.1, .5]), np.array([.1, .1]),
                                      np.array([-.1, .5])])
@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("maxiter", [1])
@pytest.mark.parametrize("maxerr", [1e-3])
def test_estimator_params(reversible, statdist, sparse, maxiter, maxerr):
    if statdist is not None and (np.any(statdist > 1) or np.any(statdist < 0)):
        with np.testing.assert_raises(ValueError):
            MaximumLikelihoodMSM(reversible=reversible, stationary_distribution_constraint=statdist,
                                 sparse=sparse, maxiter=maxiter, maxerr=maxerr)
    else:
        msm = MaximumLikelihoodMSM(reversible=reversible, stationary_distribution_constraint=statdist,
                                   sparse=sparse, maxiter=maxiter, maxerr=maxerr)
        assert_equal(msm.reversible, reversible)
        assert_equal(msm.stationary_distribution_constraint,
                     statdist / np.sum(statdist) if statdist is not None else None)
        assert_equal(msm.sparse, sparse)
        assert_equal(msm.maxiter, maxiter)
        assert_equal(msm.maxerr, maxerr)


def test_weakly_connected_count_matrix():
    count_matrix = np.array([[10, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1], [0, 0, 0, 1]], dtype=np.float32)
    with assert_raises(BaseException, msg="count matrix not strongly connected, expected failure in rev. case"):
        MaximumLikelihoodMSM().fit(count_matrix)
    # count matrix weakly connected, this should work
    msm = MaximumLikelihoodMSM(reversible=False).fit(count_matrix).fetch_model()
    assert_equal(msm.reversible, False)
    assert_equal(msm.n_states, 4)
    assert_equal(msm.lagtime, 1)
    assert_(msm.count_model is not None)
    assert_equal(msm.count_model.count_matrix, count_matrix)
    # last state is sink state
    assert_equal(msm.stationary_distribution, [0, 0, 0, 1])
    assert_array_almost_equal(msm.transition_matrix,
                              [[10. / 11, 1. / 11, 0, 0],
                               [0, 0.5, 0.5, 0],
                               [0, 1. / 3, 1. / 3, 1. / 3],
                               [0, 0, 0, 1]])
    assert_equal(msm.n_eigenvalues, 4)
    assert_equal(msm.sparse, False)

    msm = msm.submodel(np.array([1, 2]))
    assert_equal(msm.reversible, False)
    assert_equal(msm.n_states, 2)
    assert_equal(msm.count_model.state_symbols, [1, 2])
    assert_equal(msm.lagtime, 1)
    assert_equal(msm.count_model.count_matrix, [[1, 1], [1, 1]])
    assert_equal(msm.stationary_distribution, [0.5, 0.5])
    assert_array_almost_equal(msm.transition_matrix, [[0.5, 0.5], [0.5, 0.5]])
    assert_equal(msm.n_eigenvalues, 2)
    assert_equal(msm.sparse, False)


def test_strongly_connected_count_matrix():
    # transitions 6->1->2->3->4->6, disconnected are 0 and 5
    dtraj = np.array([0, 6, 1, 2, 3, 4, 6, 5])
    counts = TransitionCountEstimator(lagtime=1, count_mode="sliding").fit(dtraj).fetch_model()
    assert_equal(counts.n_states, 7)
    sets = counts.connected_sets(directed=True)
    assert_equal(len(sets), 3)
    assert_equal(len(sets[0]), 5)
    with assert_raises(BaseException, msg="count matrix not strongly connected, expected failure in rev. case"):
        MaximumLikelihoodMSM().fit(counts)
    counts = counts.submodel_largest(directed=True)  # now we are strongly connected
    # due to reversible we get 6<->1<->2<->3<->4<->6
    msm = MaximumLikelihoodMSM(reversible=True).fit(counts).fetch_model()
    # check that the msm has symbols 1,2,3,4,6
    assert_(np.all([i in msm.count_model.state_symbols for i in [1, 2, 3, 4, 6]]))
    assert_equal(msm.reversible, True)
    assert_equal(msm.n_states, 5)
    assert_equal(msm.lagtime, 1)
    assert_array_almost_equal(msm.transition_matrix, [
        [0., .5, 0., 0., .5],
        [.5, 0., .5, 0., 0.],
        [0., .5, 0., .5, 0.],
        [0., 0., .5, 0., .5],
        [.5, 0., 0., .5, 0.]
    ])
    assert_array_almost_equal(msm.stationary_distribution, [1. / 5] * 5)
    assert_equal(msm.n_eigenvalues, 5)
    assert_equal(msm.sparse, False)

    msm = msm.submodel(np.array([3, 4]))  # states 3 and 4 correspond to symbols 4 and 6
    assert_equal(msm.reversible, True)
    assert_equal(msm.n_states, 2)
    assert_equal(msm.lagtime, 1)
    assert_array_almost_equal(msm.transition_matrix, [[0, 1.], [1., 0]])
    assert_array_almost_equal(msm.stationary_distribution, [0.5, 0.5])
    assert_equal(msm.n_eigenvalues, 2)
    assert_equal(msm.sparse, False)
    assert_equal(msm.count_model.state_symbols, [4, 6])


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_birth_death_chain(fixed_seed, sparse):
    """Meta-stable birth-death chain"""
    b = 2
    q = np.zeros(7)
    p = np.zeros(7)
    q[1:] = 0.5
    p[0:-1] = 0.5
    q[2] = 1.0 - 10 ** (-b)
    q[4] = 10 ** (-b)
    p[2] = 10 ** (-b)
    p[4] = 1.0 - 10 ** (-b)

    bdc = BirthDeathChain(q, p)
    P = bdc.transition_matrix()
    dtraj = generate_traj(P, 10000, start=0)
    tau = 1

    reference_count_matrix = msmest.count_matrix(dtraj, tau, sliding=True)
    reference_largest_connected_component = msmest.largest_connected_set(reference_count_matrix)
    reference_lcs = msmest.largest_connected_submatrix(reference_count_matrix,
                                                       lcc=reference_largest_connected_component)
    reference_msm = msmest.transition_matrix(reference_lcs, reversible=True, maxerr=1e-8)
    reference_statdist = msmana.stationary_distribution(reference_msm)
    k = 3
    reference_timescales = msmana.timescales(reference_msm, k=k, tau=tau)

    msm = estimate_markov_model(dtraj, tau, sparse=sparse)
    assert_equal(tau, msm.count_model.lagtime)
    assert_array_equal(reference_largest_connected_component, msm.count_model.connected_sets()[0])
    assert_(scipy.sparse.issparse(msm.count_model.count_matrix) == sparse)
    assert_(scipy.sparse.issparse(msm.transition_matrix) == sparse)
    if sparse:
        count_matrix = msm.count_model.count_matrix.toarray()
        transition_matrix = msm.transition_matrix.toarray()
    else:
        count_matrix = msm.count_model.count_matrix
        transition_matrix = msm.transition_matrix
    assert_array_almost_equal(reference_lcs.toarray(), count_matrix)
    assert_array_almost_equal(reference_count_matrix.toarray(), count_matrix)
    assert_array_almost_equal(reference_msm.toarray(), transition_matrix)
    assert_array_almost_equal(reference_statdist, msm.stationary_distribution)
    assert_array_almost_equal(reference_timescales[1:], msm.timescales(k - 1))


class TestMSMRevPi(unittest.TestCase):
    r"""Checks if the MLMSM correctly handles the active set computation
    if a stationary distribution is given"""

    def test_valid_stationary_vector(self):
        dtraj = np.array([0, 0, 1, 0, 1, 2])
        pi_valid = np.array([0.1, 0.9, 0.0])
        pi_invalid = np.array([0.1, 0.9])
        active_set = np.array([0, 1])
        msm = estimate_markov_model(dtraj, 1, statdist=pi_valid)
        np.testing.assert_equal(msm.count_model.state_symbols, active_set)
        with self.assertRaises(ValueError):
            estimate_markov_model(dtraj, 1, statdist=pi_invalid)

    def test_valid_trajectory(self):
        pi = np.array([0.1, 0.0, 0.9])
        dtraj_invalid = np.array([1, 1, 1, 1, 1, 1, 1])
        dtraj_valid = np.array([0, 2, 0, 2, 2, 0, 1, 1])
        msm = estimate_markov_model(dtraj_valid, lag=1, statdist=pi)
        np.testing.assert_equal(msm.count_model.state_symbols, np.array([0, 2]))
        with self.assertRaises(ValueError):
            estimate_markov_model(dtraj_invalid, lag=1, statdist=pi)


def test_score(double_well_msm_all):
    scenario, est, msm = double_well_msm_all
    dtrajs_test = scenario.dtraj[80000:]
    msm.score(dtrajs_test)
    s1 = msm.score(dtrajs_test, score_method='VAMP1', score_k=2)
    np.testing.assert_(1.0 <= s1 <= 2.0)

    s2 = msm.score(dtrajs_test, score_method='VAMP2', score_k=2)
    np.testing.assert_(1.0 <= s2 <= 2.0)


def test_score_cv(double_well_msm_all):
    scenario, est, msm = double_well_msm_all

    def fit_fetch(dtrajs):
        count_model = TransitionCountEstimator(lagtime=10, count_mode="sliding").fit(dtrajs) \
            .fetch_model().submodel_largest()
        return est.fit(count_model).fetch_model()

    s1 = score_cv(fit_fetch, dtrajs=scenario.dtraj, lagtime=10, n=5, score_method='VAMP1', score_k=2).mean()
    assert 1.0 <= s1 <= 2.0
    s2 = score_cv(fit_fetch, dtrajs=scenario.dtraj, lagtime=10, n=5, score_method='VAMP2', score_k=2).mean()
    assert 1.0 <= s2 <= 2.0
    se = score_cv(fit_fetch, dtrajs=scenario.dtraj, lagtime=10, n=5, score_method='VAMPE', score_k=2).mean()
    se_inf = score_cv(fit_fetch, dtrajs=scenario.dtraj, lagtime=10, n=5, score_method='VAMPE', score_k=None).mean()


def test_reversible_property(double_well_msm_all):
    _, est, msm = double_well_msm_all
    assert_equal(est.reversible, msm.reversible)


def test_sparse_property(double_well_msm_all):
    _, est, msm = double_well_msm_all
    assert_equal(est.sparse, msm.sparse)


def test_lagtime_property(double_well_msm_all):
    scenario, est, msm = double_well_msm_all
    assert_equal(msm.lagtime, scenario.lagtime)


def test_state_symbols(double_well_msm_all):
    scenario, est, msm = double_well_msm_all
    # should always be <= full set
    assert_(len(msm.count_model.state_symbols) <= msm.count_model.n_states_full)
    # should be length of n_states
    assert_equal(len(msm.count_model.state_symbols), msm.count_model.n_states)


def test_n_states_property(double_well_msm_all):
    scenario, est, msm = double_well_msm_all
    assert_(msm.n_states <= msm.count_model.n_states)
    assert_equal(msm.n_states, scenario.n_states)


def test_connected_sets(double_well_msm_all):
    scenario, est, msm = double_well_msm_all
    cs = msm.count_model.connected_sets()
    assert_equal(len(cs), 1)
    # mode largest: re-evaluating connected_sets should yield one connected set with exactly as many states as
    # contained in the count model
    assert_equal(cs[0], np.arange(msm.count_model.n_states))


@pytest.mark.parametrize("full", [False, True], ids=["not full", "full"])
def test_count_matrix(double_well_msm_all, full):
    scenario, est, msm = double_well_msm_all
    if full:
        count_matrix = msm.count_model.count_matrix_full
        n = np.max(scenario.dtraj) + 1
        assert_equal(count_matrix.shape, (n, n))
    else:
        count_matrix = msm.count_model.count_matrix
        assert_equal(count_matrix.shape, (msm.n_states, msm.n_states))


def test_discrete_trajectories_active(double_well_msm_all):
    scenario, est, msm = double_well_msm_all
    dta = msm.count_model.transform_discrete_trajectories_to_submodel(scenario.dtraj)
    assert_equal(len(dta), 1)
    # HERE: states are shifted down from the beginning, because early states are missing
    assert_(dta[0][0] < scenario.dtraj[0])


def test_physical_time(double_well_msm_all):
    _, _, msm = double_well_msm_all
    assert_(str(msm.count_model.physical_time).startswith('1'))
    assert_(str(msm.count_model.physical_time).endswith('step'))


def test_transition_matrix(double_well_msm_all):
    scenario, est, msm = double_well_msm_all
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


def test_selected_count_fraction(double_well_msm_all):
    scenario, est, msm = double_well_msm_all
    # should always be a fraction
    assert_(0.0 <= msm.count_model.selected_count_fraction <= 1.0)
    # special case for this data set:
    assert_equal(msm.count_model.selected_count_fraction, scenario.selected_count_fraction)


def test_selected_state_fraction(double_well_msm_all):
    scenario, est, msm = double_well_msm_all
    # should always be a fraction
    assert_(0.0 <= msm.count_model.selected_state_fraction <= 1.0)


def test_statdist(double_well_msm_all):
    scenario, est, msm = double_well_msm_all
    mu = msm.stationary_distribution
    # should strictly positive (irreversibility)
    assert_(np.all(mu > 0))
    # should sum to one
    assert_almost_equal(np.sum(mu), 1., decimal=10)

    if est.stationary_distribution_constraint is not None:
        assert_array_almost_equal(msm.stationary_distribution,
                                  scenario.stationary_distribution[msm.count_model.state_symbols])


def test_eigenvalues(double_well_msm_all):
    scenario, est, msm = double_well_msm_all
    # use n_states-2 because sparse eigenvalue problem can only be solved by scipy for k < N-1
    ev = msm.eigenvalues(msm.n_states - 2)
    # stochasticity
    assert_(np.max(np.abs(ev)) <= 1 + 1e-12)
    # irreducible
    assert_(np.max(np.abs(ev[1:])) < 1)
    # ordered?
    evabs = np.abs(ev)
    for i in range(0, len(evabs) - 1):
        assert_(evabs[i] >= evabs[i + 1])
    # REVERSIBLE:
    if msm.reversible:
        assert_(np.all(np.isreal(ev)))


def test_eigenvectors(double_well_msm_all):
    scenario, est, msm = double_well_msm_all
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


def test_timescales(double_well_msm_nostatdist_constraint):
    scenario, est, msm = double_well_msm_nostatdist_constraint
    if not msm.sparse:
        ts = msm.timescales()
    else:
        k = 4
        ts = msm.timescales(k)

    # should be all positive
    assert_(np.all(ts > 0))
    assert_(est.stationary_distribution_constraint is None, msg="fixture only yields scenarios w/o statdist constraint")
    if msm.reversible:
        # REVERSIBLE: should be all real
        ts_ref = np.array([310.87, 8.5, 5.09])
        assert_(np.all(np.isreal(ts)))
        assert_almost_equal(ts[:3], ts_ref, decimal=2)
    else:
        ts_ref = np.array([310.49376926, 8.48302712, 5.02649564])
        assert_almost_equal(ts[:3], ts_ref, decimal=2)


def test_committor(double_well_msm_all):
    scenario, est, msm = double_well_msm_all
    a = 16
    b = 48
    q_forward = msm.committor_forward(a, b)
    assert_equal(q_forward[a], 0)
    assert_equal(q_forward[b], 1)
    assert_(np.all(q_forward[:30] < 0.5))
    assert_(np.all(q_forward[40:] > 0.5))
    q_backward = msm.committor_backward(a, b)
    assert_equal(q_backward[a], 1)
    assert_equal(q_backward[b], 0)
    assert_(np.all(q_backward[:30] > 0.5))
    assert_(np.all(q_backward[40:] < 0.5))
    # REVERSIBLE:
    if msm.reversible:
        assert (np.allclose(q_forward + q_backward, np.ones(msm.n_states)))


def test_mfpt(double_well_msm_nostatdist_constraint):
    scenario, est, msm = double_well_msm_nostatdist_constraint
    a = 16
    b = 48
    t = msm.mfpt(a, b)
    assert (t > 0)
    if msm.reversible:
        np.testing.assert_allclose(t, 872.69, rtol=1e-3, atol=1e-6)
    else:
        np.testing.assert_allclose(t, 872.07, rtol=1e-3, atol=1e-6)


def test_pcca(double_well_msm_nostatdist_constraint):
    scenario, est, msm = double_well_msm_nostatdist_constraint
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


def test_expectation(double_well_msm_nostatdist_constraint):
    scenario, est, msm = double_well_msm_nostatdist_constraint
    assert_almost_equal(msm.expectation(list(range(msm.n_states))), 31.73, decimal=2)


def test_correlation(double_well_msm_nostatdist_constraint):
    scenario, est, msm = double_well_msm_nostatdist_constraint
    if msm.sparse:
        k = 4
    else:
        k = msm.n_states
    # raise assertion error because size is wrong:
    maxtime = 100000
    a = [1, 2, 3]
    with assert_raises(AssertionError):
        msm.correlation(a, 1)
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


def test_relaxation(double_well_msm_nostatdist_constraint):
    scenario, est, msm = double_well_msm_nostatdist_constraint
    if msm.sparse:
        k = 4
    else:
        k = msm.n_states
    pi_perturbed = (msm.stationary_distribution ** 2)
    pi_perturbed /= pi_perturbed.sum()
    a = list(range(msm.n_states))
    maxtime = 100000
    times, rel1 = msm.relaxation(msm.stationary_distribution, a, maxtime=maxtime, k=k)
    # should be constant because we are in equilibrium
    assert_array_almost_equal(rel1 - rel1[0], np.zeros((np.shape(rel1)[0])))
    times, rel2 = msm.relaxation(pi_perturbed, a, maxtime=maxtime, k=k)
    # should relax
    assert_equal(len(times), maxtime / msm.count_model.lagtime)
    assert_equal(len(rel2), maxtime / msm.count_model.lagtime)
    assert_(rel2[0] < rel2[-1])


def test_fingerprint_correlation(double_well_msm_nostatdist_constraint):
    scenario, est, msm = double_well_msm_nostatdist_constraint
    if msm.sparse:
        k = 4
    else:
        k = msm.n_states

    if msm.reversible:
        # raise assertion error because size is wrong:
        a = [1, 2, 3]
        with assert_raises(AssertionError):
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


def test_fingerprint_relaxation(double_well_msm_nostatdist_constraint):
    scenario, est, msm = double_well_msm_nostatdist_constraint
    if msm.sparse:
        k = 4
    else:
        k = msm.n_states

    if msm.reversible:
        # raise assertion error because size is wrong:
        a = [1, 2, 3]
        with assert_raises(AssertionError):
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


def test_active_state_indices(double_well_msm_all):
    scenario, est, msm = double_well_msm_all
    from sktime.markov.sample import compute_index_states
    I = compute_index_states(scenario.dtraj, subset=msm.count_model.state_symbols)
    assert (len(I) == msm.n_states)
    # compare to histogram
    hist = count_states(scenario.dtraj)
    # number of frames should match on active subset
    A = msm.count_model.state_symbols
    for i in range(A.shape[0]):
        assert I[i].shape[0] == hist[A[i]]
        assert I[i].shape[1] == 2


def test_trajectory_weights(double_well_msm_all):
    scenario, est, msm = double_well_msm_all
    weights = msm.compute_trajectory_weights(scenario.dtraj)
    assert_almost_equal(weights[0].sum(), 1., decimal=6, err_msg="Weights should sum up to 1")


def test_simulate(double_well_msm_all):
    msm = double_well_msm_all[2]
    N = 400
    start = 1
    traj = msm.simulate(N=N, start=start)
    assert_(len(traj) <= N)
    assert_(len(np.unique(traj)) <= msm.n_states)
    assert_equal(start, traj[0])


def test_two_state_kinetics(double_well_msm_all):
    scenario, est, msm = double_well_msm_all
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


class TestMSMMinCountConnectivity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dtraj = np.array(
            [0, 3, 0, 1, 2, 3, 0, 0, 1, 0, 1, 0, 3, 1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 3, 0, 0, 3, 3, 0, 0, 1, 1, 3, 0,
             1, 0, 0, 1, 0, 0, 0, 0, 3, 0, 1, 0, 3, 2, 1, 0, 3, 1, 0, 1, 0, 1, 0, 3, 0, 0, 3, 0, 0, 0, 2, 0, 0, 3,
             0, 1, 0, 0, 0, 0, 3, 3, 3, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 3, 3, 1, 0, 0, 0, 2, 1, 3, 0, 0])
        assert (dtraj == 2).sum() == 5  # state 2 has only 5 counts,
        cls.dtraj = dtraj
        cls.mincount_connectivity = 6  # state 2 will be kicked out by this choice.
        cls.active_set_unrestricted = np.array([0, 1, 2, 3])
        cls.active_set_restricted = np.array([0, 1, 3])

    def test_msm(self):
        msm_one_over_n = estimate_markov_model(self.dtraj, lag=1, connectivity_threshold='1/n')
        msm_restrict_connectivity = estimate_markov_model(self.dtraj, lag=1,
                                                          connectivity_threshold=self.mincount_connectivity)
        np.testing.assert_equal(msm_one_over_n.count_model.state_symbols, self.active_set_unrestricted)
        np.testing.assert_equal(msm_restrict_connectivity.count_model.state_symbols, self.active_set_restricted)

    # TODO: move to test_bayesian_msm
    def test_bmsm(self):
        cc = TransitionCountEstimator(lagtime=1, count_mode="effective").fit(self.dtraj).fetch_model()
        msm = BayesianMSM().fit(cc.submodel_largest(connectivity_threshold='1/n')).fetch_model()
        msm_restricted = BayesianMSM().fit(cc.submodel_largest(connectivity_threshold=self.mincount_connectivity)) \
            .fetch_model()

        np.testing.assert_equal(msm.prior.count_model.state_symbols, self.active_set_unrestricted)
        np.testing.assert_equal(msm.samples[0].count_model.state_symbols, self.active_set_unrestricted)
        np.testing.assert_equal(msm_restricted.prior.count_model.state_symbols, self.active_set_restricted)
        np.testing.assert_equal(msm_restricted.samples[0].count_model.state_symbols, self.active_set_restricted)
        i = id(msm_restricted.prior.count_model)
        assert all(id(x.count_model) == i for x in msm_restricted.samples)


class TestMSMSimplePathologicalCases(unittest.TestCase, metaclass=GenerateTestMatrix):
    """
    example that covers disconnected states handling
    2 <- 0 <-> 1 <-> 3 - 7 -> 4 <-> 5 - 6

    """
    lag_reversible_countmode_params = [dict(lag=lag, reversible=r, count_mode=cm)
                                       for lag, r, cm in itertools.product(
            [1, 2], [True, False], ['sliding', 'sample'])]
    lag_countmode_params = [dict(lag=lag, count_mode=cm)
                            for lag, cm in itertools.product(
            [1, 2], ['sliding', 'sample'])]
    params = {
        '_test_msm_submodel_statdist': lag_reversible_countmode_params,
        '_test_raises_disconnected': lag_countmode_params,
        '_test_msm_invalid_statdist_constraint': [dict(reversible=True, count_mode='ulrich')],
        '_test_connected_sets': lag_countmode_params,
        '_test_sub_counts': [dict(count_mode=cm)
                             for cm in ['sliding', 'sample']]
    }

    @classmethod
    def setUpClass(cls):
        dtrajs = [np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 2, 2, 2, 2]),
                  np.array([0, 1, 1, 0, 0, 3, 3, 3, 0, 1, 3, 1, 3, 0, 3, 3, 1, 1]),
                  np.array([4, 5, 5, 5, 4, 4, 5, 5, 4, 4, 5, 4, 4, 4, 5, 4, 5, 5]),
                  np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]),
                  np.array([7, 7, 7, 7, 7, 4, 5, 4, 5, 4, 5, 4, 4, 4, 5, 5, 5, 4])]
        cls.dtrajs = dtrajs
        cls.connected_sets = [[0, 1, 3], [4, 5], [2], [6], [7]]
        cmat_set1 = np.array([[3, 7, 2],
                              [6, 3, 2],
                              [2, 2, 3]], dtype=np.int)
        cmat_set2 = np.array([[6, 9],
                              [8, 6]], dtype=np.int)
        cls.count_matrices = [cmat_set1, cmat_set2, None, None, None]

    def _test_connected_sets(self, lag, count_mode):
        count_model = TransitionCountEstimator(lagtime=lag, count_mode=count_mode).fit(self.dtrajs).fetch_model()
        assert all([set(c) in set(map(frozenset, self.connected_sets)) for c in count_model.connected_sets()])

    def _test_sub_counts(self, count_mode):
        count_model = TransitionCountEstimator(lagtime=1, count_mode=count_mode).fit(self.dtrajs).fetch_model()
        for cset, cmat_ref in zip(count_model.connected_sets(), self.count_matrices):
            submodel = count_model.submodel(cset)
            self.assertEqual(len(submodel.connected_sets()), 1)
            self.assertEqual(len(submodel.connected_sets()[0]), len(cset))
            self.assertEqual(submodel.count_matrix.shape[0], len(cset))

            if cmat_ref is not None:
                np.testing.assert_array_equal(submodel.count_matrix, cmat_ref)

    def _test_msm_submodel_statdist(self, lag, reversible, count_mode):
        count_model = TransitionCountEstimator(lagtime=lag, count_mode=count_mode).fit(self.dtrajs).fetch_model()

        for cset in count_model.connected_sets():
            submodel = count_model.submodel(cset)
            estimator = MaximumLikelihoodMSM(reversible=reversible).fit(submodel)
            msm = estimator.fetch_model()
            C = submodel.count_matrix
            P = C / np.sum(C, axis=-1)[:, None]

            import scipy.linalg as salg
            eigval, eigvec = salg.eig(P, left=True, right=False)

            pi = np.real(eigvec)[:, np.where(np.real(eigval) > 1. - 1e-3)[0]].squeeze()
            if np.any(pi < 0):
                pi *= -1.
            pi = pi / np.sum(pi)
            np.testing.assert_array_almost_equal(msm.stationary_distribution, pi,
                                                 decimal=1, err_msg="Failed for cset {} with "
                                                                    "cmat {}".format(cset, submodel.count_matrix))

    def _test_msm_invalid_statdist_constraint(self, reversible, count_mode):
        pass  # TODO: fix code to pass test
        # pi = np.ones(4) / 4.
        # for cset in self.count_model.connected_sets():
        #     submodel = self.count_model.submodel(cset)
        #
        #     with self.assertRaises(RuntimeError):
        #         MaximumLikelihoodMSM(reversible=reversible, stationary_distribution_constraint=pi).fit(submodel)

    def _test_raises_disconnected(self, lag, count_mode):
        count_model = TransitionCountEstimator(lagtime=lag, count_mode=count_mode).fit(self.dtrajs).fetch_model()

        with self.assertRaises(AssertionError):
            MaximumLikelihoodMSM(reversible=True).fit(count_model)

        non_reversibly_connected_set = [0, 1, 2, 3]
        submodel = count_model.submodel(non_reversibly_connected_set)

        with self.assertRaises(AssertionError):
            MaximumLikelihoodMSM(reversible=True).fit(submodel)

        fully_disconnected_set = [6, 2]
        submodel = count_model.submodel(fully_disconnected_set)
        with self.assertRaises(AssertionError):
            MaximumLikelihoodMSM(reversible=True).fit(submodel)


if __name__ == "__main__":
    unittest.main()
