r"""Unit test for the MSM module

.. moduleauthor:: F. Noe <frank DOT noe AT fu-berlin DOT de>
.. moduleauthor:: B. Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import collections
import unittest

from numpy.testing import assert_equal, assert_raises, assert_, assert_array_almost_equal, assert_array_equal

import deeptime.markov.tools.analysis as msmana
import deeptime.markov.tools.estimation as msmest
import numpy as np
import pytest
import scipy.sparse

import deeptime
from deeptime.decomposition import vamp_score_cv
from deeptime.markov.msm import BayesianMSM, MaximumLikelihoodMSM, MarkovStateModel, MarkovStateModelCollection
from deeptime.markov import TransitionCountEstimator, TransitionCountModel


def estimate_markov_model(dtrajs, lag, **kw) -> MarkovStateModel:
    statdist_constraint = kw.pop('statdist', None)
    connectivity = kw.pop('connectivity_threshold', 0.)
    sparse = kw.pop('sparse', False)
    count_model = TransitionCountEstimator(lagtime=lag, count_mode="sliding", sparse=sparse).fit(dtrajs).fetch_model()
    count_model = count_model.submodel_largest(probability_constraint=statdist_constraint,
                                               connectivity_threshold=connectivity)
    est = MaximumLikelihoodMSM(stationary_distribution_constraint=statdist_constraint, sparse=sparse, **kw)
    est.fit(count_model)
    return est.fetch_model()


@pytest.mark.parametrize("reversible", [True, False])
@pytest.mark.parametrize("statdist", [None, np.array([0.5, 0.5]), np.array([1.1, .5]), np.array([.1, .1]),
                                      np.array([-.1, .5])])
@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("maxiter", [1])
@pytest.mark.parametrize("maxerr", [1e-3])
def test_estimator_params(reversible, statdist, sparse, maxiter, maxerr):
    if statdist is not None and (np.any(statdist > 1) or np.any(statdist < 0)):
        with assert_raises(ValueError):
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
    assert_equal(MaximumLikelihoodMSM().fit(count_matrix).fetch_model().n_connected_msms, 3,
                 err_msg="Count matrix not strongly connected, should decay into three sets.")
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

    bdc = deeptime.data.birth_death_chain(q, p)
    dtraj = bdc.msm.simulate(10000, start=0)
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
        assert_equal(msm.count_model.state_symbols, active_set)
        with self.assertRaises(ValueError):
            estimate_markov_model(dtraj, 1, statdist=pi_invalid)

    def test_valid_trajectory(self):
        pi = np.array([0.1, 0.0, 0.9])
        dtraj_invalid = np.array([1, 1, 1, 1, 1, 1, 1])
        dtraj_valid = np.array([0, 2, 0, 2, 2, 0, 1, 1])
        msm = estimate_markov_model(dtraj_valid, lag=1, statdist=pi)
        assert_equal(msm.count_model.state_symbols, np.array([0, 2]))
        with self.assertRaises(ValueError):
            estimate_markov_model(dtraj_invalid, lag=1, statdist=pi)


class FF:

    def __init__(self, est):
        self._est = est

    def __call__(self, dtrajs):
        count_model = TransitionCountEstimator(lagtime=10, count_mode="sliding", n_states=85).fit(dtrajs) \
            .fetch_model().submodel_largest()
        return self._est.fit(count_model).fetch_model()


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_score_cv(double_well_msm_all, n_jobs):
    scenario, est, msm = double_well_msm_all
    est.lagtime = 10

    fit_fetch = FF(est)

    with assert_raises(ValueError):
        vamp_score_cv(fit_fetch, trajs=scenario.dtraj, lagtime=10, n=5, r=1, dim=2, n_jobs=1, splitting_mode="noop")
    with assert_raises(ValueError):
        vamp_score_cv(fit_fetch, trajs=scenario.dtraj)  # uses blocksplit but no lagtime
    s1 = vamp_score_cv(fit_fetch, trajs=scenario.dtraj, lagtime=10, n=5, r=1, dim=2, n_jobs=n_jobs).mean()
    assert 1.0 <= s1 <= 2.0
    s2 = vamp_score_cv(fit_fetch, trajs=scenario.dtraj, lagtime=10, n=5, r=2, dim=2, n_jobs=n_jobs).mean()
    assert 1.0 <= s2 <= 2.0
    se = vamp_score_cv(fit_fetch, trajs=scenario.dtraj, lagtime=10, n=5, r="E", dim=2, n_jobs=n_jobs).mean()
    se_inf = vamp_score_cv(fit_fetch, trajs=scenario.dtraj, lagtime=10, n=5, r="E", dim=None, n_jobs=n_jobs).mean()


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
        assert_equal(msm_one_over_n.count_model.state_symbols, self.active_set_unrestricted)
        assert_equal(msm_restrict_connectivity.count_model.state_symbols, self.active_set_restricted)

    # TODO: move to test_bayesian_msm
    def test_bmsm(self):
        cc = TransitionCountEstimator(lagtime=1, count_mode="effective").fit(self.dtraj).fetch_model()
        msm = BayesianMSM().fit(cc.submodel_largest(connectivity_threshold='1/n')).fetch_model()
        msm_restricted = BayesianMSM().fit(cc.submodel_largest(connectivity_threshold=self.mincount_connectivity)) \
            .fetch_model()

        assert_equal(msm.prior.count_model.state_symbols, self.active_set_unrestricted)
        assert_equal(msm.samples[0].count_model.state_symbols, self.active_set_unrestricted)
        assert_equal(msm_restricted.prior.count_model.state_symbols, self.active_set_restricted)
        assert_equal(msm_restricted.samples[0].count_model.state_symbols, self.active_set_restricted)
        i = id(msm_restricted.prior.count_model)
        assert all(id(x.count_model) == i for x in msm_restricted.samples)


DisconnectedStatesScenario = collections.namedtuple(
    "DisconnectedStatesScenario", ["dtrajs", "connected_sets", "weakly_connected_sets", "count_matrices"]
)


@pytest.fixture
def disconnected_states():
    """
    example that covers disconnected states handling
    2 <- 0 <-> 1 <-> 3 - 7 -> 4 <-> 5 - 6
    """
    dtrajs = [np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 2, 2, 2, 2]),
              np.array([0, 1, 1, 0, 0, 3, 3, 3, 0, 1, 3, 1, 3, 0, 3, 3, 1, 1]),
              np.array([4, 5, 5, 5, 4, 4, 5, 5, 4, 4, 5, 4, 4, 4, 5, 4, 5, 5]),
              np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]),
              np.array([7, 7, 7, 7, 7, 4, 5, 4, 5, 4, 5, 4, 4, 4, 5, 5, 5, 4])]
    connected_sets = [[0, 1, 3], [4, 5], [2], [6], [7]]
    weakly_connected_sets = [[0, 1, 2, 3], [4, 5, 7], [6]]
    cmat_set1 = np.array([[3, 7, 2],
                          [6, 3, 2],
                          [2, 2, 3]], dtype=np.int)
    cmat_set2 = np.array([[6, 9],
                          [8, 6]], dtype=np.int)
    count_matrices = [cmat_set1, cmat_set2, None, None, None]
    return DisconnectedStatesScenario(dtrajs=dtrajs, connected_sets=connected_sets,
                                      weakly_connected_sets=weakly_connected_sets, count_matrices=count_matrices)


@pytest.mark.parametrize("lag", [1, 2])
@pytest.mark.parametrize("count_mode", ["sliding", "sample"])
def test_connected_sets(disconnected_states, lag, count_mode):
    count_model = TransitionCountEstimator(lagtime=lag, count_mode=count_mode).fit(
        disconnected_states.dtrajs).fetch_model()
    assert all(
        [set(c) in set(map(frozenset, disconnected_states.connected_sets)) for c in count_model.connected_sets()])


@pytest.mark.parametrize("count_mode", ["sliding", "sample"])
def test_sub_counts(disconnected_states, count_mode):
    count_model = TransitionCountEstimator(lagtime=1, count_mode=count_mode) \
        .fit(disconnected_states.dtrajs).fetch_model()
    for cset, cmat_ref in zip(count_model.connected_sets(), disconnected_states.count_matrices):
        submodel = count_model.submodel(cset)
        assert_equal(len(submodel.connected_sets()), 1)
        assert_equal(len(submodel.connected_sets()[0]), len(cset))
        assert_equal(submodel.count_matrix.shape[0], len(cset))

        if cmat_ref is not None:
            assert_array_equal(submodel.count_matrix, cmat_ref)


@pytest.mark.parametrize("lag", [1, 2])
@pytest.mark.parametrize("reversible", [True, False])
@pytest.mark.parametrize("count_mode", ["sliding", "sample"])
def test_msm_submodel_statdist(disconnected_states, lag, reversible, count_mode):
    count_model = TransitionCountEstimator(lagtime=lag, count_mode=count_mode).fit(
        disconnected_states.dtrajs).fetch_model()

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
        assert_array_almost_equal(msm.stationary_distribution, pi,
                                  decimal=1, err_msg="Failed for cset {} with "
                                                     "cmat {}".format(cset, submodel.count_matrix))


@pytest.mark.parametrize("lagtime", [1, 2])
@pytest.mark.parametrize("reversible", [True, False])
@pytest.mark.parametrize("count_mode", ["sliding", "sample"])
def test_msm_invalid_statdist_constraint(disconnected_states, lagtime, reversible, count_mode):
    pi = np.ones(4) / 4.
    count_model = TransitionCountEstimator(lagtime=lagtime, count_mode=count_mode) \
        .fit(disconnected_states.dtrajs).fetch_model()
    for cset in count_model.connected_sets():
        submodel = count_model.submodel(cset)

        with assert_raises(ValueError):
            MaximumLikelihoodMSM(reversible=reversible, stationary_distribution_constraint=pi).fit(submodel)


@pytest.mark.parametrize("lag", [1, 2])
@pytest.mark.parametrize("count_mode", ["sliding", "sample"])
def test_reversible_disconnected(disconnected_states, lag, count_mode):
    r"""disconnected states: 2 <- 0 <-> 1 <-> 3 | 7 -> 4 <-> 5 | 6"""
    count_model = TransitionCountEstimator(lagtime=lag, count_mode=count_mode) \
        .fit(disconnected_states.dtrajs).fetch_model()

    msm = MaximumLikelihoodMSM(reversible=True).fit(count_model).fetch_model()
    assert_equal(msm.n_connected_msms, len(disconnected_states.connected_sets))
    for i, subset in enumerate(disconnected_states.connected_sets):
        # can do this because subsets are ordered in decreasing cardinality
        assert_equal(msm.state_symbols(i), subset)

    non_reversibly_connected_set = [0, 1, 2, 3]
    submodel = count_model.submodel(non_reversibly_connected_set)

    msm = MaximumLikelihoodMSM(reversible=True).fit(submodel).fetch_model()
    assert_equal(msm.n_connected_msms, 2)
    assert_equal(msm.state_symbols(0), [0, 1, 3])
    assert_equal(msm.state_symbols(1), [2])

    fully_disconnected_set = [6, 2]
    submodel = count_model.submodel(fully_disconnected_set)
    msm = MaximumLikelihoodMSM(reversible=True).fit(submodel).fetch_model()
    assert_equal(msm.n_connected_msms, 2)
    assert_equal(msm.state_symbols(0), [6])
    assert_equal(msm.state_symbols(1), [2])


def test_nonreversible_disconnected():
    msm1 = MarkovStateModel([[.7, .3], [.3, .7]])
    msm2 = MarkovStateModel([[.9, .05, .05], [.3, .6, .1], [.1, .1, .8]])
    traj = np.concatenate([msm1.simulate(1000000), 2 + msm2.simulate(1000000)])
    counts = TransitionCountEstimator(lagtime=1, count_mode="sliding").fit(traj)
    msm = MaximumLikelihoodMSM(reversible=True).fit(counts).fetch_model()
    assert_equal(msm.transition_matrix.shape, (3, 3))
    assert_equal(msm.stationary_distribution.shape, (3,))
    assert_equal(msm.state_symbols(), [2, 3, 4])
    assert_equal(msm.state_symbols(1), [0, 1])
    msm.select(1)
    assert_equal(msm.transition_matrix.shape, (2, 2))
    assert_equal(msm.stationary_distribution.shape, (2,))
    assert_equal(msm.state_symbols(), [0, 1])
    assert_equal(msm.state_symbols(0), [2, 3, 4])
    with assert_raises(IndexError):
        msm.select(2)


def test_invalid_arguments():
    with assert_raises(ValueError):
        # negative counts
        MaximumLikelihoodMSM().fit(-1 * np.ones((5, 5))).fetch_model()
    with assert_raises(ValueError):
        # non quadratic count matrix
        MaximumLikelihoodMSM().fit(np.ones((3, 5))).fetch_model()
    with assert_raises(ValueError):
        # stationary distribution not over whole state space
        MaximumLikelihoodMSM(stationary_distribution_constraint=np.array([1 / 3, 1 / 3, 1 / 3])).fit(np.ones((5, 5)))
    with assert_raises(ValueError):
        # no counts but statdist constraint
        MaximumLikelihoodMSM(stationary_distribution_constraint=np.array([.5, .5])).fit(np.zeros((2, 2)))
    with assert_raises(ValueError):
        # fit with transition count estimator that hasn't been fit
        MaximumLikelihoodMSM().fit(TransitionCountEstimator(1, "sliding"))
    with assert_raises(ValueError):
        # fit with bogus object
        MaximumLikelihoodMSM().fit(object())
    with assert_raises(ValueError):
        # fit from timeseries without lagtime
        MaximumLikelihoodMSM().fit(np.array([0, 1, 2, 3, 4, 5, 6]))
    with assert_raises(ValueError):
        # empty collection is not allowed
        MarkovStateModelCollection([], [], False, [], 1.)
    with assert_raises(ValueError):
        # number of elements in lists must match
        MarkovStateModelCollection([np.array([[.5, .5], [.5, .5]])], [], False, [], 1.)
    with assert_raises(ValueError):
        # number of states in lists must match
        MarkovStateModelCollection([np.array([[.5, .5], [.5, .5]])], [None], False,
                                   [TransitionCountModel(np.ones((3, 3)))], 1.)
