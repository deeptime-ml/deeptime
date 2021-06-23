""" Unittests for the deeptime.markov.msm.MarkovStateModel """

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_raises, assert_equal, assert_

from deeptime.data import BirthDeathChain
from deeptime.markov import count_states, TransitionCountModel
from deeptime.markov.msm import MaximumLikelihoodMSM, MarkovStateModel
from deeptime.markov.tools.analysis import stationary_distribution
from deeptime.markov.tools.estimation import count_matrix, transition_matrix


@pytest.fixture
def msm():
    return MarkovStateModel([[0.9, 0.1], [0.1, 0.9]],
                            count_model=TransitionCountModel([[90, 10], [10, 90]]))


def test_simulate(msm):
    N = 1000
    traj = msm.simulate(n_steps=N, start=0, seed=42)

    # test shapes and sizes
    assert traj.size == N
    assert traj.min() >= 0
    assert traj.max() <= 1

    # test statistics of transition matrix
    C = count_matrix(traj, 1)
    Pest = transition_matrix(C)
    assert np.max(np.abs(Pest - msm.transition_matrix)) < 0.025


def test_simulate_stats(msm):
    # test statistics of starting state
    N = 5000
    trajs = [msm.simulate(1, seed=i + 1) for i in range(N)]
    ss = np.concatenate(trajs).astype(int)
    pi = stationary_distribution(msm.transition_matrix)
    piest = count_states(ss) / float(N)
    np.testing.assert_allclose(piest, pi, atol=0.025)
    assert_(msm.stationary)


def test_simulate_recover_transition_matrix(msm):
    # test if transition matrix can be reconstructed
    N = 5000
    trajs = msm.simulate(N, seed=42)
    # trajs = msmgen.generate_traj(self.P, N, random_state=self.random_state)
    C = count_matrix(trajs, 1, sparse_return=False)
    T = transition_matrix(C)
    np.testing.assert_allclose(T, msm.transition_matrix, atol=.01)


def test_simulate_stop_eq_start(msm):
    M = 10
    N = 10
    trajs = [msm.simulate(N, start=0, stop=0) for _ in range(M)]
    for traj in trajs:
        assert traj.size == 1


def test_simulate_with_stop(msm):
    # test if we always stop at stopping state
    M = 100
    N = 10
    stop = 1
    trajs = [msm.simulate(N, start=0, stop=stop, seed=42) for _ in range(M)]
    for traj in trajs:
        assert traj.size == N or traj[-1] == stop
        assert stop not in traj[:-1]


def test_empirical_vs_ground_truth_koopman_model():
    bdc = BirthDeathChain([0, .5, .5], [.5, .5, 0.])
    dtraj = bdc.msm.simulate(10000)
    est = MaximumLikelihoodMSM(reversible=True, stationary_distribution_constraint=bdc.stationary_distribution,
                               lagtime=1)
    msm_ref = est.fit_fetch(dtraj)
    assert_almost_equal(bdc.msm.koopman_model.score(r=2), msm_ref.score(r=2), decimal=2)


def test_update_transition_matrix():
    msm = MarkovStateModel([[1., 0.], [0., 1.]])
    with assert_raises(ValueError):
        msm.update_transition_matrix(np.array([[1., np.inf], [0., 1.]]))
    with assert_raises(ValueError):
        msm.update_transition_matrix(np.array([[1., .1], [0., 1.]]))
    with assert_raises(ValueError):
        msm.update_transition_matrix(None)


def test_submodel(msm):
    sub_msm = msm.submodel([1])
    assert_equal(sub_msm.n_states, 1)
    assert_equal(sub_msm.count_model.state_symbols, [1])
    assert_equal(sub_msm.count_model.states, [0])
    assert_equal(sub_msm. transition_matrix, [[1.]])

    with assert_raises(ValueError):
        msm.submodel([0, 5])  # state 5 does not exist
