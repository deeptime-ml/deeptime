""" Unittests for the deeptime.markov.msm.MarkovStateModel """

import numpy as np
import pytest
import deeptime
from deeptime.markov import count_states


@pytest.fixture
def msm():
    return deeptime.markov.msm.MarkovStateModel([[0.9, 0.1], [0.1, 0.9]])


def test_simulate(msm):
    N = 1000
    traj = msm.simulate(n_steps=N, start=0, seed=42)

    # test shapes and sizes
    assert traj.size == N
    assert traj.min() >= 0
    assert traj.max() <= 1

    # test statistics of transition matrix
    C = deeptime.markov.tools.estimation.count_matrix(traj, 1)
    Pest = deeptime.markov.tools.estimation.transition_matrix(C)
    assert np.max(np.abs(Pest - msm.transition_matrix)) < 0.025


def test_simulate_stats(msm):
    # test statistics of starting state
    N = 5000
    trajs = [msm.simulate(1, seed=i+1) for i in range(N)]
    ss = np.concatenate(trajs).astype(int)
    pi = deeptime.markov.tools.analysis.stationary_distribution(msm.transition_matrix)
    piest = count_states(ss) / float(N)
    np.testing.assert_allclose(piest, pi, atol=0.025)


def test_simulate_recover_transition_matrix(msm):
    # test if transition matrix can be reconstructed
    N = 5000
    trajs = msm.simulate(N, seed=42)
    # trajs = msmgen.generate_traj(self.P, N, random_state=self.random_state)
    C = deeptime.markov.tools.estimation.count_matrix(trajs, 1, sparse_return=False)
    T = deeptime.markov.tools.estimation.transition_matrix(C)
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
