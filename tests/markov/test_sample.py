import unittest

import numpy as np
import pytest
from numpy.testing import assert_equal, assert_array_almost_equal

from deeptime.markov import sample
from tests.testing_utilities import timing


@unittest.skip("performance test")
def test_performance():
    import pyemma.util.discrete_trajectories as dt
    state = np.random.RandomState(42)
    n_states = 10000
    dtrajs = [state.randint(0, n_states, size=100000) for _ in range(500)]

    selection = np.random.choice(np.arange(n_states), size=(500,), replace=False)
    with timing('pyemma'):
        out2 = dt.index_states(dtrajs, selection)
    with timing('cpp'):
        out = sample.compute_index_states(dtrajs, selection)

    assert_equal(len(out), len(out2))
    for o1, o2 in zip(out, out2):
        assert_array_almost_equal(o1, o2)


def test_onetraj():
    dtraj = [0, 1, 2, 3, 2, 1, 0]
    # should be a ValueError because this is not a subset
    res = sample.compute_index_states(dtraj)
    expected = [np.array([[0, 0], [0, 6]]), np.array([[0, 1], [0, 5]]), np.array([[0, 2], [0, 4]]),
                np.array([[0, 3]])]
    assert (len(res) == len(expected))
    for i in range(len(res)):
        assert (res[i].shape == expected[i].shape)
        assert (np.alltrue(res[i] == expected[i]))


def test_onetraj_sub():
    dtraj = [0, 1, 2, 3, 2, 1, 0]
    # should be a ValueError because this is not a subset
    res = sample.compute_index_states(dtraj, subset=[2, 3])
    expected = [np.array([[0, 2], [0, 4]]), np.array([[0, 3]])]
    assert (len(res) == len(expected))
    for i in range(len(res)):
        assert (res[i].shape == expected[i].shape)
        assert (np.alltrue(res[i] == expected[i]))


def test_twotraj():
    dtrajs = [[0, 1, 2, 3, 2, 1, 0], [3, 4, 5]]
    # should be a ValueError because this is not a subset
    res = sample.compute_index_states(dtrajs)
    expected = [np.array([[0, 0], [0, 6]]), np.array([[0, 1], [0, 5]]), np.array([[0, 2], [0, 4]]),
                np.array([[0, 3], [1, 0]]), np.array([[1, 1]]), np.array([[1, 2]])]
    assert (len(res) == len(expected))
    for i in range(len(res)):
        assert (res[i].shape == expected[i].shape)
        assert (np.alltrue(res[i] == expected[i]))


def test_sample_by_sequence():
    dtraj = [0, 1, 2, 3, 2, 1, 0]
    idx = sample.compute_index_states(dtraj)
    seq = [0, 1, 1, 1, 0, 0, 0, 0, 1, 1]
    sidx = sample.indices_by_sequence(idx, seq)
    assert (np.alltrue(sidx.shape == (len(seq), 2)))
    for t in range(sidx.shape[0]):
        assert (sidx[t, 0] == 0)  # did we pick the right traj?
        assert (dtraj[sidx[t, 1]] == seq[t])  # did we pick the right states?


@pytest.mark.parametrize("replace", [True, False])
def test_sample_by_state_replace(replace):
    dtraj = [0, 1, 2, 3, 5, 5, 3, 2, 1, 0]
    idx = sample.compute_index_states(dtraj)
    sidx = sample.indices_by_state(idx, 5, replace=replace)
    for i in range(4):
        if replace:
            assert (sidx[i].shape[0] == 5)
        else:
            assert (sidx[i].shape[0] == 2)
        for t in range(sidx[i].shape[0]):
            assert (dtraj[sidx[i][t, 1]] == i)


def test_sample_by_state_replace_subset():
    dtraj = [0, 1, 2, 3, 2, 1, 0]
    idx = sample.compute_index_states(dtraj)
    subset = [1, 2]
    sidx = sample.indices_by_state(idx, 5, subset=subset)
    for i in range(len(subset)):
        assert (sidx[i].shape[0] == 5)
        for t in range(sidx[i].shape[0]):
            assert (dtraj[sidx[i][t, 1]] == subset[i])


def test_index_states_subset():
    dtraj = [55, 66, 77]
    subset = [33, 44, 55, 66, 77, 88]
    indices = sample.compute_index_states(dtraj, subset=subset)
    assert_equal(len(indices), len(subset))
    assert_equal(indices[0].size, 0)
    assert_equal(indices[1].size, 0)
    assert_equal(indices[2].size, 2)
    assert_equal(indices[3].size, 2)
    assert_equal(indices[4].size, 2)
    assert_equal(indices[5].size, 0)
    assert_equal(indices[2], [[0, 0]])
    assert_equal(indices[3], [[0, 1]])
    assert_equal(indices[4], [[0, 2]])
