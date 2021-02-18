import unittest

import numpy as np

from deeptime.markov import sample
from tests.testing_utilities import timing


class TestSampleIndices(unittest.TestCase):

    @unittest.skip("performance test")
    def test_performance(self):
        import pyemma.util.discrete_trajectories as dt
        state = np.random.RandomState(42)
        n_states = 10000
        dtrajs = [state.randint(0, n_states, size=100000) for _ in range(500)]

        selection = np.random.choice(np.arange(n_states), size=(500,), replace=False)
        with timing('pyemma'):
            out2 = dt.index_states(dtrajs, selection)
        with timing('cpp'):
            out = sample.compute_index_states(dtrajs, selection)

        assert len(out) == len(out2)
        for o1, o2 in zip(out, out2):
            np.testing.assert_array_almost_equal(o1, o2)

    def test_subset_error(self):
        dtraj = [0, 1, 2, 3, 2, 1, 0]
        # should be a ValueError because this is not a subset
        with self.assertRaises(ValueError):
            sample.compute_index_states(dtraj, subset=[3, 4, 5])

    def test_onetraj(self):
        dtraj = [0, 1, 2, 3, 2, 1, 0]
        # should be a ValueError because this is not a subset
        res = sample.compute_index_states(dtraj)
        expected = [np.array([[0, 0], [0, 6]]), np.array([[0, 1], [0, 5]]), np.array([[0, 2], [0, 4]]),
                    np.array([[0, 3]])]
        assert (len(res) == len(expected))
        for i in range(len(res)):
            assert (res[i].shape == expected[i].shape)
            assert (np.alltrue(res[i] == expected[i]))

    def test_onetraj_sub(self):
        dtraj = [0, 1, 2, 3, 2, 1, 0]
        # should be a ValueError because this is not a subset
        res = sample.compute_index_states(dtraj, subset=[2, 3])
        expected = [np.array([[0, 2], [0, 4]]), np.array([[0, 3]])]
        assert (len(res) == len(expected))
        for i in range(len(res)):
            assert (res[i].shape == expected[i].shape)
            assert (np.alltrue(res[i] == expected[i]))

    def test_twotraj(self):
        dtrajs = [[0, 1, 2, 3, 2, 1, 0], [3, 4, 5]]
        # should be a ValueError because this is not a subset
        res = sample.compute_index_states(dtrajs)
        expected = [np.array([[0, 0], [0, 6]]), np.array([[0, 1], [0, 5]]), np.array([[0, 2], [0, 4]]),
                    np.array([[0, 3], [1, 0]]), np.array([[1, 1]]), np.array([[1, 2]])]
        assert (len(res) == len(expected))
        for i in range(len(res)):
            assert (res[i].shape == expected[i].shape)
            assert (np.alltrue(res[i] == expected[i]))

    def test_sample_by_sequence(self):
        dtraj = [0, 1, 2, 3, 2, 1, 0]
        idx = sample.compute_index_states(dtraj)
        seq = [0, 1, 1, 1, 0, 0, 0, 0, 1, 1]
        sidx = sample.indices_by_sequence(idx, seq)
        assert (np.alltrue(sidx.shape == (len(seq), 2)))
        for t in range(sidx.shape[0]):
            assert (sidx[t, 0] == 0)  # did we pick the right traj?
            assert (dtraj[sidx[t, 1]] == seq[t])  # did we pick the right states?

    def test_sample_by_state_replace(self):
        dtraj = [0, 1, 2, 3, 2, 1, 0]
        idx = sample.compute_index_states(dtraj)
        sidx = sample.indices_by_state(idx, 5)
        for i in range(4):
            assert (sidx[i].shape[0] == 5)
            for t in range(sidx[i].shape[0]):
                assert (dtraj[sidx[i][t, 1]] == i)

    def test_sample_by_state_replace_subset(self):
        dtraj = [0, 1, 2, 3, 2, 1, 0]
        idx = sample.compute_index_states(dtraj)
        subset = [1, 2]
        sidx = sample.indices_by_state(idx, 5, subset=subset)
        for i in range(len(subset)):
            assert (sidx[i].shape[0] == 5)
            for t in range(sidx[i].shape[0]):
                assert (dtraj[sidx[i][t, 1]] == subset[i])
