# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group
#
# MSMTools is free software: you can redistribute it and/or modify
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

'''
@author: noe, trendelkampschroer
'''
import unittest
import numpy as np
import msmtools.generation as msmgen
import msmtools.estimation as msmest
import msmtools.analysis as msmana


class TestTrajGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.P = np.array([[0.9, 0.1],
                          [0.1, 0.9]])

    def setUp(self):
        self.random_state = np.random.RandomState(42)

    def test_trajectory(self):
        N = 1000
        traj = msmgen.generate_traj(self.P, N, start=0, random_state=self.random_state)

        # test shapes and sizes
        assert traj.size == N
        assert traj.min() >= 0
        assert traj.max() <= 1

        # test statistics of transition matrix
        C = msmest.count_matrix(traj, 1)
        Pest = msmest.transition_matrix(C)
        assert np.max(np.abs(Pest - self.P)) < 0.025

    def test_trajectories(self):
        # test number of trajectories
        M = 10
        N = 10
        trajs = msmgen.generate_trajs(self.P, M, N, start=0, random_state=self.random_state)
        assert len(trajs) == M

    def test_stats(self):
        # test statistics of starting state
        N = 5000
        trajs = msmgen.generate_trajs(self.P, N, 1, random_state=self.random_state)
        ss = np.concatenate(trajs).astype(int)
        pi = msmana.stationary_distribution(self.P)
        piest = msmest.count_states(ss) / float(N)
        np.testing.assert_allclose(piest, pi, atol=0.025)

    def test_transitionmatrix(self):
        # test if transition matrix can be reconstructed
        N = 5000
        trajs = msmgen.generate_traj(self.P, N, random_state=self.random_state)
        C = msmest.count_matrix(trajs, 1, sparse_return=False)
        T = msmest.transition_matrix(C)
        np.testing.assert_allclose(T, self.P, atol=.01)

    def test_stop_eq_start(self):
        M = 10
        N = 10
        trajs = msmgen.generate_trajs(self.P, M, N, start=0, stop=0, random_state=self.random_state)
        for traj in trajs:
            assert traj.size == 1

    def test_stop(self):
        # test if we always stop at stopping state
        M = 100
        N = 10
        stop = 1
        trajs = msmgen.generate_trajs(self.P, M, N, start=0, stop=stop, random_state=self.random_state)
        for traj in trajs:
            assert traj.size == N or traj[-1] == stop
            assert stop not in traj[:-1]

if __name__ == "__main__":
    unittest.main()
