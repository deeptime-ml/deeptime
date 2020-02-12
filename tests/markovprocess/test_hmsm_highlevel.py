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

import unittest

import numpy as np
from sktime.markovprocess.generation import _markovprocess_generation_bindings as generation_bindings


class TestHMMReconstruction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # generate observations
        n_steps = int(1e6)
        cls.T_hidden = np.array([[0.7, 0.2, 0.1],
                                 [0.1, 0.8, 0.1],
                                 [0.1, 0.2, 0.7]])
        cls.n_hidden = cls.T_hidden.shape[0]
        n_obs_per_hidden_state = 5
        cls.n_observable = cls.n_hidden * n_obs_per_hidden_state

        def gaussian(x, mu, sigma):
            prop = 1 / np.sqrt(2. * np.pi * sigma ** 2) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
            return prop / prop.sum()

        cls.observed_alphabet = np.arange(cls.n_observable)
        cls.output_probabilities = np.array([gaussian(cls.observed_alphabet, mu, 2.5) for mu in
                                             np.arange((n_obs_per_hidden_state - 1) // 2,
                                                       cls.n_observable, n_obs_per_hidden_state)])

        cls.hidden_state_traj = generation_bindings.trajectory(n_steps, 0, cls.T_hidden)
        cls.observable_state_traj = np.zeros_like(cls.hidden_state_traj) - 1
        for state in range(cls.n_hidden):
            ix = np.where(cls.hidden_state_traj == state)[0]
            cls.observable_state_traj[ix] = np.random.choice(cls.n_observable,
                                                             p=cls.output_probabilities[state],
                                                             size=ix.shape[0])
        assert -1 not in np.unique(cls.observable_state_traj)

        # TODO: estimate hmm

    def test_observation_probabilities(self):
        pass

    def test_stationary_distribution(self):
        # compare against pi(self.T_hidden)
        pass

    def test_hidden_transition_matrix(self):
        pass

    def test_hidden_path(self):
        # assert that viterbi path comes close to self.hidden_stat_traj
        pass

    def test_heuristics(self):
        # can we do some rudimentary checks on the initial guess?
        pass


if __name__ == "__main__":
    unittest.main()
