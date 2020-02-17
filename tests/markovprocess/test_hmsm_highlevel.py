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
from sktime.markovprocess.hmm import MaximumLikelihoodHMSM
from sktime.markovprocess.hmm.maximum_likelihood_hmm import initial_guess_discrete_from_data
import itertools
import msmtools.analysis
from ..util import GenerateTestMatrix


parameter_options = {'reversible': [True, False],
                     'init_heuristics': [initial_guess_discrete_from_data],
                     'lagtime': [1]}

sorted_kwargs = sorted(parameter_options)
parameter_grid = list(itertools.product(*(parameter_options[key] for key in sorted_kwargs)))
kwarg_grid = [{k: v for k, v in zip(sorted_kwargs, param_tuple)} for param_tuple in
              parameter_grid]


def permutation_matrices(n):
    for mat in itertools.permutations(np.eye(n)):
        yield np.stack(mat)


def compile_test_signature(parameters):
    s = ''
    for n, p in zip(sorted_kwargs, parameters):
        s += f'{n}: {p} '
    return s


class TestHMMReconstruction(unittest.TestCase, metaclass=GenerateTestMatrix):
    global sorted_kwargs

    params = {
        '_test_observation_probabilities': kwarg_grid,
        '_test_stationary_distribution': kwarg_grid,
        '_test_hidden_transition_matrix': kwarg_grid,
        '_test_hidden_path': kwarg_grid
    }
    @classmethod
    def setUpClass(cls):
        # generate observations
        cls.n_steps = int(1e5)
        cls.T_hidden = np.array([[0.7, 0.2, 0.1],
                                 [0.1, 0.8, 0.1],
                                 [0.1, 0.2, 0.7]])

        cls.hidden_stationary_distribution = msmtools.analysis.stationary_distribution(cls.T_hidden)

        cls.n_hidden = cls.T_hidden.shape[0]
        n_obs_per_hidden_state = 5
        cls.n_observable = cls.n_hidden * n_obs_per_hidden_state

        def gaussian(x, mu, sigma):
            prop = 1 / np.sqrt(2. * np.pi * sigma ** 2) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
            return prop / prop.sum()

        cls.observed_alphabet = np.arange(cls.n_observable)
        cls.output_probabilities = np.array([gaussian(cls.observed_alphabet, mu, 2.) for mu in
                                             np.arange((n_obs_per_hidden_state - 1) // 2,
                                                       cls.n_observable, n_obs_per_hidden_state)])

        cls.hidden_state_traj = generation_bindings.trajectory(cls.n_steps, 0, cls.T_hidden)
        cls.observable_state_traj = np.zeros_like(cls.hidden_state_traj) - 1
        for state in range(cls.n_hidden):
            ix = np.where(cls.hidden_state_traj == state)[0]
            cls.observable_state_traj[ix] = np.random.choice(cls.n_observable,
                                                             p=cls.output_probabilities[state],
                                                             size=ix.shape[0])
        assert -1 not in np.unique(cls.observable_state_traj)

        def estimate_hmm(**kwargs):
            init_heuristics = kwargs.pop('init_heuristics')
            initial_hmm = init_heuristics(cls.observable_state_traj,
                                          n_hidden_states=cls.n_hidden,
                                          lagtime=kwargs['lagtime'])
            hmm = MaximumLikelihoodHMSM(initial_hmm, **kwargs).fit(cls.observable_state_traj).fetch_model()
            return hmm

        test_models = [estimate_hmm(**dict(zip(sorted_kwargs, param))) for param in parameter_grid]
        cls.models = {compile_test_signature(p):m for p, m in zip(parameter_grid, test_models)}

    def _test_observation_probabilities(self, **kwargs):
        test_sign = compile_test_signature([kwargs[key] for key in sorted_kwargs])
        model = self.models[test_sign]

        minerr = 1e6
        for perm in itertools.permutations(range(self.n_hidden)):
            err = np.max(np.abs(model.output_probabilities[np.array(perm)] -
                                self.output_probabilities))
            minerr = min(minerr, err)
        np.testing.assert_almost_equal(minerr, 0, decimal=2, err_msg=f'failed for {test_sign}')

    def _test_stationary_distribution(self, **kwargs):
        test_sign = compile_test_signature([kwargs[key] for key in sorted_kwargs])
        model = self.models[test_sign]
        minerr = 1e6
        for perm in itertools.permutations(range(self.n_hidden)):
            minerr = min(minerr, np.max(np.abs(model.transition_model.stationary_distribution[np.array(perm)] -
                                                self.hidden_stationary_distribution)))
        np.testing.assert_almost_equal(minerr, 0, decimal=2)

    def _test_hidden_transition_matrix(self, **kwargs):
        test_sign = compile_test_signature([kwargs[key] for key in sorted_kwargs])
        model = self.models[test_sign]
        minerr = 1e6
        for perm in permutation_matrices(self.n_hidden):
            minerr = min(minerr, np.max(np.abs(perm.T @ model.transition_model.transition_matrix @ perm -
                                                self.T_hidden)))
        np.testing.assert_almost_equal(minerr, 0, decimal=2)

    def _test_hidden_path(self, **kwargs):
        test_sign = compile_test_signature([kwargs[key] for key in sorted_kwargs])
        model = self.models[test_sign]
        minerr = 1e6
        for perm in itertools.permutations(range(self.n_hidden)):
            viterbi_est = model.compute_viterbi_paths([self.observable_state_traj])[0]
            minerr = min(minerr, (np.array(perm)[viterbi_est] != self.hidden_state_traj).sum()
                         / self.n_steps)

        np.testing.assert_almost_equal(minerr, 0, decimal=1)


if __name__ == "__main__":
    unittest.main()
