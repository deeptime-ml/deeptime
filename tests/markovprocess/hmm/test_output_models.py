import unittest

import numpy as np
from sktime.markovprocess.hmm.output_model import DiscreteOutputModel
from sktime.markovprocess.hmm.output_model import GaussianOutputModel


class TestDiscrete(unittest.TestCase):

    def test_basic_properties(self):
        # two obs states, four hidden states
        output_probs = np.array([
            [0.5, 0.3, 0.2, 0.0],
            [0.5, 0.3, 0.2, 0.0]
        ])
        m = DiscreteOutputModel(output_probs, ignore_outliers=False)
        np.testing.assert_equal(m.ignore_outliers, False)
        m.ignore_outliers = True
        np.testing.assert_equal(m.ignore_outliers, True)
        np.testing.assert_allclose(m.prior, 0.)
        np.testing.assert_equal(m.n_hidden_states, 2)
        np.testing.assert_equal(m.n_observable_states, 4)
        np.testing.assert_equal(m.output_probabilities, output_probs)

    def test_invalid_ctor_args(self):
        with self.assertRaises(ValueError):
            # not row stochastic
            DiscreteOutputModel(np.array([
                [0.5, 0.3, 0.2, 0.1],
                [0.5, 0.3, 0.2, 0.0]
            ]))
        # no-op: this does not raise
        DiscreteOutputModel(np.array([
            [0., 1., 0., 0.],
            [1., 0., 0., 0.]
        ]), prior=np.random.normal(size=(2, 4)).astype(np.float64))
        with self.assertRaises(ValueError):
            # prior has wrong shape, raise
            DiscreteOutputModel(np.array([
                [0., 1., 0., 0.],
                [1., 0., 0., 0.]
            ]), prior=np.random.normal(size=(2, 5)).astype(np.float64))

    def test_observation_trajectory(self):
        output_probabilities = np.array([
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.1, 0.3, 0.1, 0.3, 0.2],
            [0.1, 0.1, 0.1, 0.1, 0.6],
            [0.6, 0.1, 0.1, 0.1, 0.1],
        ])
        m = DiscreteOutputModel(output_probabilities)
        np.testing.assert_equal(m.n_hidden_states, 4)
        np.testing.assert_equal(m.n_observable_states, 5)
        traj = m.generate_observation_trajectory(np.array([1] * 2000000))
        bc = np.bincount(traj.astype(np.int32), minlength=m.n_observable_states).astype(np.float32)
        bc /= np.sum(bc)
        np.testing.assert_array_almost_equal(bc, np.array([0.1, 0.3, 0.1, 0.3, 0.2]), decimal=2)

    def test_output_probability_trajectory(self):
        output_probabilities = np.array([
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.1, 0.3, 0.1, 0.3, 0.2],
            [0.1, 0.1, 0.1, 0.1, 0.6],
            [0.6, 0.1, 0.1, 0.1, 0.1],
        ])


        m = DiscreteOutputModel(output_probabilities)
        obs_traj = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        prob_traj = m.to_state_probability_trajectory(obs_traj)
        for state, prob in zip(obs_traj, prob_traj):
            np.testing.assert_equal(prob, output_probabilities[:, state])

    def test_sample(self):
        output_probabilities = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.9, 0.0]
        ])
        m = DiscreteOutputModel(output_probabilities)
        obs_per_state = [
            np.array([0] * 50000 + [1] * 50000),  # state 0
            np.array([1] * 30000 + [2] * 70000)   # state 1
        ]
        m.sample(obs_per_state)
        np.testing.assert_array_almost_equal(m.output_probabilities, np.array([[.5, .5, 0.], [0, .3, .7]]), decimal=2)

    @unittest.skip("reference to bhmm")
    def test_observation_trajectory2(self):
        from bhmm.output_models import DiscreteOutputModel as DOM
        m = DOM(np.array([
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.1, 0.3, 0.1, 0.3, 0.2],
            [0.1, 0.1, 0.1, 0.1, 0.6],
            [0.6, 0.1, 0.1, 0.1, 0.1],
        ]))
        np.testing.assert_equal(m.nstates, 4)
        np.testing.assert_equal(m.nsymbols, 5)
        traj = m.generate_observation_trajectory(np.array([0] * 1000000))
        bc = np.bincount(traj.astype(np.int32), minlength=m.nsymbols).astype(np.float32)
        bc /= np.sum(bc)
        print(bc)

class TestGaussian(unittest.TestCase):

    def test_observation_trajectory(self):
        m = GaussianOutputModel(3, means=np.array([-1., 0., 1.]), sigmas=np.array([.5, .2, .1]))
        np.testing.assert_equal(m.n_hidden_states, 3)
        np.testing.assert_equal(m.ignore_outliers, True)
        for state in range(3):
            traj = m.generate_observation_trajectory(np.array([state]*100000))
            np.testing.assert_almost_equal(np.mean(traj), m.means[state], decimal=3)
            np.testing.assert_almost_equal(np.sqrt(np.var(traj)), m.sigmas[state], decimal=3)

    def test_output_probability_trajectory(self):
        m = GaussianOutputModel(3, means=np.array([-1., 0., 1.]), sigmas=np.array([.5, .2, .1]))
        stateprobs = m.to_state_probability_trajectory(np.array([-1., 0., 1.]))
        print(stateprobs)

if __name__ == '__main__':
    unittest.main()
