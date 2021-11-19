import unittest

import numpy as np
import deeptime
from deeptime.markov.hmm import DiscreteOutputModel, GaussianOutputModel


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
            np.array([1] * 30000 + [2] * 70000)  # state 1
        ]
        m.sample(obs_per_state)
        # the output probabilities of the unpopulated states are left as-is (can't sample), hence we compare against
        # [[.5, .5, .1], [.1, .3, .7]] instead of [[.5, .5, .0], [.0, .3, .7]]
        np.testing.assert_array_almost_equal(m.output_probabilities, np.array([[.5, .5, .1], [.1, .3, .7]]), decimal=2)

    def test_fit(self):
        output_probabilities = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.9, 0.0]
        ])
        n_trajs = 100
        m = DiscreteOutputModel(output_probabilities)
        obs = [np.random.randint(0, 3, size=10000 + np.random.randint(-3, 3)) for _ in range(n_trajs)]
        weights = [np.random.dirichlet([2, 3, 4], size=o.size) for o in obs]
        m.fit(obs, weights)
        np.testing.assert_allclose(m.output_probabilities, 1. / 3, atol=.01)

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
            traj = m.generate_observation_trajectory(np.array([state] * 2000000))
            np.testing.assert_almost_equal(np.mean(traj), m.means[state], decimal=2)
            np.testing.assert_almost_equal(np.sqrt(np.var(traj)), m.sigmas[state], decimal=3)

    def test_output_probability_trajectory(self):
        m = GaussianOutputModel(3, means=np.array([-1., 0., 1.]), sigmas=np.array([.5, .2, .1]))
        m.ignore_outliers = True
        for state in range(m.n_hidden_states):
            mean = m.means[state]
            stateprobs = m.to_state_probability_trajectory(np.random.normal(loc=mean, scale=1e-4, size=(1000,)))
            np.testing.assert_equal(np.argmax(stateprobs, axis=-1), state)

    def test_sample(self):
        m = GaussianOutputModel(3)
        means = np.array([-1., 1., 3.])
        sigmas = np.array([.1, .2, .3])
        obs_per_state = [
            np.random.normal(means[0], sigmas[0], size=(100000,)),
            np.random.normal(means[1], sigmas[1], size=(100000,)),
            np.random.normal(means[2], sigmas[2], size=(100000,)),
        ]
        m.sample(obs_per_state)
        np.testing.assert_array_almost_equal(m.means, means, decimal=2)
        np.testing.assert_array_almost_equal(m.sigmas, sigmas, decimal=2)

    def test_fit(self):
        expected_means = np.array([-55., 0., 7.])
        expected_stds = np.array([3., .5, 1.])

        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=3, means_init=expected_means)
        gmm.means_ = expected_means[..., None]
        gmm.covariances_ = np.array(expected_stds[..., None, None])
        gmm.weights_ = np.array([1/3, 1/3, 1/3])

        obs = gmm.sample(100000 + np.random.randint(-3, 3))[0].squeeze()

        init = deeptime.markov.hmm.init.gaussian.from_data(obs, n_hidden_states=3, reversible=True)
        hmm_est = deeptime.markov.hmm.MaximumLikelihoodHMM(init, lagtime=1)
        hmm = hmm_est.fit(obs).fetch_model()

        np.testing.assert_array_almost_equal(hmm.transition_model.transition_matrix, np.eye(3), decimal=3)
        m = hmm.output_model
        for mean, sigma in zip(m.means, m.sigmas):
            # find the mean closest to this one (order might have changed)
            mean_ix = np.argmin(np.abs(expected_means-mean))
            np.testing.assert_almost_equal(mean, expected_means[mean_ix], decimal=1)
            np.testing.assert_almost_equal(sigma*sigma, expected_stds[mean_ix], decimal=1)
