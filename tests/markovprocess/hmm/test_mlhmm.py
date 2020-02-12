import unittest

import numpy as np
import sktime.markovprocess.hmm._hmm_bindings as _bindings

from sktime.markovprocess.hmm.hmm import viterbi


class TestAlgorithmsAgainstReference(unittest.TestCase):
    """ Tests against example from Wikipedia: http://en.wikipedia.org/wiki/Forward-backward_algorithm#Example """

    def setUp(self) -> None:
        # weather transition probabilities: 1=rain and 2=no rain
        self.transition_probabilities = np.array([
            [0.7, 0.3],
            [0.3, 0.7]
        ])
        # discrete traj: 1 = umbrella, 2 = no umbrella
        self.dtraj = np.array([0, 0, 1, 0, 0])
        # conditional probabilities
        self.conditional_probabilities = np.array([
            [0.9, 0.1], [0.2, 0.8]
        ])
        self.state_probabilities = np.array([
            [0.9, 0.2],
            [0.9, 0.2],
            [0.1, 0.8],
            [0.9, 0.2],
            [0.9, 0.2]
        ])

    def test_forward(self):
        alpha_out = np.zeros_like(self.state_probabilities)
        logprob = _bindings.util.forward(self.transition_probabilities, self.state_probabilities, np.array([0.5, 0.5]),
                                         alpha_out=alpha_out)
        ref_logprob = -3.3725
        ref_alpha = np.array([
            [0.8182, 0.1818],
            [0.8834, 0.1166],
            [0.1907, 0.8093],
            [0.7308, 0.2692],
            [0.8673, 0.1327]
        ])
        np.testing.assert_array_almost_equal(logprob, ref_logprob, decimal=4)
        np.testing.assert_array_almost_equal(alpha_out, ref_alpha, decimal=4)

    def test_backward(self):
        beta_out = np.zeros_like(self.state_probabilities)
        _bindings.util.backward(self.transition_probabilities, self.state_probabilities, beta_out=beta_out)

        ref_beta = np.array([
            [0.5923, 0.4077],
            [0.3763, 0.6237],
            [0.6533, 0.3467],
            [0.6273, 0.3727],
            [.5, .5]
        ])
        np.testing.assert_array_almost_equal(beta_out, ref_beta, decimal=4)

    def test_state_probabilities(self):
        ref_alpha = np.array([
            [0.8182, 0.1818],
            [0.8834, 0.1166],
            [0.1907, 0.8093],
            [0.7308, 0.2692],
            [0.8673, 0.1327]
        ])
        ref_beta = np.array([
            [0.5923, 0.4077],
            [0.3763, 0.6237],
            [0.6533, 0.3467],
            [0.6273, 0.3727],
            [.5, .5]
        ])
        gamma = np.zeros((len(self.dtraj), self.transition_probabilities.shape[0]))
        _bindings.util.state_probabilities(ref_alpha, ref_beta, gamma_out=gamma)

        gamma_ref = np.array([
            [0.8673, 0.1327],
            [0.8204, 0.1796],
            [0.3075, 0.6925],
            [0.8204, 0.1796],
            [0.8673, 0.1327]
        ])
        np.testing.assert_array_almost_equal(gamma, gamma_ref, decimal=4)

    def test_viterbi(self):
        path = viterbi(self.transition_probabilities, self.state_probabilities, np.array([0.5, 0.5]))
        np.testing.assert_array_equal(path, self.dtraj)


class TestMLHMM(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
