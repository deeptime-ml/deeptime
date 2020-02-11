import unittest

import numpy as np
import sktime.markovprocess.hmm._hmm_bindings as _bindings

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
        beta_out = np.zeros((6, 2))
        _bindings.util.backward(self.transition_probabilities, self.state_probabilities, beta_out=beta_out)

        ref_beta = np.array([
            [1]
        ])
        print(beta_out)
        import bhmm.hidden as refimpl
        b = refimpl.backward(self.transition_probabilities, self.state_probabilities)
        print(b)


class TestMLHMM(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()