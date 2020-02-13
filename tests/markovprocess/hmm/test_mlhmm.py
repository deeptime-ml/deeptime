import unittest

import numpy as np
import sktime.markovprocess.hmm._hmm_bindings as _bindings

from sktime.data.double_well import DoubleWellDiscrete
from sktime.markovprocess.hmm import MaximumLikelihoodHMSM
from sktime.markovprocess.hmm.hmm import viterbi
from sktime.markovprocess.hmm.maximum_likelihood_hmm import initial_guess_discrete_from_data
from sktime.markovprocess.hmm.output_model import DiscreteOutputModel
from tests.util import assert_array_not_equal


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

    @classmethod
    def setUpClass(cls) -> None:
        dtraj = DoubleWellDiscrete().dtraj
        initial_hmm_10 = initial_guess_discrete_from_data(dtraj, n_hidden_states=2, lagtime=10)
        cls.hmm_lag10 = MaximumLikelihoodHMSM(initial_hmm_10, lagtime=10).fit(dtraj).fetch_model()
        initial_hmm_1 = initial_guess_discrete_from_data(dtraj, n_hidden_states=2, lagtime=1)
        cls.hmm_lag1 = MaximumLikelihoodHMSM(initial_hmm_1).fit(dtraj).fetch_model()

    def test_output_model(self):
        assert isinstance(self.hmm_lag1.output_model, DiscreteOutputModel)
        assert isinstance(self.hmm_lag10.output_model, DiscreteOutputModel)

    def test_reversible(self):
        np.testing.assert_(self.hmm_lag1.transition_model.reversible)
        np.testing.assert_(self.hmm_lag10.transition_model.reversible)

    def test_stationary(self):
        np.testing.assert_(self.hmm_lag1.transition_model.stationary)
        np.testing.assert_(self.hmm_lag10.transition_model.stationary)

    def test_lag(self):
        assert self.hmm_lag1.transition_model.lagtime == 1
        assert self.hmm_lag10.transition_model.lagtime == 10

    def test_nstates(self):
        np.testing.assert_equal(self.hmm_lag1.n_hidden_states, 2)
        np.testing.assert_equal(self.hmm_lag1.transition_model.n_states, 2)
        np.testing.assert_equal(self.hmm_lag10.n_hidden_states, 2)
        np.testing.assert_equal(self.hmm_lag10.transition_model.n_states, 2)

    def test_transition_matrix(self):
        import msmtools.analysis as msmana
        for P in [self.hmm_lag1.transition_model.transition_matrix, self.hmm_lag10.transition_model.transition_matrix]:
            np.testing.assert_(msmana.is_transition_matrix(P))
            np.testing.assert_(msmana.is_reversible(P))

    def test_eigenvalues(self):
        for ev in [self.hmm_lag1.transition_model.eigenvalues(2), self.hmm_lag10.transition_model.eigenvalues(2)]:
            np.testing.assert_equal(len(ev), 2)
            np.testing.assert_allclose(ev[0], 1)
            np.testing.assert_(ev[1] < 1.)

    def test_eigenvectors_left(self):
        for evec in [self.hmm_lag1.transition_model.eigenvectors_left(2),
                     self.hmm_lag10.transition_model.eigenvectors_left(2)]:
            np.testing.assert_equal(evec.shape, (2, 2))
            np.testing.assert_equal(np.sign(evec[0, 0]), np.sign(evec[0, 1]))
            assert_array_not_equal(np.sign(evec[1, 0]), np.sign(evec[1, 1]))

    def test_eigenvectors_right(self):
        for evec in [self.hmm_lag1.transition_model.eigenvectors_right(),
                     self.hmm_lag10.transition_model.eigenvectors_right()]:
            np.testing.assert_equal(evec.shape, (2, 2))
            np.testing.assert_allclose(evec[0, 0], evec[1, 0])
            assert_array_not_equal(np.sign(evec[0, 1]), np.sign(evec[1, 1]))

    def test_initial_distribution(self):
        for mu in [self.hmm_lag1.initial_distribution, self.hmm_lag10.initial_distribution]:
            # normalization
            assert np.isclose(mu.sum(), 1.0)
            # should be on one side
            assert np.isclose(mu[0], 1.0) or np.isclose(mu[0], 0.0)

    def test_stationary_distribution(self):
        for mu in [self.hmm_lag1.transition_model.stationary_distribution,
                   self.hmm_lag10.transition_model.stationary_distribution]:
            # normalization
            assert np.isclose(mu.sum(), 1.0)
            # positivity
            assert np.all(mu > 0.0)
            # this data: approximately equal probability
            assert np.max(np.abs(mu[0]-mu[1])) < 0.05

    def test_lifetimes(self):
        for l in [self.hmm_lag1.transition_model.lifetimes, self.hmm_lag10.lifetimes]:
            assert len(l) == 2
            assert np.all(l > 0.0)
        # this data: lifetimes about 680
        assert np.max(np.abs(self.hmm_lag10.lifetimes - 680)) < 20.0

    def test_timescales(self):
        for l in [self.hmm_lag1.transition_model.timescales(2), self.hmm_lag10.transition_model.timescales(2)]:
            assert len(l) == 1
            assert np.all(l > 0.0)
        # this data: lifetimes about 680
        print(self.hmm_lag10.transition_model.timescales())
        assert np.abs(self.hmm_lag10.transition_model.timescales(2)[0] - 340) < 20.0


if __name__ == '__main__':
    unittest.main()
