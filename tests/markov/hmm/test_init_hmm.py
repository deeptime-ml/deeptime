import unittest

import deeptime.markov.tools.analysis as msmana
import numpy as np

import deeptime.markov.hmm.init as init
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MarkovStateModel


class TestInitHMMDiscrete(unittest.TestCase):

    def test_discrete_2_2(self):
        # 2x2 transition matrix
        P = np.array([[0.99, 0.01], [0.01, 0.99]])
        # generate realization
        T = 10000
        dtrajs = [MarkovStateModel(P).simulate(T)]
        # estimate initial HMM with 2 states - should be identical to P
        init_hmm = init.discrete.metastable_from_data(dtrajs, n_hidden_states=2, lagtime=1)
        # test
        A = init_hmm.transition_model.transition_matrix
        B = init_hmm.output_probabilities
        # Test stochasticity
        np.testing.assert_(msmana.is_transition_matrix(A))
        np.testing.assert_allclose(B.sum(axis=1), np.ones(B.shape[0]))
        # A should be close to P
        if B[0, 0] < B[1, 0]:
            B = B[np.array([1, 0]), :]
        np.testing.assert_array_almost_equal(A, P, decimal=2)
        np.testing.assert_array_almost_equal(B, np.eye(2), decimal=2)

    def test_discrete_4_2(self):
        # 4x4 transition matrix
        n_states = 2
        P = np.array([[0.90, 0.10, 0.00, 0.00],
                      [0.10, 0.89, 0.01, 0.00],
                      [0.00, 0.01, 0.89, 0.10],
                      [0.00, 0.00, 0.10, 0.90]])
        # generate realization
        T = 50000
        dtrajs = [MarkovStateModel(P).simulate(T)]
        # estimate initial HMM with 2 states - should be identical to P
        hmm = init.discrete.metastable_from_data(dtrajs, n_states, lagtime=1, regularize=False)
        # Test if model fit is close to reference. Note that we do not have an exact reference, so we cannot set the
        # tolerance in a rigorous way to test statistical significance. These are just sanity checks.
        Tij = hmm.transition_model.transition_matrix
        B = hmm.output_probabilities
        # Test stochasticity
        np.testing.assert_(msmana.is_transition_matrix(Tij))
        np.testing.assert_allclose(B.sum(axis=1), np.ones(B.shape[0]))
        Tij_ref = np.array([[0.99, 0.01],
                            [0.01, 0.99]])
        Bref = np.array([[0.5, 0.5, 0.0, 0.0],
                         [0.0, 0.0, 0.5, 0.5]])
        np.testing.assert_array_almost_equal(Tij, Tij_ref, decimal=2)
        if np.max(B - Bref) < .05:
            np.testing.assert_allclose(B, Bref, atol=0.06)
        else:
            np.testing.assert_allclose(B[[1, 0]], Bref, atol=0.06)

    def test_discrete_6_3(self):
        # 4x4 transition matrix
        n_states = 3
        P = np.array([[0.90, 0.10, 0.00, 0.00, 0.00, 0.00],
                      [0.20, 0.79, 0.01, 0.00, 0.00, 0.00],
                      [0.00, 0.01, 0.84, 0.15, 0.00, 0.00],
                      [0.00, 0.00, 0.05, 0.94, 0.01, 0.00],
                      [0.00, 0.00, 0.00, 0.02, 0.78, 0.20],
                      [0.00, 0.00, 0.00, 0.00, 0.10, 0.90]])
        # generate realization
        T = 10000
        dtrajs = [MarkovStateModel(P).simulate(T)]
        # estimate initial HMM with 2 states - should be identical to P
        hmm = init.discrete.metastable_from_data(dtrajs, n_states, 1)
        # Test stochasticity and reversibility
        Tij = hmm.transition_model.transition_matrix
        B = hmm.output_probabilities
        np.testing.assert_(msmana.is_transition_matrix(Tij))
        np.testing.assert_(msmana.is_reversible(Tij))
        np.testing.assert_allclose(B.sum(axis=1), np.ones(B.shape[0]))

    # ------------------------------------------------------------------------------------------------------
    # Test correct initialization of pathological cases - single states, partial connectivity, etc.
    # ------------------------------------------------------------------------------------------------------

    def test_1state_1obs(self):
        dtraj = np.array([0, 0, 0, 0, 0])
        Aref = np.array([[1.0]])
        Bref = np.array([[1.0]])
        for rev in [True, False]:  # reversibiliy doesn't matter in this example
            hmm = init.discrete.metastable_from_data(dtraj, 1, 1, reversible=rev)
            np.testing.assert_allclose(hmm.transition_model.transition_matrix, Aref)
            np.testing.assert_allclose(hmm.output_probabilities, Bref)

    def test_2state_2obs_deadend(self):
        dtraj = np.array([0, 0, 0, 0, 1])
        Aref = np.array([[1.0]])
        for rev in [True, False]:  # reversibiliy doesn't matter in this example
            hmm = init.discrete.metastable_from_data(dtraj, 1, 1, reversible=rev)
            np.testing.assert_allclose(hmm.transition_model.transition_matrix, Aref)
            # output must be 1 x 2, and no zeros
            B = hmm.output_probabilities
            np.testing.assert_equal(B.shape, (1, 2))
            np.testing.assert_array_less(0, B)

    def test_2state_2obs_Pgiven(self):
        obs = np.array([0, 0, 1, 1, 0])
        Aref = np.array([[1.0]])
        for rev in [True, False]:  # reversibiliy doesn't matter in this example
            hmm = init.discrete.metastable_from_data(obs, n_hidden_states=1, lagtime=1, reversible=rev)
            np.testing.assert_(msmana.is_transition_matrix(hmm.transition_model.transition_matrix))
            np.testing.assert_allclose(hmm.transition_model.transition_matrix, Aref)
            # output must be 1 x 2, and no zeros
            np.testing.assert_equal(hmm.output_probabilities.shape, (1, 2))
            np.testing.assert_(np.all(hmm.output_probabilities > 0))

    def test_2state_2obs_unidirectional(self):
        dtraj = np.array([0, 0, 0, 0, 1])
        Aref_naked = np.array([[0.75, 0.25],
                               [0, 1]])
        Bref_naked = np.array([[1., 0.],
                               [0., 1.]])
        perm = [1, 0]  # permutation
        for rev in [True, False]:  # reversibiliy doesn't matter in this example
            hmm = init.discrete.metastable_from_data(dtraj, n_hidden_states=2, lagtime=1, reversible=rev,
                                                     regularize=False, mode='all')
            assert np.allclose(hmm.transition_model.transition_matrix, Aref_naked) \
                   or np.allclose(hmm.transition_model.transition_matrix,
                                  Aref_naked[np.ix_(perm, perm)])  # test permutation
            assert np.allclose(hmm.output_probabilities, Bref_naked) \
                   or np.allclose(hmm.output_probabilities, Bref_naked[perm])  # test permutation

    def test_3state_fail(self):
        dtraj = np.array([0, 1, 0, 0, 1, 1])
        # this example doesn't admit more than 2 metastable states. Raise.
        with self.assertRaises(ValueError):
            init.discrete.metastable_from_data(dtraj, 3, 1, reversible=False)

    def test_3state_prev(self):
        dtraj = np.array([0, 1, 2, 0, 3, 4])
        import deeptime.markov.tools.estimation as msmest
        for rev in [True, False]:
            hmm = init.discrete.metastable_from_data(dtraj, n_hidden_states=3, lagtime=1, reversible=rev)
            assert msmana.is_transition_matrix(hmm.transition_model.transition_matrix)
            if rev:
                assert msmana.is_reversible(hmm.transition_model.transition_matrix)
            assert np.allclose(hmm.output_probabilities.sum(axis=1), 1)

        for rev in [True, False]:
            C = TransitionCountEstimator(lagtime=1, count_mode="sliding").fit(dtraj).fetch_model().count_matrix
            C += msmest.prior_neighbor(C, 0.001)
            hmm = init.discrete.metastable_from_data(dtraj, n_hidden_states=3, lagtime=1, reversible=rev)
            np.testing.assert_(msmana.is_transition_matrix(hmm.transition_model.transition_matrix))
            if rev:
                np.testing.assert_(msmana.is_reversible(hmm.transition_model.transition_matrix))
            np.testing.assert_allclose(hmm.output_probabilities.sum(axis=1), 1.)

    def test_state_splitting(self):
        dtraj = np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                          0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 1, 2, 2, 2, 2, 2, 2])
        hmm0 = init.discrete.metastable_from_data(dtraj, n_hidden_states=3, lagtime=1, separate_symbols=np.array([0]))
        piref = np.array([0.35801876, 0.55535398, 0.08662726])
        Aref = np.array([[0.76462978, 0.10261978, 0.13275044],
                         [0.06615566, 0.89464821, 0.03919614],
                         [0.54863966, 0.25128039, 0.20007995]])
        Bref = np.array([[0, 1, 0],
                         [0, 0, 1],
                         [1, 0, 0]])
        np.testing.assert_array_almost_equal(hmm0.transition_model.transition_matrix, Aref, decimal=2)
        np.testing.assert_array_almost_equal(hmm0.output_probabilities, Bref, decimal=2)
        np.testing.assert_array_almost_equal(hmm0.initial_distribution, piref, decimal=2)

    def test_state_splitting_empty(self):
        dtraj = np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                          0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 1, 2, 2, 2, 2, 2, 2])
        # create empty labels
        dtraj += 2
        # include an empty label in separate
        hmm0 = init.discrete.metastable_from_data(dtraj, 3, lagtime=1, separate_symbols=np.array([1, 2]),
                                                  mode='populous')
        piref = np.array([0.35801876, 0.55535398, 0.08662726])
        Aref = np.array([[0.76462978, 0.10261978, 0.13275044],
                         [0.06615566, 0.89464821, 0.03919614],
                         [0.54863966, 0.25128039, 0.20007995]])
        Bref = np.array([[0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 1],
                         [0, 0, 1, 0, 0]])
        np.testing.assert_(np.max(np.abs(hmm0.output_probabilities - Bref)) < 0.01)
        np.testing.assert_array_almost_equal(hmm0.transition_model.transition_matrix, Aref, decimal=2)
        np.testing.assert_array_almost_equal(hmm0.initial_distribution, piref, decimal=2)

    def test_state_splitting_fail(self):
        dtraj = np.array([0, 0, 1, 1])
        with self.assertRaises(ValueError):
            init.discrete.metastable_from_data(dtraj, 2, 1, separate_symbols=np.array([0, 2]))
