
# This file is part of BHMM (Bayesian Hidden Markov Models).
#
# Copyright (c) 2016 Frank Noe (Freie Universitaet Berlin)
# and John D. Chodera (Memorial Sloan-Kettering Cancer Center, New York)
#
# BHMM is free software: you can redistribute it and/or modify
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

import numpy as np
import unittest
from bhmm import init_discrete_hmm
from bhmm.init.discrete import init_discrete_hmm_spectral
import msmtools.estimation as msmest


class TestHMM(unittest.TestCase):

    # ------------------------------------------------------------------------------------------------------
    # Test correct initialization of metastable trajectories
    # ------------------------------------------------------------------------------------------------------

    def test_discrete_2_2(self):
        # 2x2 transition matrix
        P = np.array([[0.99, 0.01], [0.01, 0.99]])
        # generate realization
        import msmtools.generation as msmgen
        T = 10000
        dtrajs = [msmgen.generate_traj(P, T)]
        C = msmest.count_matrix(dtrajs, 1).toarray()
        # estimate initial HMM with 2 states - should be identical to P
        hmm = init_discrete_hmm(dtrajs, 2)
        # test
        A = hmm.transition_matrix
        B = hmm.output_model.output_probabilities
        # Test stochasticity
        import msmtools.analysis as msmana
        msmana.is_transition_matrix(A)
        np.allclose(B.sum(axis=1), np.ones(B.shape[0]))
        # A should be close to P
        if B[0, 0] < B[1, 0]:
            B = B[np.array([1, 0]), :]
        assert(np.max(A-P) < 0.01)
        assert(np.max(B-np.eye(2)) < 0.01)

    def test_discrete_4_2(self):
        # 4x4 transition matrix
        nstates = 2
        P = np.array([[0.90, 0.10, 0.00, 0.00],
                      [0.10, 0.89, 0.01, 0.00],
                      [0.00, 0.01, 0.89, 0.10],
                      [0.00, 0.00, 0.10, 0.90]])
        # generate realization
        import msmtools.generation as msmgen
        T = 10000
        dtrajs = [msmgen.generate_traj(P, T)]
        C = msmest.count_matrix(dtrajs, 1).toarray()
        # estimate initial HMM with 2 states - should be identical to P
        hmm = init_discrete_hmm(dtrajs, nstates)
        # Test if model fit is close to reference. Note that we do not have an exact reference, so we cannot set the
        # tolerance in a rigorous way to test statistical significance. These are just sanity checks.
        Tij = hmm.transition_matrix
        B = hmm.output_model.output_probabilities
        # Test stochasticity
        import msmtools.analysis as msmana
        msmana.is_transition_matrix(Tij)
        np.allclose(B.sum(axis=1), np.ones(B.shape[0]))
        # if (B[0,0]<B[1,0]):
        #     B = B[np.array([1,0]),:]
        Tij_ref = np.array([[0.99, 0.01],
                            [0.01, 0.99]])
        Bref = np.array([[0.5, 0.5, 0.0, 0.0],
                         [0.0, 0.0, 0.5, 0.5]])
        assert(np.max(Tij-Tij_ref) < 0.01)
        assert(np.max(B-Bref) < 0.05 or np.max(B[[1, 0]]-Bref) < 0.05)

    def test_discrete_6_3(self):
        # 4x4 transition matrix
        nstates = 3
        P = np.array([[0.90, 0.10, 0.00, 0.00, 0.00, 0.00],
                      [0.20, 0.79, 0.01, 0.00, 0.00, 0.00],
                      [0.00, 0.01, 0.84, 0.15, 0.00, 0.00],
                      [0.00, 0.00, 0.05, 0.94, 0.01, 0.00],
                      [0.00, 0.00, 0.00, 0.02, 0.78, 0.20],
                      [0.00, 0.00, 0.00, 0.00, 0.10, 0.90]])
        # generate realization
        import msmtools.generation as msmgen
        T = 10000
        dtrajs = [msmgen.generate_traj(P, T)]
        C = msmest.count_matrix(dtrajs, 1).toarray()
        # estimate initial HMM with 2 states - should be identical to P
        hmm = init_discrete_hmm(dtrajs, nstates)
        # Test stochasticity and reversibility
        Tij = hmm.transition_matrix
        B = hmm.output_model.output_probabilities
        import msmtools.analysis as msmana
        msmana.is_transition_matrix(Tij)
        msmana.is_reversible(Tij)
        np.allclose(B.sum(axis=1), np.ones(B.shape[0]))

    # ------------------------------------------------------------------------------------------------------
    # Test correct initialization of pathological cases - single states, partial connectivity, etc.
    # ------------------------------------------------------------------------------------------------------

    def test_1state_1obs(self):
        dtraj = np.array([0, 0, 0, 0, 0])
        C = msmest.count_matrix(dtraj, 1).toarray()
        Aref = np.array([[1.0]])
        Bref = np.array([[1.0]])
        for rev in [True, False]:  # reversibiliy doesn't matter in this example
            hmm = init_discrete_hmm(dtraj, 1, reversible=rev)
            assert(np.allclose(hmm.transition_matrix, Aref))
            assert(np.allclose(hmm.output_model.output_probabilities, Bref))

    def test_2state_2obs_deadend(self):
        dtraj = np.array([0, 0, 0, 0, 1])
        C = msmest.count_matrix(dtraj, 1).toarray()
        Aref = np.array([[1.0]])
        for rev in [True, False]:  # reversibiliy doesn't matter in this example
            hmm = init_discrete_hmm(dtraj, 1, reversible=rev)
            assert(np.allclose(hmm.transition_matrix, Aref))
            # output must be 1 x 2, and no zeros
            B = hmm.output_model.output_probabilities
            assert(np.array_equal(B.shape, np.array([1, 2])))
            assert(np.all(B > 0.0))

    def test_2state_2obs_Pgiven(self):
        obs = np.array([0, 0, 1, 1, 0])
        C = msmest.count_matrix(obs, 1).toarray()
        Aref = np.array([[1.0]])
        for rev in [True, False]:  # reversibiliy doesn't matter in this example
            P = msmest.transition_matrix(C, reversible=rev)
            p0, P0, B0 = init_discrete_hmm_spectral(C, 1, reversible=rev, P=P)
            assert(np.allclose(P0, Aref))
            # output must be 1 x 2, and no zeros
            assert(np.array_equal(B0.shape, np.array([1, 2])))
            assert(np.all(B0 > 0.0))

    def test_2state_2obs_unidirectional(self):
        dtraj = np.array([0, 0, 0, 0, 1])
        C = msmest.count_matrix(dtraj, 1).toarray()
        Aref_naked = np.array([[0.75, 0.25],
                               [0   , 1   ]])
        Bref_naked = np.array([[1.,  0.],
                               [0.,  1.]])
        perm = [1, 0]  # permutation
        for rev in [True, False]:  # reversibiliy doesn't matter in this example
            hmm = init_discrete_hmm(dtraj, 2, reversible=rev, method='spectral', regularize=False)
            assert np.allclose(hmm.transition_matrix, Aref_naked) \
                   or np.allclose(hmm.transition_matrix, Aref_naked[np.ix_(perm, perm)])  # test permutation
            assert np.allclose(hmm.output_model.output_probabilities, Bref_naked) \
                   or np.allclose(hmm.output_model.output_probabilities, Bref_naked[perm])  # test permutation

    def test_3state_fail(self):
        dtraj = np.array([0, 1, 0, 0, 1, 1])
        C = msmest.count_matrix(dtraj, 1).toarray()
        # this example doesn't admit more than 2 metastable states. Raise.
        with self.assertRaises(NotImplementedError):
            init_discrete_hmm(dtraj, 3, reversible=False)

    def test_3state_prev(self):
        import msmtools.analysis as msmana
        dtraj = np.array([0, 1, 2, 0, 3, 4])
        C = msmest.count_matrix(dtraj, 1).toarray()
        for rev in [True, False]:
            hmm = init_discrete_hmm(dtraj, 3, reversible=rev)
            assert msmana.is_transition_matrix(hmm.transition_matrix)
            if rev:
                assert msmana.is_reversible(hmm.transition_matrix)
            assert np.allclose(hmm.output_model.output_probabilities.sum(axis=1), 1)

    def test_state_splitting(self):
        dtraj = np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                          0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 1, 2, 2, 2, 2, 2, 2])
        C = msmest.count_matrix(dtraj, 1).toarray()
        hmm0 = init_discrete_hmm(dtraj, 3, separate=[0])
        piref = np.array([0.35801876, 0.55535398, 0.08662726])
        Aref = np.array([[0.76462978, 0.10261978, 0.13275044],
                         [0.06615566, 0.89464821, 0.03919614],
                         [0.54863966, 0.25128039, 0.20007995]])
        Bref = np.array([[0, 1, 0],
                         [0, 0, 1],
                         [1, 0, 0]])
        assert np.allclose(hmm0.initial_distribution, piref, atol=1e-5)
        assert np.allclose(hmm0.transition_matrix, Aref, atol=1e-5)
        assert np.max(np.abs(hmm0.output_model.output_probabilities - Bref)) < 0.01

    def test_state_splitting_empty(self):
        dtraj = np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                          0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 1, 2, 2, 2, 2, 2, 2])
        dtraj += 2  # create empty labels
        C = msmest.count_matrix(dtraj, 1).toarray()
        hmm0 = init_discrete_hmm(dtraj, 3, separate=[1, 2])  # include an empty label in separate
        piref = np.array([0.35801876, 0.55535398, 0.08662726])
        Aref = np.array([[0.76462978, 0.10261978, 0.13275044],
                         [0.06615566, 0.89464821, 0.03919614],
                         [0.54863966, 0.25128039, 0.20007995]])
        Bref = np.array([[0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 1],
                         [0, 0, 1, 0, 0]])
        assert np.allclose(hmm0.initial_distribution, piref, atol=1e-5)
        assert np.allclose(hmm0.transition_matrix, Aref, atol=1e-5)
        assert np.max(np.abs(hmm0.output_model.output_probabilities - Bref)) < 0.01

    def test_state_splitting_fail(self):
        dtraj = np.array([0, 0, 1, 1])
        with self.assertRaises(ValueError):
            init_discrete_hmm(dtraj, 2, separate=[0, 2])

if __name__ == "__main__":
    unittest.main()
