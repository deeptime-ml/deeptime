
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

import unittest
import numpy as np
import bhmm


class TestMLHMM_Pathologic(unittest.TestCase):

    def test_1state(self):
        obs = np.array([0, 0, 0, 0, 0], dtype=int)
        hmm = bhmm.estimate_hmm([obs], nstates=1, lag=1, accuracy=1e-6)
        p0_ref = np.array([1.0])
        A_ref = np.array([[1.0]])
        B_ref = np.array([[1.0]])
        assert np.allclose(hmm.initial_distribution, p0_ref)
        assert np.allclose(hmm.transition_matrix, A_ref)
        assert np.allclose(hmm.output_model.output_probabilities, B_ref)

    def test_1state_fail(self):
        obs = np.array([0, 0, 0, 0, 0], dtype=int)
        with self.assertRaises(NotImplementedError):
            bhmm.estimate_hmm([obs], nstates=2, lag=1, accuracy=1e-6)

    def test_2state_step(self):
        obs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
        hmm = bhmm.estimate_hmm([obs], nstates=2, lag=1, accuracy=1e-6)
        p0_ref = np.array([1, 0])
        A_ref = np.array([[0.8, 0.2],
                          [0.0, 1.0]])
        B_ref = np.array([[1, 0],
                          [0, 1]])
        perm = [1, 0]  # permutation
        assert np.allclose(hmm.initial_distribution, p0_ref, atol=1e-5) \
               or np.allclose(hmm.initial_distribution, p0_ref[perm], atol=1e-5)
        assert np.allclose(hmm.transition_matrix, A_ref, atol=1e-5) \
               or np.allclose(hmm.transition_matrix, A_ref[np.ix_(perm, perm)], atol=1e-5)
        assert np.allclose(hmm.output_model.output_probabilities, B_ref, atol=1e-5) \
               or np.allclose(hmm.output_model.output_probabilities, B_ref[[perm]], atol=1e-5)

    def test_2state_2step(self):
        obs = np.array([0, 1, 0], dtype=int)
        hmm = bhmm.estimate_hmm([obs], nstates=2, lag=1, accuracy=1e-6)
        p0_ref = np.array([1, 0])
        A_ref = np.array([[0.0, 1.0],
                          [1.0, 0.0]])
        B_ref = np.array([[1, 0],
                          [0, 1]])
        perm = [1, 0]  # permutation
        assert np.allclose(hmm.initial_distribution, p0_ref, atol=1e-5) \
               or np.allclose(hmm.initial_distribution, p0_ref[perm], atol=1e-5)
        assert np.allclose(hmm.transition_matrix, A_ref, atol=1e-5) \
               or np.allclose(hmm.transition_matrix, A_ref[np.ix_(perm, perm)], atol=1e-5)
        assert np.allclose(hmm.output_model.output_probabilities, B_ref, atol=1e-5) \
               or np.allclose(hmm.output_model.output_probabilities, B_ref[[perm]], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
