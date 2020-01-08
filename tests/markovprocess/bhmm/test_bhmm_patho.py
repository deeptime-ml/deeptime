
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

from sktime.markovprocess import bhmm


class TestBHMMPathological(unittest.TestCase):

    def test_2state_rev_step(self):
        obs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
        mle = bhmm.estimate_hmm([obs], n_states=2, lag=1)
        # this will generate disconnected count matrices and should fail:
        with self.assertRaises(NotImplementedError):
            bhmm.bayesian_hmm([obs], mle, reversible=True, p0_prior=None, transition_matrix_prior=None)

    def test_2state_nonrev_step(self):
        obs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
        mle = bhmm.estimate_hmm([obs], n_states=2, lag=1)
        sampled = bhmm.bayesian_hmm([obs], mle, reversible=False, nsample=2000,
                                    p0_prior='mixed', transition_matrix_prior='mixed').fetch_model()
        tmatrix_samples = np.array([s.transition_matrix for s in sampled])
        std = tmatrix_samples.std(axis=0)
        assert np.all(std[0] > 0)
        assert np.max(np.abs(std[1])) < 1e-3

    def test_2state_rev_2step(self):
        obs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0], dtype=int)
        mle = bhmm.estimate_hmm([obs], n_states=2, lag=1)
        sampled = bhmm.bayesian_hmm([obs], mle, reversible=False, nsample=100,
                                    p0_prior='mixed', transition_matrix_prior='mixed').fetch_model()
        tmatrix_samples = np.array([s.transition_matrix for s in sampled])
        std = tmatrix_samples.std(axis=0)
        assert np.all(std > 0)


if __name__ == "__main__":
    unittest.main()
