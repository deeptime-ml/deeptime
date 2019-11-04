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
from bhmm.init.gaussian import init_model_gaussian1d
import msmtools.analysis as msmana


class TestHMM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from bhmm import testsystems
        cls._nstates = 3
        cls._model, cls._observations, cls._states = testsystems.generate_synthetic_observations(nstates=cls._nstates,
                                                                                                 output='gaussian')

    def test_gmm(self):
        # Fit a Gaussian mixture model to obtain emission distributions and state stationary probabilities.
        from bhmm._external.sklearn import mixture
        gmm = mixture.GMM(n_components=self._nstates)
        gmm.fit(np.concatenate(self._observations)[:,None])
        assert gmm.n_components == self._nstates
        assert np.all(gmm.weights_ > 0)  # make sure we don't have empty states.

    def test_init(self):
        initial_model = init_model_gaussian1d(self._observations, self._nstates)
        assert initial_model.nstates == self._nstates
        assert msmana.is_transition_matrix(initial_model.transition_matrix)



if __name__ == "__main__":
    unittest.main()
