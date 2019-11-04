
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

import bhmm
from bhmm.util import testsystems

from numpy.testing import assert_array_almost_equal


class TestHMM(unittest.TestCase):

    def test_hmm(self):
        # Create a simple HMM model.
        model = testsystems.dalton_model(nstates=3)
        # Test model parameter access.
        np.testing.assert_equal(model.transition_matrix.shape, (3, 3))
        np.testing.assert_equal(model.stationary_distribution.shape, (3, ))

        return

    def test_two_state_model(self):
        """Test the creation of a simple two-state HMM model with analytical parameters.
        """
        from bhmm import HMM
        # Create a simple two-state model.
        nstates = 2
        Tij = testsystems.generate_transition_matrix(reversible=True)
        # stationary distribution
        import msmtools.analysis as msmana
        Pi = msmana.stationary_distribution(Tij)
        from bhmm import GaussianOutputModel
        means = [-1, +1]
        sigmas = [1, 1]
        output_model = GaussianOutputModel(nstates, means=means, sigmas=sigmas)
        model = bhmm.HMM(Pi, Tij, output_model)
        # Test model is correct.
        assert_array_almost_equal(model.transition_matrix, Tij)
        assert_array_almost_equal(model.stationary_distribution, Pi)
        assert(np.allclose(model.output_model.means, np.array(means)))
        assert(np.allclose(model.output_model.sigmas, np.array(sigmas)))

    def test_attributes(self):
        """ Tests that attributes used in properties remain in sync
        """
        model = testsystems.force_spectroscopy_model()
        assert(hasattr(model, 'is_stationary'))
        assert(hasattr(model, '_nstates'))
        assert(hasattr(model, '_ensure_spectral_decomposition'))
        assert(hasattr(model, '_spectral_decomp_available'))
        assert(hasattr(model, '_Pi'))
        assert(hasattr(model, '_Tij'))


if __name__ == "__main__":
    unittest.main()
