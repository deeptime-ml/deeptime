
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

from sktime.markovprocess.bhmm.output_models.gaussian import GaussianOutputModel


class TestOutputGaussian(unittest.TestCase):

    def setUp(self):
        nstates = 3
        means = np.array([-0.5, 0.0, 0.5])
        sigmas = np.array([0.2, 0.2, 0.2])
        self.G = GaussianOutputModel(nstates, means=means, sigmas=sigmas)

        # random Gaussian samples
        self.obs = np.random.randn(10000)

    def test_p_obs(self):
        for i in range(10):
            p_c = self.G.p_obs(self.obs)
            # TODO: test something useful

if __name__ == "__main__":
    unittest.main()
