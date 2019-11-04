
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

from __future__ import print_function
import numpy as np
import unittest
import time
from bhmm.output_models.gaussian import GaussianOutputModel

print_speedup = False


class TestOutputGaussian(unittest.TestCase):

    def setUp(self):
        nstates = 3
        means = np.array([-0.5, 0.0, 0.5])
        sigmas = np.array([0.2, 0.2, 0.2])
        self.G = GaussianOutputModel(nstates, means=means, sigmas=sigmas)

        # random Gaussian samples
        self.obs = np.random.randn(10000)

    def tearDown(self):
        pass

    def test_p_obs(self):
        # compare results
        self.G.set_implementation('c')
        time1 = time.time()
        for i in range(10):
            p_c = self.G.p_obs(self.obs)
        time2 = time.time()
        t_c = time2-time1

        self.G.set_implementation('python')
        time1 = time.time()
        for i in range(10):
            p_p = self.G.p_obs(self.obs)
        time2 = time.time()
        t_p = time2-time1

        assert(np.allclose(p_c, p_p))

        # speed report
        if print_speedup:
            print('p_obs speedup c/python = '+str(t_p/t_c))


if __name__ == "__main__":
    unittest.main()
