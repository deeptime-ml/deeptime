
# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# MSMTools is free software: you can redistribute it and/or modify
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

r"""Unit test for the pathways-module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest
import numpy as np
from scipy.sparse import csr_matrix

from msmtools.flux.sparse.pathways import pathways
from msmtools.flux.sparse.tpt import flux_matrix
from msmtools.analysis import committor, statdist
from tests.numeric import assert_allclose

class TestPathways(unittest.TestCase):

    def setUp(self):
        P = np.array([[0.8,  0.15, 0.05,  0.0,  0.0],
                      [0.1,  0.75, 0.05, 0.05, 0.05],
                      [0.05,  0.1,  0.8,  0.0,  0.05],
                      [0.0,  0.2, 0.0,  0.8,  0.0],
                      [0.0,  0.02, 0.02, 0.0,  0.96]])
        P = csr_matrix(P)
        A = [0]
        B = [4]
        mu = statdist(P)
        qminus = committor(P, A, B, forward=False, mu=mu)
        qplus = committor(P, A, B, forward=True, mu=mu)
        self.A = A
        self.B = B
        self.F = flux_matrix(P, mu, qminus, qplus, netflux=True)

        self.paths = [np.array([0, 1, 4]), np.array([0, 2, 4]), np.array([0, 1, 2, 4])]
        self.capacities = [0.0072033898305084252, 0.0030871670702178975, 0.00051452784503631509]
    def test_pathways(self):
        paths, capacities = pathways(self.F, self.A, self.B)
        assert_allclose(capacities, self.capacities)
        N = len(paths)
        for i in range(N):
            self.assertTrue(np.all(paths[i] == self.paths[i]))


if __name__=="__main__":
    unittest.main()
