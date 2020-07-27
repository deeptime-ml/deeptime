# This file is part of scikit-time and MSMTools.
#
# Copyright (c) 2020, 2015, 2014 AI4Science Group, Freie Universitaet Berlin (GER)
#
# scikit-time and MSMTools is free software: you can redistribute it and/or modify
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

r"""Unit test for the reversible mle newton module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest

from os.path import abspath, join
from os import pardir

import numpy as np
from scipy.sparse import csr_matrix

from tests.markov.tools.numeric import assert_allclose
from sktime.markov.tools.estimation.sparse.mle.newton.mle_rev import solve_mle_rev

testpath = abspath(join(abspath(__file__), pardir)) + '/testfiles/'


class TestReversibleEstimatorNewton(unittest.TestCase):
    def setUp(self):
        """Count matrix for bith-death chain with 100 states"""
        self.C = np.loadtxt(testpath + 'C.dat')
        self.P_ref = self.C/self.C.sum(axis=1)[:,np.newaxis]

    def test_estimator(self):
        P, pi = solve_mle_rev(csr_matrix(self.C))
        assert_allclose(P.toarray(), self.P_ref)


if __name__ == "__main__":
    unittest.main()
