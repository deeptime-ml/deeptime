
# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group
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

import unittest

import numpy as np
from tests.numeric import assert_allclose
import scipy.sparse

from msmtools.estimation import log_likelihood

"""Unit tests for the transition_matrix module"""


class TestTransitionMatrix(unittest.TestCase):
    def setUp(self):
        """Small test cases"""
        self.C1 = scipy.sparse.csr_matrix([[1, 3], [3, 1]])

        self.T1 = scipy.sparse.csr_matrix([[0.8, 0.2], [0.9, 0.1]])

        """Zero row sum throws an error"""
        self.T0 = scipy.sparse.csr_matrix([[0, 1], [0.9, 0.1]])

    def tearDown(self):
        pass

    def test_count_matrix(self):
        """Small test cases"""
        log = log_likelihood(self.C1, self.T1)
        assert_allclose(log, np.log(0.8 * 0.2 ** 3 * 0.9 ** 3 * 0.1))


if __name__ == "__main__":
    unittest.main()
