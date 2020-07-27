
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

import unittest

from tests.numeric import assert_allclose
import scipy.sparse

from msmtools.estimation.sparse import transition_matrix

"""Unit tests for the transition_matrix module"""


class TestTransitionMatrixNonReversible(unittest.TestCase):
    def setUp(self):
        """Small test cases"""
        self.C1 = scipy.sparse.csr_matrix([[1, 3], [3, 1]])
        self.C2 = scipy.sparse.csr_matrix([[0, 2], [1, 1]])

        self.T1 = scipy.sparse.csr_matrix([[0.25, 0.75], [0.75, 0.25]])
        self.T2 = scipy.sparse.csr_matrix([[0, 1], [0.5, 0.5]])

        """Zero row sum throws an error"""
        self.C0 = scipy.sparse.csr_matrix([[0, 0], [3, 1]])

    def tearDown(self):
        pass

    def test_count_matrix(self):
        """Small test cases"""
        T = transition_matrix.transition_matrix_non_reversible(self.C1).toarray()
        assert_allclose(T, self.T1.toarray())

        T = transition_matrix.transition_matrix_non_reversible(self.C1).toarray()
        assert_allclose(T, self.T1.toarray())


if __name__ == "__main__":
    unittest.main()
