
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

r"""Unit tests for the committor module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import unittest
import numpy as np
from msmtools.util.birth_death_chain import BirthDeathChain
from tests.numeric import assert_allclose

from msmtools.analysis.sparse import committor


class TestCommittor(unittest.TestCase):
    def setUp(self):
        p = np.zeros(100)
        q = np.zeros(100)
        p[0:-1] = 0.5
        q[1:] = 0.5
        p[49] = 0.01
        q[51] = 0.1

        self.bdc = BirthDeathChain(q, p)

    def tearDown(self):
        pass

    def test_forward_comittor(self):
        P = self.bdc.transition_matrix_sparse()
        un = committor.forward_committor(P, list(range(10)), list(range(90, 100)))
        u = self.bdc.committor_forward(9, 90)
        assert_allclose(un, u)

    def test_backward_comittor(self):
        P = self.bdc.transition_matrix_sparse()
        un = committor.backward_committor(P, list(range(10)), list(range(90, 100)))
        u = self.bdc.committor_backward(9, 90)
        assert_allclose(un, u)


if __name__ == "__main__":
    unittest.main()
