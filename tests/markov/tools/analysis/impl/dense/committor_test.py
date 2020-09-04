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

r"""Unit tests for the committor module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest
import numpy as np

from sktime.data import birth_death_chain

from tests.markov.tools.numeric import assert_allclose

from sktime.markov.tools.analysis.dense import committor


class TestCommittor(unittest.TestCase):
    def setUp(self):
        p = np.zeros(10)
        q = np.zeros(10)
        p[0:-1] = 0.5
        q[1:] = 0.5
        p[4] = 0.01
        q[6] = 0.1

        self.bdc = birth_death_chain(q, p)

    def test_forward_comittor(self):
        P = self.bdc.transition_matrix
        un = committor.forward_committor(P, [0, 1], [8, 9])
        u = self.bdc.committor_forward(1, 8)
        assert_allclose(un, u)

    def test_backward_comittor(self):
        P = self.bdc.transition_matrix
        un = committor.backward_committor(P, [0, 1], [8, 9])
        u = self.bdc.committor_backward(1, 8)
        assert_allclose(un, u)
