
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

r"""Unit test, dense implementation of hitting probabilities

.. moduleauthor:: F.Noe <frank DOT noe AT fu-berlin DOT de>

"""
import unittest

import numpy as np
from tests.numeric import assert_allclose

from msmtools.analysis.dense.hitting_probability import hitting_probability


class TestHitting(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_hitting1(self):
        P = np.array([[0., 1., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.]])
        sol = np.array([1, 0, 0])
        assert_allclose(hitting_probability(P, 1), sol)
        assert_allclose(hitting_probability(P, [1, 2]), sol)

    def test_hitting2(self):
        P = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.1, 0.8, 0.1, 0.0],
                      [0.0, 0.0, 0.8, 0.2],
                      [0.0, 0.0, 0.2, 0.8]])
        sol = np.array([0., 0.5, 1., 1.])
        assert_allclose(hitting_probability(P, [2, 3]), sol)

    def test_hitting3(self):
        P = np.array([[0.9, 0.1, 0.0, 0.0, 0.0],
                      [0.1, 0.9, 0.0, 0.0, 0.0],
                      [0.0, 0.1, 0.4, 0.5, 0.0],
                      [0.0, 0.0, 0.0, 0.8, 0.2],
                      [0.0, 0.0, 0.0, 0.2, 0.8]])
        sol = np.array([0.0, 0.0, 8.33333333e-01, 1.0, 1.0])
        assert_allclose(hitting_probability(P, 3), sol)
        assert_allclose(hitting_probability(P, [3, 4]), sol)


if __name__ == "__main__":
    unittest.main()
