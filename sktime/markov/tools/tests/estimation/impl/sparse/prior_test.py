
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

r"""Unit test for the prior module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import unittest

import numpy as np
from tests.numeric import assert_allclose

from scipy.sparse import csr_matrix

from msmtools.estimation.sparse import prior

from msmtools.util.numeric import allclose_sparse


class TestPrior(unittest.TestCase):
    def setUp(self):
        C = np.array([[4, 4, 0, 2], [4, 4, 1, 0], [0, 1, 4, 4], [0, 0, 4, 4]])
        self.C = csr_matrix(C)

        self.alpha_def = 0.001
        self.alpha = -0.5

        B_neighbor = np.array([[1, 1, 0, 1], [1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1]])
        B_const = np.ones_like(C)
        B_rev = np.triu(B_const)

        self.B_neighbor = csr_matrix(B_neighbor)
        self.B_const = B_const
        self.B_rev = B_rev

    def tearDown(self):
        pass

    def test_prior_neighbor(self):
        Bn = prior.prior_neighbor(self.C)
        self.assertTrue(allclose_sparse(Bn, self.alpha_def * self.B_neighbor))

        Bn = prior.prior_neighbor(self.C, alpha=self.alpha)
        self.assertTrue(allclose_sparse(Bn, self.alpha * self.B_neighbor))

    def test_prior_const(self):
        Bn = prior.prior_const(self.C)
        assert_allclose(Bn, self.alpha_def * self.B_const)

        Bn = prior.prior_const(self.C, alpha=self.alpha)
        assert_allclose(Bn, self.alpha * self.B_const)

    def test_prior_rev(self):
        Bn = prior.prior_rev(self.C)
        assert_allclose(Bn, -1.0 * self.B_rev)

        Bn = prior.prior_rev(self.C, alpha=self.alpha)
        assert_allclose(Bn, self.alpha * self.B_rev)


if __name__ == "__main__":
    unittest.main()
