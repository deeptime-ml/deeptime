
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

r"""Unit tests for the covariance module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import unittest

import numpy as np
from tests.numeric import assert_allclose

from msmtools.estimation.dense.covariance import tmatrix_cov, dirichlet_covariance, error_perturbation


class TestCovariance(unittest.TestCase):
    def setUp(self):
        alpha1 = np.array([1.0, 2.0, 1.0])
        cov1 = 1.0 / 80 * np.array([[3.0, -2.0, -1.0], [-2.0, 4.0, -2.0], [-1.0, -2.0, 3.0]])

        alpha2 = np.array([2.0, 1.0, 2.0])
        cov2 = 1.0 / 150 * np.array([[6, -2, -4], [-2, 4, -2], [-4, -2, 6]])

        self.C = np.zeros((3, 3))
        self.C[0, :] = alpha1 - 1.0
        self.C[1, :] = alpha2 - 1.0
        self.C[2, :] = alpha1 - 1.0

        self.cov = np.zeros((3, 3, 3))
        self.cov[0, :, :] = cov1
        self.cov[1, :, :] = cov2
        self.cov[2, :, :] = cov1

    def tearDown(self):
        pass

    def test_tmatrix_cov(self):
        cov = tmatrix_cov(self.C)
        assert_allclose(cov, self.cov)

        cov = tmatrix_cov(self.C, row=1)
        assert_allclose(cov, self.cov[1, :, :])


class TestDirichletCovariance(unittest.TestCase):
    def setUp(self):
        self.alpha1 = np.array([1.0, 2.0, 1.0])
        self.cov1 = 1.0 / 80 * np.array([[3.0, -2.0, -1.0], [-2.0, 4.0, -2.0], [-1.0, -2.0, 3.0]])

        self.alpha2 = np.array([2.0, 1.0, 2.0])
        self.cov2 = 1.0 / 150 * np.array([[6, -2, -4], [-2, 4, -2], [-4, -2, 6]])

    def tearDown(self):
        pass

    def test_dirichlet_covariance(self):
        cov = dirichlet_covariance(self.alpha1)
        assert_allclose(cov, self.cov1)

        cov = dirichlet_covariance(self.alpha2)
        assert_allclose(cov, self.cov2)


class TestErrorPerturbation(unittest.TestCase):
    def setUp(self):
        alpha1 = np.array([1.0, 2.0, 1.0])
        cov1 = 1.0 / 80 * np.array([[3.0, -2.0, -1.0], [-2.0, 4.0, -2.0], [-1.0, -2.0, 3.0]])

        alpha2 = np.array([2.0, 1.0, 2.0])
        cov2 = 1.0 / 150 * np.array([[6, -2, -4], [-2, 4, -2], [-4, -2, 6]])

        self.C = np.zeros((3, 3))
        self.C[0, :] = alpha1 - 1.0
        self.C[1, :] = alpha2 - 1.0
        self.C[2, :] = alpha1 - 1.0

        self.cov = np.zeros((3, 3, 3))
        self.cov[0, :, :] = cov1
        self.cov[1, :, :] = cov2
        self.cov[2, :, :] = cov1

        """Scalar-valued observable f(P)=P_11+P_22+P_33"""
        self.S1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        """Vector-valued observable f(P)=(P_11, P_12)"""
        self.S2 = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                            [[0, 1, 0], [0, 0, 0], [0, 0, 0]]])

        """Error-perturbation scalar observable"""
        self.x = (self.S1[:, :, np.newaxis] * self.cov * self.S1[:, np.newaxis, :]).sum()

        """Error-perturbation vector observable"""
        tmp = self.S2[:, np.newaxis, :, :, np.newaxis] * self.cov[np.newaxis, np.newaxis, :, :, :] * \
              self.S2[np.newaxis, :, :, np.newaxis, :]
        self.X = np.sum(tmp, axis=(2, 3, 4))

    def tearDown(self):
        pass

    def test_error_perturbation(self):
        xn = error_perturbation(self.C, self.S1)
        assert_allclose(xn, self.x)

        Xn = error_perturbation(self.C, self.S2)
        assert_allclose(Xn, self.X)


if __name__ == "__main__":
    unittest.main()
