r"""This module provides unit tests for the expectations module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest
import numpy as np
from tests.markov.tools.numeric import assert_allclose
from scipy.linalg import eig

from deeptime.markov.tools.analysis import _expectations


class TestEcMatrixVector(unittest.TestCase):
    def setUp(self):
        self.dim = 100
        C = np.random.randint(0, 50, size=(self.dim, self.dim))
        C = 0.5 * (C + np.transpose(C))
        self.T = C / np.sum(C, axis=1)[:, np.newaxis]
        """Eigenvalues and left eigenvectors, sorted"""
        v, L = eig(np.transpose(self.T))
        ind = np.argsort(np.abs(v))[::-1]
        v = v[ind]
        L = L[:, ind]
        """Compute stationary distribution"""
        self.mu = L[:, 0] / np.sum(L[:, 0])

        pass

    def tearDown(self):
        pass

    def test_ec_matrix_vector(self):
        p0 = self.mu
        T = self.T

        N = 20
        EC_n = _expectations.ec_matrix_vector(p0, T, N)

        """
        If p0 is the stationary vector the computation can
        be carried out by a simple multiplication
        """
        EC_true = N * self.mu[:, np.newaxis] * T
        assert_allclose(EC_true, EC_n)

        """Zero length chain"""
        N = 0
        EC_n = _expectations.ec_matrix_vector(p0, T, N)
        EC_true = np.zeros(T.shape)
        assert_allclose(EC_true, EC_n)


class TestEcGeometricSeries(unittest.TestCase):
    def setUp(self):
        self.dim = 100
        C = np.random.randint(0, 50, size=(self.dim, self.dim))
        C = 0.5 * (C + np.transpose(C))
        self.T = C / np.sum(C, axis=1)[:, np.newaxis]
        """Eigenvalues and left eigenvectors, sorted"""
        v, L = eig(np.transpose(self.T))
        ind = np.argsort(np.abs(v))[::-1]
        v = v[ind]
        L = L[:, ind]
        """Compute stationary distribution"""
        self.mu = L[:, 0] / np.sum(L[:, 0])

        pass

    def tearDown(self):
        pass

    def test_ec_geometric_series(self):
        p0 = self.mu
        T = self.T

        N = 2000
        EC_n = _expectations.ec_geometric_series(p0, T, N)

        """
        If p0 is the stationary vector the computation can
        be carried out by a simple multiplication
        """
        EC_true = N * self.mu[:, np.newaxis] * T
        assert_allclose(EC_true, EC_n)

        """Zero length chain"""
        N = 0
        EC_n = _expectations.ec_geometric_series(p0, T, N)
        EC_true = np.zeros(T.shape)
        assert_allclose(EC_true, EC_n)


class TestGeometricSeries(unittest.TestCase):
    def setUp(self):
        self.q = 2.0
        self.q_array = np.array([2.0, 1.0, 0.8, -0.3, -1.0, -2.0])
        self.n = 9

        self.s = 0
        for i in range(self.n + 1):
            self.s += self.q ** i

        self.s_array = np.zeros(self.q_array.shape)
        for i in range(self.n + 1):
            self.s_array += self.q_array ** i

    def tearDown(self):
        pass

    def test_geometric_series(self):
        x = _expectations.geometric_series(self.q, self.n)
        assert_allclose(x, self.s)

        x = _expectations.geometric_series(self.q_array, self.n)
        assert_allclose(x, self.s_array)

        """Assert ValueError for negative n"""
        with self.assertRaises(ValueError):
            _expectations.geometric_series(self.q, -2)

        with self.assertRaises(ValueError):
            _expectations.geometric_series(self.q_array, -2)
