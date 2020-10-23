"""Unit tests for the count_matrix module"""

import unittest

import numpy as np
from tests.markov.tools.numeric import assert_allclose

from os.path import abspath, join
from os import pardir

from deeptime.markov.tools.estimation import count_matrix

testpath = abspath(join(abspath(__file__), pardir)) + '/testfiles/'



class TestCountMatrixtMult(unittest.TestCase):
    def setUp(self):
        M = 10
        self.M = M

        """Small test cases"""
        dtraj_short = np.array([0, 0, 1, 0, 1, 1, 0])
        self.dtrajs_short = [dtraj_short for i in range(M)]

        self.B1_lag = M * np.array([[1, 2], [2, 1]])
        self.B2_lag = M * np.array([[0, 1], [1, 1]])
        self.B3_lag = M * np.array([[2, 0], [0, 0]])

        self.B1_sliding = M * np.array([[1, 2], [2, 1]])
        self.B2_sliding = M * np.array([[1, 2], [1, 1]])
        self.B3_sliding = M * np.array([[2, 1], [0, 1]])

        """Larger test cases"""
        dtraj_long = np.loadtxt(testpath + 'dtraj.dat').astype(int)
        self.dtrajs_long = [dtraj_long for i in range(M)]
        self.C1_lag = M * np.loadtxt(testpath + 'C_1_lag.dat')
        self.C7_lag = M * np.loadtxt(testpath + 'C_7_lag.dat')
        self.C13_lag = M * np.loadtxt(testpath + 'C_13_lag.dat')

        self.C1_sliding = M * np.loadtxt(testpath + 'C_1_sliding.dat')
        self.C7_sliding = M * np.loadtxt(testpath + 'C_7_sliding.dat')
        self.C13_sliding = M * np.loadtxt(testpath + 'C_13_sliding.dat')

    def tearDown(self):
        pass

    def test_count_matrix_mult(self):
        """Small test cases"""
        C = count_matrix(self.dtrajs_short, 1, sliding=False).toarray()
        assert_allclose(C, self.B1_lag)

        C = count_matrix(self.dtrajs_short, 2, sliding=False).toarray()
        assert_allclose(C, self.B2_lag)

        C = count_matrix(self.dtrajs_short, 3, sliding=False).toarray()
        assert_allclose(C, self.B3_lag)

        C = count_matrix(self.dtrajs_short, 1).toarray()
        assert_allclose(C, self.B1_sliding)

        C = count_matrix(self.dtrajs_short, 2).toarray()
        assert_allclose(C, self.B2_sliding)

        C = count_matrix(self.dtrajs_short, 3).toarray()
        assert_allclose(C, self.B3_sliding)

        """Larger test cases"""
        C = count_matrix(self.dtrajs_long, 1, sliding=False).toarray()
        assert_allclose(C, self.C1_lag)

        C = count_matrix(self.dtrajs_long, 7, sliding=False).toarray()
        assert_allclose(C, self.C7_lag)

        C = count_matrix(self.dtrajs_long, 13, sliding=False).toarray()
        assert_allclose(C, self.C13_lag)

        C = count_matrix(self.dtrajs_long, 1).toarray()
        assert_allclose(C, self.C1_sliding)

        C = count_matrix(self.dtrajs_long, 7).toarray()
        assert_allclose(C, self.C7_sliding)

        C = count_matrix(self.dtrajs_long, 13).toarray()
        assert_allclose(C, self.C13_sliding)

        """Test raising of value error if lag greater than trajectory length"""
        with self.assertRaises(ValueError):
            C = count_matrix(self.dtrajs_short, 10)

    def test_nstates_keyword(self):
        C = count_matrix(self.dtrajs_short, 1, sliding=False, nstates=10)
        self.assertTrue(C.shape == (10, 10))

        with self.assertRaises(ValueError):
            C = count_matrix(self.dtrajs_short, 1, sliding=False, nstates=1)


class TestCountMatrix(unittest.TestCase):
    def setUp(self):
        """Small test cases"""
        self.S_short = np.array([0, 0, 1, 0, 1, 1, 0])
        self.B1_lag = np.array([[1, 2], [2, 1]])
        self.B2_lag = np.array([[0, 1], [1, 1]])
        self.B3_lag = np.array([[2, 0], [0, 0]])

        self.B1_sliding = np.array([[1, 2], [2, 1]])
        self.B2_sliding = np.array([[1, 2], [1, 1]])
        self.B3_sliding = np.array([[2, 1], [0, 1]])

        """Larger test cases"""
        self.S_long = np.loadtxt(testpath + 'dtraj.dat').astype(int)
        self.C1_lag = np.loadtxt(testpath + 'C_1_lag.dat')
        self.C7_lag = np.loadtxt(testpath + 'C_7_lag.dat')
        self.C13_lag = np.loadtxt(testpath + 'C_13_lag.dat')

        self.C1_sliding = np.loadtxt(testpath + 'C_1_sliding.dat')
        self.C7_sliding = np.loadtxt(testpath + 'C_7_sliding.dat')
        self.C13_sliding = np.loadtxt(testpath + 'C_13_sliding.dat')

    def tearDown(self):
        pass

    def test_count_matrix(self):
        """Small test cases"""
        C = count_matrix(self.S_short, 1, sliding=False).toarray()
        assert_allclose(C, self.B1_lag)

        C = count_matrix(self.S_short, 2, sliding=False).toarray()
        assert_allclose(C, self.B2_lag)

        C = count_matrix(self.S_short, 3, sliding=False).toarray()
        assert_allclose(C, self.B3_lag)

        C = count_matrix(self.S_short, 1).toarray()
        assert_allclose(C, self.B1_sliding)

        C = count_matrix(self.S_short, 2).toarray()
        assert_allclose(C, self.B2_sliding)

        C = count_matrix(self.S_short, 3).toarray()
        assert_allclose(C, self.B3_sliding)

        """Larger test cases"""
        C = count_matrix(self.S_long, 1, sliding=False).toarray()
        assert_allclose(C, self.C1_lag)

        C = count_matrix(self.S_long, 7, sliding=False).toarray()
        assert_allclose(C, self.C7_lag)

        C = count_matrix(self.S_long, 13, sliding=False).toarray()
        assert_allclose(C, self.C13_lag)

        C = count_matrix(self.S_long, 1).toarray()
        assert_allclose(C, self.C1_sliding)

        C = count_matrix(self.S_long, 7).toarray()
        assert_allclose(C, self.C7_sliding)

        C = count_matrix(self.S_long, 13).toarray()
        assert_allclose(C, self.C13_sliding)

        """Test raising of value error if lag greater than trajectory length"""
        with self.assertRaises(ValueError):
            C = count_matrix(self.S_short, 10)

    def test_nstates_keyword(self):
        C = count_matrix(self.S_short, 1, nstates=10)
        self.assertTrue(C.shape == (10, 10))

        with self.assertRaises(ValueError):
            C = count_matrix(self.S_short, 1, nstates=1)


class TestArguments(unittest.TestCase):
    def testInputList(self):
        dtrajs = [0, 1, 2, 0, 0, 1, 2, 1, 0]
        count_matrix(dtrajs, 1)

    def testInput1Array(self):
        dtrajs = np.array([0, 1, 2, 0, 0, 1, 2, 1, 0])
        count_matrix(dtrajs, 1)

    def testInputNestedLists(self):
        dtrajs = [[0, 1, 2, 0, 0, 1, 2, 1, 0],
                  [0, 1, 0, 1, 1, 1, 1, 0, 2]]
        count_matrix(dtrajs, 1)

    def testInputNestedListsDiffSize(self):
        dtrajs = [[0, 1, 2, 0, 0, 1, 2, 1, 0],
                  [0, 1, 0, 1, 1, 1, 1, 0, 2, 1, 2, 1]]
        count_matrix(dtrajs, 1)

    def testInputArray(self):
        dtrajs = np.array([0, 1, 2, 0, 0, 1, 2, 1, 0])
        count_matrix(dtrajs, 1)

    def testInputArrays(self):
        """ this is not supported, has to be list of ndarrays """
        dtrajs = np.array([[0, 1, 2, 0, 0, 1, 2, 1, 0],
                           [0, 1, 2, 0, 0, 1., 2, 1, 1]])

        with self.assertRaises(ValueError):
            count_matrix(dtrajs, 1)

    def testInputFloat(self):
        dtraj_with_floats = [0.0, 1, 0, 2, 3, 1, 0.1]
        # dtraj_int = [0, 1, 0, 2, 3, 1, 0]
        with self.assertRaises(ValueError):
            count_matrix(dtraj_with_floats, 1)
