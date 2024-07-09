import unittest

import numpy as np
from scipy.sparse import csr_matrix
from tests.markov.tools.numeric import assert_allclose

from os.path import abspath, join
from os import pardir

from deeptime.markov.tools.estimation import girsanov_reweighted_count_matrix

testpath = abspath(join(abspath(__file__), pardir)) + '/testfiles/'


class TestGirsanovReweightedCountMatrixMult(unittest.TestCase):
    def setUp(self):
        M = 10
        self.M = M

        """Small test cases"""
        dtraj_short = np.array([0, 0, 1, 0, 1, 1, 0])
        self.dtrajs_short = [dtraj_short for i in range(M)]

        gtraj_short = np.array([1., 1., 1., 1., 1., 1., 1.])
        self.gtrajs_short = [gtraj_short for i in range(M)]

        Mtraj_short = np.array([0., 0., 0., 0., 0., 0., 0.])
        self.Mtrajs_short = [Mtraj_short for i in range(M)]

        self.B1_sliding = M * np.array([[1, 2], [2, 1]])
        self.B2_sliding = M * np.array([[1, 2], [1, 1]])
        self.B3_sliding = M * np.array([[2, 1], [0, 1]])

        self.B1_sliding_4nstates = M * np.array([[1, 2, 0, 0], [2, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.B2_sliding_4nstates = M * np.array([[1, 2, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.B3_sliding_4nstates = M * np.array([[2, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        """Larger test cases"""
        dtraj_long = np.loadtxt(testpath + 'dtraj.dat').astype(int)
        self.gtrajs_long = [np.ones(len(dtraj_long))  for i in range(M)]
        self.Mtrajs_long = [np.zeros(len(dtraj_long)) for i in range(M)]
        self.dtrajs_long = [dtraj_long for i in range(M)]
      
        self.C1_sliding = M * np.loadtxt(testpath + 'C_1_sliding.dat')
        self.C7_sliding = M * np.loadtxt(testpath + 'C_7_sliding.dat')
        self.C13_sliding = M * np.loadtxt(testpath + 'C_13_sliding.dat')

    def tearDown(self):
        pass

    def test_girsanov_reweighted_count_matrix_mult(self):
        """Small test cases"""
        C = girsanov_reweighted_count_matrix(self.dtrajs_short, 1, (self.gtrajs_short, self.Mtrajs_short)).toarray()
        assert_allclose(C, self.B1_sliding)

        C = girsanov_reweighted_count_matrix(self.dtrajs_short, 2, (self.gtrajs_short, self.Mtrajs_short)).toarray()
        assert_allclose(C, self.B2_sliding)

        C = girsanov_reweighted_count_matrix(self.dtrajs_short, 3, (self.gtrajs_short, self.Mtrajs_short)).toarray()
        assert_allclose(C, self.B3_sliding)

        """Larger test cases"""
        C = girsanov_reweighted_count_matrix(self.dtrajs_long, 1, (self.gtrajs_long, self.Mtrajs_long)).toarray()
        assert_allclose(C, self.C1_sliding)

        C = girsanov_reweighted_count_matrix(self.dtrajs_long, 7, (self.gtrajs_long, self.Mtrajs_long)).toarray()
        assert_allclose(C, self.C7_sliding)

        C = girsanov_reweighted_count_matrix(self.dtrajs_long, 13, (self.gtrajs_long, self.Mtrajs_long)).toarray()
        assert_allclose(C, self.C13_sliding)

        """Test raising of value error if lag greater than trajectory length"""
        with self.assertRaises(ValueError):
            C = girsanov_reweighted_count_matrix(self.dtrajs_short, 10, (self.gtrajs_short, self.Mtrajs_short))

    def test_sliding_keyword(self):
        with self.assertRaises(NotImplementedError):
            C = girsanov_reweighted_count_matrix(self.dtrajs_short, 1, (self.gtrajs_short, self.Mtrajs_short), sliding=False)

    def test_sparse_return_keyword(self):
        # dense representation
        C = girsanov_reweighted_count_matrix(self.dtrajs_short, 1, (self.gtrajs_short, self.Mtrajs_short), sparse_return=False)
        assert_allclose(C, self.B1_sliding)

        C = girsanov_reweighted_count_matrix(self.dtrajs_short, 1, (self.gtrajs_short, self.Mtrajs_short), sparse_return=False, nstates=4)
        assert_allclose(C, self.B1_sliding_4nstates)

        C = girsanov_reweighted_count_matrix(self.dtrajs_short, 2, (self.gtrajs_short, self.Mtrajs_short), sparse_return=False, nstates=4)
        assert_allclose(C, self.B2_sliding_4nstates)

        C = girsanov_reweighted_count_matrix(self.dtrajs_short, 3, (self.gtrajs_short, self.Mtrajs_short), sparse_return=False, nstates=4)
        assert_allclose(C, self.B3_sliding_4nstates)

        # sparse representation
        C = girsanov_reweighted_count_matrix(self.dtrajs_short, 1, (self.gtrajs_short, self.Mtrajs_short), sparse_return=True)
        assert_allclose(C.data, csr_matrix(self.B1_sliding).data)
        assert_allclose(C.indices, csr_matrix(self.B1_sliding).indices)
        assert_allclose(C.indptr, csr_matrix(self.B1_sliding).indptr)

        C = girsanov_reweighted_count_matrix(self.dtrajs_short, 1, (self.gtrajs_short, self.Mtrajs_short), sparse_return=True, nstates=4)
        assert_allclose(C.data, csr_matrix(self.B1_sliding_4nstates).data)
        assert_allclose(C.indices, csr_matrix(self.B1_sliding_4nstates).indices)
        assert_allclose(C.indptr, csr_matrix(self.B1_sliding_4nstates).indptr)

        C = girsanov_reweighted_count_matrix(self.dtrajs_short, 2, (self.gtrajs_short, self.Mtrajs_short), sparse_return=True, nstates=4)
        assert_allclose(C.data, csr_matrix(self.B2_sliding_4nstates).data)
        assert_allclose(C.indices, csr_matrix(self.B2_sliding_4nstates).indices)
        assert_allclose(C.indptr, csr_matrix(self.B2_sliding_4nstates).indptr)

        C = girsanov_reweighted_count_matrix(self.dtrajs_short, 3, (self.gtrajs_short, self.Mtrajs_short), sparse_return=True, nstates=4)
        assert_allclose(C.data, csr_matrix(self.B3_sliding_4nstates).data)
        assert_allclose(C.indices, csr_matrix(self.B3_sliding_4nstates).indices)
        assert_allclose(C.indptr, csr_matrix(self.B3_sliding_4nstates).indptr)
    
    def test_nstates_keyword(self):
        C = girsanov_reweighted_count_matrix(self.dtrajs_short, 1, (self.gtrajs_short, self.Mtrajs_short), nstates=10)
        self.assertTrue(C.shape == (10, 10))

        with self.assertRaises(ValueError):
            C = girsanov_reweighted_count_matrix(self.dtrajs_short, 1, (self.gtrajs_short, self.Mtrajs_short), nstates=1)


class TestGirsanovReweightedCountMatrix(unittest.TestCase):
    def setUp(self):
        """Small test cases"""
        self.dtraj_short = np.array([0, 0, 1, 0, 1, 1, 0])
        self.gtraj_short = np.array([1., 1., 1., 1., 1., 1., 1.])
        self.Mtraj_short = np.array([0., 0., 0., 0., 0., 0., 0.])

        self.B1_sliding = np.array([[1, 2], [2, 1]])
        self.B2_sliding = np.array([[1, 2], [1, 1]])
        self.B3_sliding = np.array([[2, 1], [0, 1]])

        self.B1_sliding_4nstates = np.array([[1, 2, 0, 0], [2, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.B2_sliding_4nstates = np.array([[1, 2, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.B3_sliding_4nstates = np.array([[2, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        """Larger test cases"""
        self.dtraj_long = np.loadtxt(testpath + 'dtraj.dat').astype(int)
        self.gtraj_long = np.ones(len(self.dtraj_long))  
        self.Mtraj_long = np.zeros(len(self.dtraj_long)) 

        self.C1_sliding = np.loadtxt(testpath + 'C_1_sliding.dat')
        self.C7_sliding = np.loadtxt(testpath + 'C_7_sliding.dat')
        self.C13_sliding = np.loadtxt(testpath + 'C_13_sliding.dat')

    def tearDown(self):
        pass

    def test_girsanov_reweighted_count_matrix(self):
        """Small test cases"""
        C = girsanov_reweighted_count_matrix(self.dtraj_short, 1, (self.gtraj_short, self.Mtraj_short)).toarray()
        assert_allclose(C, self.B1_sliding)

        C = girsanov_reweighted_count_matrix(self.dtraj_short, 2, (self.gtraj_short, self.Mtraj_short)).toarray()
        assert_allclose(C, self.B2_sliding)

        C = girsanov_reweighted_count_matrix(self.dtraj_short, 3, (self.gtraj_short, self.Mtraj_short)).toarray()
        assert_allclose(C, self.B3_sliding)

        """Larger test cases"""
        C = girsanov_reweighted_count_matrix(self.dtraj_long, 1, (self.gtraj_long, self.Mtraj_long)).toarray()
        assert_allclose(C, self.C1_sliding)

        C = girsanov_reweighted_count_matrix(self.dtraj_long, 7, (self.gtraj_long, self.Mtraj_long)).toarray()
        assert_allclose(C, self.C7_sliding)

        C = girsanov_reweighted_count_matrix(self.dtraj_long, 13, (self.gtraj_long, self.Mtraj_long)).toarray()
        assert_allclose(C, self.C13_sliding)

        """Test raising of value error if lag greater than trajectory length"""
        with self.assertRaises(ValueError):
            C = girsanov_reweighted_count_matrix(self.dtraj_short, 10, (self.gtraj_short, self.Mtraj_short))

    def test_sliding_keyword(self):
        with self.assertRaises(NotImplementedError):
            C = girsanov_reweighted_count_matrix(self.dtraj_short, 1, (self.gtraj_short, self.Mtraj_short), sliding=False)

    def test_sparse_return_keyword(self):
        # dense representation
        C = girsanov_reweighted_count_matrix(self.dtraj_short, 1, (self.gtraj_short, self.Mtraj_short), sparse_return=False)
        assert_allclose(C, self.B1_sliding)

        C = girsanov_reweighted_count_matrix(self.dtraj_short, 1, (self.gtraj_short, self.Mtraj_short), sparse_return=False, nstates=4)
        assert_allclose(C, self.B1_sliding_4nstates)

        C = girsanov_reweighted_count_matrix(self.dtraj_short, 2, (self.gtraj_short, self.Mtraj_short), sparse_return=False, nstates=4)
        assert_allclose(C, self.B2_sliding_4nstates)

        C = girsanov_reweighted_count_matrix(self.dtraj_short, 3, (self.gtraj_short, self.Mtraj_short), sparse_return=False, nstates=4)
        assert_allclose(C, self.B3_sliding_4nstates)

        # sparse representation
        C = girsanov_reweighted_count_matrix(self.dtraj_short, 1, (self.gtraj_short, self.Mtraj_short), sparse_return=True)
        assert_allclose(C.data, csr_matrix(self.B1_sliding).data)
        assert_allclose(C.indices, csr_matrix(self.B1_sliding).indices)
        assert_allclose(C.indptr, csr_matrix(self.B1_sliding).indptr)

        C = girsanov_reweighted_count_matrix(self.dtraj_short, 1, (self.gtraj_short, self.Mtraj_short), sparse_return=True, nstates=4)
        assert_allclose(C.data, csr_matrix(self.B1_sliding_4nstates).data)
        assert_allclose(C.indices, csr_matrix(self.B1_sliding_4nstates).indices)
        assert_allclose(C.indptr, csr_matrix(self.B1_sliding_4nstates).indptr)

        C = girsanov_reweighted_count_matrix(self.dtraj_short, 2, (self.gtraj_short, self.Mtraj_short), sparse_return=True, nstates=4)
        assert_allclose(C.data, csr_matrix(self.B2_sliding_4nstates).data)
        assert_allclose(C.indices, csr_matrix(self.B2_sliding_4nstates).indices)
        assert_allclose(C.indptr, csr_matrix(self.B2_sliding_4nstates).indptr)

        C = girsanov_reweighted_count_matrix(self.dtraj_short, 3, (self.gtraj_short, self.Mtraj_short), sparse_return=True, nstates=4)
        assert_allclose(C.data, csr_matrix(self.B3_sliding_4nstates).data)
        assert_allclose(C.indices, csr_matrix(self.B3_sliding_4nstates).indices)
        assert_allclose(C.indptr, csr_matrix(self.B3_sliding_4nstates).indptr)

    
    def test_nstates_keyword(self):
        C = girsanov_reweighted_count_matrix(self.dtraj_short, 1, (self.gtraj_short, self.Mtraj_short), nstates=10)
        self.assertTrue(C.shape == (10, 10))

        with self.assertRaises(ValueError):
            C = girsanov_reweighted_count_matrix(self.dtraj_short, 1, (self.gtraj_short, self.Mtraj_short), nstates=1)

class TestArguments(unittest.TestCase):
    def testInputList(self):
        dtrajs = [0, 1, 2, 0, 0, 1, 2, 1, 0]
        gtrajs = np.ones(len(dtrajs))
        Mtrajs = np.zeros(len(dtrajs))
        girsanov_reweighted_count_matrix(dtrajs, 1, [list(gtrajs),list(Mtrajs)])
        girsanov_reweighted_count_matrix(dtrajs, 1, (list(gtrajs),list(Mtrajs)))
        

    def testInputArray(self):
        dtrajs = np.array([0, 1, 2, 0, 0, 1, 2, 1, 0])
        gtrajs = np.ones(len(dtrajs))
        Mtrajs = np.zeros(len(dtrajs))
        girsanov_reweighted_count_matrix(dtrajs, 1, [gtrajs,Mtrajs])
        girsanov_reweighted_count_matrix(dtrajs, 1, np.array([gtrajs,Mtrajs]))
        girsanov_reweighted_count_matrix(dtrajs, 1, (gtrajs,Mtrajs))

    def testInputNestedLists(self):
        dtrajs = [[0, 1, 2, 0, 0, 1, 2, 1, 0],
                  [0, 1, 0, 1, 1, 1, 1, 0, 2]]
        gtrajs = np.ones_like(dtrajs).astype(float)
        Mtrajs = np.zeros_like(dtrajs).astype(float)
        girsanov_reweighted_count_matrix(dtrajs, 1, [gtrajs,Mtrajs])
        girsanov_reweighted_count_matrix(dtrajs, 1, np.array([gtrajs,Mtrajs]))
        girsanov_reweighted_count_matrix(dtrajs, 1, (gtrajs,Mtrajs))

    def testInputNestedListsDiffSize(self):
        dtrajs = [[0, 1, 2, 0, 0, 1, 2, 1, 0],
                  [0, 1, 0, 1, 1, 1, 1, 0, 2, 1, 2, 1]]
        gtrajs = [[1., 1, 1., 1, 1, 1, 1, 1, 1],
                  [1., 1, 1, 1, 1, 1, 1, 1., 1, 1, 1, 1]]
        Mtrajs = [[0, 0., 0, 0, 0, 0., 0, 0, 0],
                  [0, 0, 0, 0, 0, 0., 0., 0, 0, 0., 0, 0]]
        girsanov_reweighted_count_matrix(dtrajs, 1, [gtrajs,Mtrajs])
        girsanov_reweighted_count_matrix(dtrajs, 1, (gtrajs,Mtrajs))

    def testInputInt(self):
        '''int input for reweighting factors is not supported'''
        dtrajs = [[0, 1, 2, 0, 0, 1, 2, 1, 0],
                  [0, 1, 0, 1, 1, 1, 1, 0, 2]]
        gtrajs = np.ones_like(dtrajs)
        Mtrajs = np.zeros_like(dtrajs)
        
        with self.assertRaises(ValueError):
            girsanov_reweighted_count_matrix(dtrajs, 1, (gtrajs,Mtrajs))

    def testInputTupleOfTuple(self):
        '''input for nested tuples for reweighting factors is not supported'''
        dtrajs = [[0, 1, 2, 0, 0, 1, 2, 1, 0],
                  [0, 1, 0, 1, 1, 1, 1, 0, 2]]
        reweighting_factors = ((np.ones((9,)),np.ones((9,))),(np.ones((9,)),np.ones((9,))))
        
        with self.assertRaises(ValueError):
            girsanov_reweighted_count_matrix(dtrajs, 1, reweighting_factors)

    def testInputWrongShapes(self):
        '''input for nested tuples for reweighting factors is not supported'''
        dtrajs = [[0, 1, 2, 0, 0, 1, 2, 1, 0],
                  [0, 1, 0, 1, 1, 1, 1, 0, 2]]
        
        reweighting_factors = ([np.ones((9,)),np.ones((9,)),np.ones((9,)),np.ones((9,))])
        with self.assertRaises(ValueError):
            girsanov_reweighted_count_matrix(dtrajs, 1, reweighting_factors)
        
        reweighting_factors = (np.ones((9,)),np.ones((9,)),np.ones((9,)),np.ones((9,)))
        with self.assertRaises(ValueError):
            girsanov_reweighted_count_matrix(dtrajs, 1, reweighting_factors)
