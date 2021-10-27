import unittest

from tests.markov.tools.numeric import assert_allclose
import scipy.sparse

from deeptime.markov.tools.estimation.sparse import transition_matrix

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
