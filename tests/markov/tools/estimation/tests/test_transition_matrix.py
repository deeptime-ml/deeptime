import unittest
import warnings

import numpy as np
from tests.markov.tools.numeric import assert_allclose
import scipy.sparse

from deeptime.markov.tools.estimation import transition_matrix, tmatrix_cov, error_perturbation

"""Various unit tests for the transition_matrix module"""


class TestTransitionMatrixSparse(unittest.TestCase):
    def setUp(self):
        """Small test cases"""
        self.C1 = scipy.sparse.csr_matrix([[1, 3], [3, 1]])
        self.C2 = scipy.sparse.csr_matrix([[0, 2], [1, 1]])

        self.T1 = scipy.sparse.csr_matrix([[0.25, 0.75], [0.75, 0.25]])
        self.T2 = scipy.sparse.csr_matrix([[0, 1], [0.5, 0.5]])

        self.pi1 = np.array([0.25, 0.25])
        self.pi2 = np.array([1.0 / 3, 2.0 / 3])

        """Zero row sum throws an error"""
        self.C0 = scipy.sparse.csr_matrix([[0, 0], [3, 1]])

    def test_transition_matrix(self):
        """Non-reversible"""
        T = transition_matrix(self.C1).toarray()
        assert_allclose(T, self.T1.toarray())

        T = transition_matrix(self.C2).toarray()
        assert_allclose(T, self.T2.toarray())

        """Reversible"""
        T = transition_matrix(self.C1, reversible=True).toarray()
        assert_allclose(T, self.T1.toarray())

        T = transition_matrix(self.C2, reversible=True).toarray()
        assert_allclose(T, self.T2.toarray())

        """Reversible with fixed pi"""
        T = transition_matrix(self.C1, reversible=True, mu=self.pi1).toarray()
        assert_allclose(T, self.T1.toarray())

        """Reversible with fixed pi"""
        T = transition_matrix(self.C1, reversible=True, mu=self.pi1, method='sparse').toarray()
        assert_allclose(T, self.T1.toarray())

        T = transition_matrix(self.C2, reversible=True, mu=self.pi2).toarray()
        assert_allclose(T, self.T2.toarray())


class TestTransitionRevPiSym(unittest.TestCase):
    def setUp(self):
        """Small test cases"""
        self.C = np.array([[1, 2, 0],
                           [1, 1, 1],
                           [1, 5, 10]])
        self.Cs = scipy.sparse.csr_matrix(self.C)

        self.P_nonrev = np.array([[ 0.33333333,  0.66666667,  0.        ],
                                  [ 0.33333333,  0.33333333,  0.33333333],
                                  [ 0.0625    ,  0.3125    ,  0.625     ]])
        self.P_rev_ml = np.array([[ 0.33333333,  0.60290585,  0.06376081],
                                  [ 0.39709415,  0.33333333,  0.26957252],
                                  [ 0.05054485,  0.32445515,  0.625     ]])
        self.P_rev_pirev = np.array([[ 0.33333333,  0.61904762,  0.04761905],
                                     [ 0.36111111,  0.33333333,  0.30555556],
                                     [ 0.03125   ,  0.34375   ,  0.625     ]])

    def test_dense(self):
        # non-reversible
        P = transition_matrix(self.C)
        assert_allclose(P, self.P_nonrev)

        # reversible maximum likelihood
        P = transition_matrix(self.C, reversible=True)
        assert_allclose(P, self.P_rev_ml)
        P = transition_matrix(self.C, reversible=True, rev_pisym=False)
        assert_allclose(P, self.P_rev_ml)

        # reversible, pi symmetrization
        P = transition_matrix(self.C, reversible=True, rev_pisym=True)
        assert_allclose(P, self.P_rev_pirev)

    def test_sparse(self):
        # non-reversible
        P = transition_matrix(self.Cs).toarray()
        assert_allclose(P, self.P_nonrev)

        # non-rev return pi
        P, pi = transition_matrix(self.Cs, return_statdist=True)
        assert_allclose(P.T.dot(pi), pi)

        P, pi = transition_matrix(self.Cs, method='sparse', return_statdist=True)
        assert_allclose(P.T.dot(pi), pi)

        # primal-dual interior-point solver
        P, pi = transition_matrix(self.Cs, method='sparse', return_statdist=True, sparse_newton=True, reversible=True)
        assert_allclose(P.T.dot(pi), pi)
        P = transition_matrix(self.Cs, method='sparse', return_statdist=False, sparse_newton=True, reversible=True)

        # reversible maximum likelihood
        P = transition_matrix(self.Cs, reversible=True).toarray()
        assert_allclose(P, self.P_rev_ml)
        P = transition_matrix(self.Cs, reversible=True, rev_pisym=False).toarray()
        assert_allclose(P, self.P_rev_ml)

        P, pi = transition_matrix(self.Cs, reversible=True, rev_pisym=False, return_statdist=True, method='sparse')
        assert_allclose(P.T.dot(pi), pi)

        # reversible, pi symmetrization
        P = transition_matrix(self.Cs, reversible=True, rev_pisym=True).toarray()
        assert_allclose(P, self.P_rev_pirev)
        P, pi = transition_matrix(self.Cs, reversible=True, rev_pisym=True, return_statdist=True)
        assert_allclose(P.T.dot(pi), pi)
        P, pi = transition_matrix(self.Cs, reversible=True, rev_pisym=True, return_statdist=True, method='sparse')
        assert_allclose(P.T.dot(pi), pi)


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

        cov = tmatrix_cov(self.C, k=1)
        assert_allclose(cov, self.cov[1, :, :])


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

    def test_error_perturbation_sparse(self):
        Csparse = scipy.sparse.csr_matrix(self.C)

        with warnings.catch_warnings(record=True) as w:
            xn = error_perturbation(Csparse, self.S1)
            assert_allclose(xn, self.x)

            Xn = error_perturbation(Csparse, self.S2)
            assert_allclose(Xn, self.X)
