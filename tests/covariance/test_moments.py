import unittest
import numpy as np

__author__ = 'noe'

from deeptime.covariance import moments_XX, moments_XXXY, moments_block


class TestMoments(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X_2 = np.random.rand(10000, 2)
        cls.Y_2 = np.random.rand(10000, 2)
        # dense data
        cls.X_10 = np.random.rand(10000, 10)
        cls.Y_10 = np.random.rand(10000, 10)
        cls.X_100 = np.random.rand(10000, 100)
        cls.Y_100 = np.random.rand(10000, 100)
        # sparse zero data
        cls.X_10_sparsezero = np.zeros((10000, 10))
        cls.X_10_sparsezero[:, 0] = cls.X_10[:, 0]
        cls.Y_10_sparsezero = np.zeros((10000, 10))
        cls.Y_10_sparsezero[:, 0] = cls.Y_10[:, 0]
        cls.X_100_sparsezero = np.zeros((10000, 100))
        cls.X_100_sparsezero[:, :10] = cls.X_100[:, :10]
        cls.Y_100_sparsezero = np.zeros((10000, 100))
        cls.Y_100_sparsezero[:, :10] = cls.Y_100[:, :10]
        # sparse const data
        cls.X_10_sparseconst = np.ones((10000, 10))
        cls.X_10_sparseconst[:, 0] = cls.X_10[:, 0]
        cls.Y_10_sparseconst = 2 * np.ones((10000, 10))
        cls.Y_10_sparseconst[:, 0] = cls.Y_10[:, 0]
        cls.X_100_sparseconst = np.ones((10000, 100))
        cls.X_100_sparseconst[:, :10] = cls.X_100[:, :10]
        cls.Y_100_sparseconst = 2 * np.ones((10000, 100))
        cls.Y_100_sparseconst[:, :10] = cls.Y_100[:, :10]
        # boolean data
        cls.Xb_2 = np.random.randint(0, 2, size=(10000, 2))
        cls.Xb_2 = cls.Xb_2.astype(np.bool_)
        cls.Xb_10 = np.random.randint(0, 2, size=(10000, 10))
        cls.Xb_10 = cls.Xb_10.astype(np.bool_)
        cls.Xb_10_sparsezero = np.zeros((10000, 10), dtype=np.bool_)
        cls.Xb_10_sparsezero[:, 0] = cls.Xb_10[:, 0]
        # generate weights:
        cls.weights = np.random.rand(10000)
        # Set the lag time for time-lagged tests:
        cls.lag = 50
        # column subsets
        cls.cols_2 = np.array([0])
        cls.cols_10 = np.random.choice(10, 5, replace=False)
        cls.cols_100 = np.random.choice(100, 20, replace=False)
        # sparse tolerance
        cls.sparse_tol = 1e-14

        return cls

    def _test_moments_X(self, X, column_selection, remove_mean=False, sparse_mode='auto', sparse_tol=0.0, weights=None):
        # proposed solution
        w, s_X, C_XX = moments_XX(X, remove_mean=remove_mean, modify_data=False, sparse_mode=sparse_mode,
                                  sparse_tol=sparse_tol,
                                  weights=weights)
        # reference
        X = X.astype(np.float64)
        if weights is not None:
            X1 = weights[:, None] * X
            w = weights.sum()
        else:
            X1 = X
            w = X.shape[0]
        s_X_ref = X1.sum(axis=0)
        if remove_mean:
            X = X - (1.0 / w) * s_X_ref
        if weights is not None:
            X1 = weights[:, None] * X
        else:
            X1 = X
        C_XX_ref = np.dot(X1.T, X)
        # test
        assert np.allclose(s_X, s_X_ref)
        assert np.allclose(C_XX, C_XX_ref)
        # column subsets
        w, s_X, C_XX = moments_XX(X, remove_mean=remove_mean, modify_data=False,
                                  sparse_mode=sparse_mode, sparse_tol=sparse_tol,
                                  weights=weights, column_selection=column_selection)
        assert np.allclose(C_XX, C_XX_ref[:, column_selection])
        # diagonal only
        if sparse_mode != 'sparse':
            w, s_X, C_XX = moments_XX(X, remove_mean=remove_mean, modify_data=False,
                                      sparse_mode=sparse_mode, weights=weights, diag_only=True)
            assert np.allclose(C_XX, np.diag(C_XX_ref))

    def test_moments_X(self):
        # simple test, dense
        self._test_moments_X(self.X_10, self.cols_10, remove_mean=False, sparse_mode='dense')
        self._test_moments_X(self.X_100, self.cols_100, remove_mean=False, sparse_mode='dense')
        # mean-free, dense
        self._test_moments_X(self.X_10, self.cols_10, remove_mean=True, sparse_mode='dense')
        self._test_moments_X(self.X_100, self.cols_100, remove_mean=True, sparse_mode='dense')
        # weighted test, simple, dense:
        self._test_moments_X(self.X_10, self.cols_10, remove_mean=False, sparse_mode='dense', weights=self.weights)
        self._test_moments_X(self.X_100, self.cols_100, remove_mean=False, sparse_mode='dense', weights=self.weights)
        # weighted test, mean-free, dense:
        self._test_moments_X(self.X_10, self.cols_10, remove_mean=True, sparse_mode='dense', weights=self.weights)
        self._test_moments_X(self.X_100, self.cols_100, remove_mean=True, sparse_mode='dense', weights=self.weights)

    def test_moments_X_sparsezero(self):
        # simple test, sparse
        self._test_moments_X(self.X_10_sparsezero, self.cols_10, remove_mean=False, sparse_mode='sparse',
                             sparse_tol=self.sparse_tol)
        self._test_moments_X(self.X_100_sparsezero, self.cols_100, remove_mean=False, sparse_mode='sparse',
                             sparse_tol=self.sparse_tol)
        # mean-free, sparse
        self._test_moments_X(self.X_10_sparsezero, self.cols_10, remove_mean=True, sparse_mode='sparse',
                             sparse_tol=self.sparse_tol)
        self._test_moments_X(self.X_100_sparsezero, self.cols_100, remove_mean=True, sparse_mode='sparse',
                             sparse_tol=self.sparse_tol)
        # weighted, sparse
        self._test_moments_X(self.X_10_sparsezero, self.cols_10, remove_mean=False, sparse_mode='sparse',
                             sparse_tol=self.sparse_tol, weights=self.weights)
        self._test_moments_X(self.X_100_sparsezero, self.cols_100, remove_mean=False, sparse_mode='sparse',
                             sparse_tol=self.sparse_tol, weights=self.weights)
        # weighted, mean-free, sparse
        self._test_moments_X(self.X_10_sparsezero, self.cols_10, remove_mean=True, sparse_mode='sparse',
                             sparse_tol=self.sparse_tol, weights=self.weights)
        self._test_moments_X(self.X_100_sparsezero, self.cols_100, remove_mean=True, sparse_mode='sparse',
                             sparse_tol=self.sparse_tol, weights=self.weights)

    def test_moments_X_sparseconst(self):
        # simple test, sparse
        self._test_moments_X(self.X_10_sparseconst, self.cols_10, remove_mean=False, sparse_mode='sparse',
                             sparse_tol=self.sparse_tol)
        self._test_moments_X(self.X_100_sparseconst, self.cols_100, remove_mean=False, sparse_mode='sparse',
                             sparse_tol=self.sparse_tol)
        # mean-free, sparse
        self._test_moments_X(self.X_10_sparseconst, self.cols_10, remove_mean=True, sparse_mode='sparse',
                             sparse_tol=self.sparse_tol)
        self._test_moments_X(self.X_100_sparseconst, self.cols_100, remove_mean=True, sparse_mode='sparse',
                             sparse_tol=self.sparse_tol)
        # weighted, sparse:
        self._test_moments_X(self.X_10_sparseconst, self.cols_10, remove_mean=False, sparse_mode='sparse',
                             sparse_tol=self.sparse_tol, weights=self.weights)
        self._test_moments_X(self.X_100_sparseconst, self.cols_100, remove_mean=False, sparse_mode='sparse',
                             sparse_tol=self.sparse_tol, weights=self.weights)
        # weighted, mean-free, sparse:
        self._test_moments_X(self.X_10_sparseconst, self.cols_10, remove_mean=True, sparse_mode='sparse',
                             sparse_tol=self.sparse_tol, weights=self.weights)
        self._test_moments_X(self.X_100_sparseconst, self.cols_100, remove_mean=True, sparse_mode='sparse',
                             sparse_tol=self.sparse_tol, weights=self.weights)

    def test_boolean_moments(self):
        # standard tests
        self._test_moments_X(self.Xb_10, self.cols_10, remove_mean=False, sparse_mode='dense')
        self._test_moments_X(self.Xb_10, self.cols_10, remove_mean=True, sparse_mode='dense')
        self._test_moments_X(self.Xb_10_sparsezero, self.cols_10, remove_mean=False, sparse_mode='sparse',
                             sparse_tol=self.sparse_tol)
        self._test_moments_X(self.Xb_10_sparsezero, self.cols_10, remove_mean=True, sparse_mode='sparse',
                             sparse_tol=self.sparse_tol)
        # test integer recovery
        Cxx_ref = np.dot(self.Xb_10.astype(np.int64).T, self.Xb_10.astype(np.int64))  # integer
        s_X_ref = np.sum(self.Xb_10, axis=0)
        w, s_X, Cxx = moments_XX(self.Xb_10, remove_mean=False, modify_data=False, sparse_mode='dense')
        s_X = np.round(s_X).astype(np.int64)
        Cxx = np.round(Cxx).astype(np.int64)
        assert np.array_equal(s_X, s_X_ref)
        assert np.array_equal(Cxx, Cxx_ref)

    def _test_moments_XY(self, X, Y, column_selection, symmetrize=False, remove_mean=False, sparse_mode='auto',
                         sparse_tol=0.0, weights=None):
        w1, s_X, s_Y, C_XX, C_XY = moments_XXXY(X, Y, remove_mean=remove_mean, modify_data=False,
                                                symmetrize=symmetrize, sparse_mode=sparse_mode,
                                                sparse_tol=sparse_tol, weights=weights)
        # reference
        T = X.shape[0]
        if weights is not None:
            X1 = weights[:, None] * X
            Y1 = weights[:, None] * Y
        else:
            X1 = X
            Y1 = Y
        s_X_ref = X1.sum(axis=0)
        s_Y_ref = Y1.sum(axis=0)
        if symmetrize:
            s_X_ref = s_X_ref + s_Y_ref
            s_Y_ref = s_X_ref
            if weights is not None:
                w = 2 * np.sum(weights)
            else:
                w = 2 * T
        else:
            if weights is not None:
                w = np.sum(weights)
            else:
                w = T
        if remove_mean:
            X = X - s_X_ref / float(w)
            Y = Y - s_Y_ref / float(w)
        if weights is not None:
            X1 = weights[:, None] * X
            Y1 = weights[:, None] * Y
        else:
            X1 = X
            Y1 = Y
        if symmetrize:
            C_XX_ref = np.dot(X1.T, X) + np.dot(Y1.T, Y)
            C_XY_ref = np.dot(X1.T, Y) + np.dot(Y1.T, X)
        else:
            C_XX_ref = np.dot(X1.T, X)
            C_XY_ref = np.dot(X1.T, Y)
        # test
        assert np.allclose(w1, w)
        assert np.allclose(s_X, s_X_ref)
        assert np.allclose(s_Y, s_Y_ref)
        assert np.allclose(C_XX, C_XX_ref)
        assert np.allclose(C_XY, C_XY_ref)
        # column subsets
        w1, s_X, s_Y, C_XX, C_XY = moments_XXXY(X, Y, remove_mean=remove_mean, modify_data=False,
                                                symmetrize=symmetrize, sparse_mode=sparse_mode,
                                                sparse_tol=sparse_tol,
                                                weights=weights, column_selection=column_selection)
        assert np.allclose(C_XX, C_XX_ref[:, column_selection])
        assert np.allclose(C_XY, C_XY_ref[:, column_selection])
        # diagonal only
        if sparse_mode != 'sparse' and X.shape[1] == Y.shape[1]:
            w1, s_X, s_Y, C_XX, C_XY = moments_XXXY(X, Y, remove_mean=remove_mean, modify_data=False,
                                                    symmetrize=symmetrize, sparse_mode=sparse_mode,
                                                    sparse_tol=sparse_tol,
                                                    weights=weights, diag_only=True)
            assert np.allclose(C_XX, np.diag(C_XX_ref))
            assert np.allclose(C_XY, np.diag(C_XY_ref))

    def test_moments_XY(self):
        # simple test, dense
        self._test_moments_XY(self.X_10, self.Y_10, self.cols_10, symmetrize=False, remove_mean=False,
                              sparse_mode='dense')
        self._test_moments_XY(self.X_100, self.Y_10, self.cols_10, symmetrize=False, remove_mean=False,
                              sparse_mode='dense')
        self._test_moments_XY(self.X_100, self.Y_100, self.cols_100, symmetrize=False, remove_mean=False,
                              sparse_mode='dense')
        # mean-free, dense
        self._test_moments_XY(self.X_10, self.Y_10, self.cols_10, symmetrize=False, remove_mean=True,
                              sparse_mode='dense')
        self._test_moments_XY(self.X_100, self.Y_10, self.cols_10, symmetrize=False, remove_mean=True,
                              sparse_mode='dense')
        self._test_moments_XY(self.X_100, self.Y_100, self.cols_100, symmetrize=False, remove_mean=True,
                              sparse_mode='dense')

    def test_moments_XY_weighted(self):
        # weighted test, dense
        self._test_moments_XY(self.X_10, self.Y_10, self.cols_10, symmetrize=False, remove_mean=False,
                              sparse_mode='dense', weights=self.weights)
        self._test_moments_XY(self.X_100, self.Y_100, self.cols_100, symmetrize=False, remove_mean=False,
                              sparse_mode='dense', weights=self.weights)
        # weighted test, mean-free, dense
        self._test_moments_XY(self.X_10, self.Y_10, self.cols_10, symmetrize=False, remove_mean=True,
                              sparse_mode='dense', weights=self.weights)
        self._test_moments_XY(self.X_100, self.Y_100, self.cols_100, symmetrize=False, remove_mean=True,
                              sparse_mode='dense', weights=self.weights)

    def test_moments_XY_sym(self):
        # simple test, dense, symmetric
        self._test_moments_XY(self.X_2, self.Y_2, self.cols_2, symmetrize=True, remove_mean=False, sparse_mode='dense')
        self._test_moments_XY(self.X_10, self.Y_10, self.cols_10, symmetrize=True, remove_mean=False,
                              sparse_mode='dense')
        self._test_moments_XY(self.X_100, self.Y_100, self.cols_100, symmetrize=True, remove_mean=False,
                              sparse_mode='dense')
        # mean-free, dense, symmetric
        self._test_moments_XY(self.X_2, self.Y_2, self.cols_2, symmetrize=True, remove_mean=True, sparse_mode='dense')
        self._test_moments_XY(self.X_10, self.Y_10, self.cols_10, symmetrize=True, remove_mean=True,
                              sparse_mode='dense')
        self._test_moments_XY(self.X_100, self.Y_100, self.cols_100, symmetrize=True, remove_mean=True,
                              sparse_mode='dense')

    def test_moments_XY_weighted_sym(self):
        # simple test, dense, symmetric
        self._test_moments_XY(self.X_2, self.Y_2, self.cols_2, symmetrize=True, remove_mean=False, sparse_mode='dense',
                              weights=self.weights)
        self._test_moments_XY(self.X_10, self.Y_10, self.cols_10, symmetrize=True, remove_mean=False,
                              sparse_mode='dense'
                              , weights=self.weights)
        self._test_moments_XY(self.X_100, self.Y_100, self.cols_100, symmetrize=True, remove_mean=False,
                              sparse_mode='dense',
                              weights=self.weights)
        # mean-free, dense, symmetric
        self._test_moments_XY(self.X_2, self.Y_2, self.cols_2, symmetrize=True, remove_mean=True, sparse_mode='dense',
                              weights=self.weights)
        self._test_moments_XY(self.X_10, self.Y_10, self.cols_10, symmetrize=True, remove_mean=True,
                              sparse_mode='dense',
                              weights=self.weights)
        self._test_moments_XY(self.X_100, self.Y_100, self.cols_100, symmetrize=True, remove_mean=True,
                              sparse_mode='dense',
                              weights=self.weights)

    def test_moments_XY_sparsezero(self):
        # simple test, sparse
        self._test_moments_XY(self.X_10_sparsezero, self.Y_10_sparsezero, self.cols_10, symmetrize=False,
                              remove_mean=False,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol)
        self._test_moments_XY(self.X_100_sparsezero, self.Y_10_sparsezero, self.cols_10, symmetrize=False,
                              remove_mean=False,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol)
        self._test_moments_XY(self.X_100_sparsezero, self.Y_100_sparsezero, self.cols_100, symmetrize=False,
                              remove_mean=False,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol)
        # mean-free, sparse
        self._test_moments_XY(self.X_10_sparsezero, self.Y_10_sparsezero, self.cols_10, symmetrize=False,
                              remove_mean=True,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol)
        self._test_moments_XY(self.X_100_sparsezero, self.Y_10_sparsezero, self.cols_10, symmetrize=False,
                              remove_mean=True,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol)
        self._test_moments_XY(self.X_100_sparsezero, self.Y_100_sparsezero, self.cols_100, symmetrize=False,
                              remove_mean=True,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol)

    def test_moments_XY_weighted_sparsezero(self):
        # weighted test, sparse
        self._test_moments_XY(self.X_10_sparsezero, self.Y_10_sparsezero, self.cols_10, symmetrize=False,
                              remove_mean=False,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol, weights=self.weights)
        self._test_moments_XY(self.X_100_sparsezero, self.Y_100_sparsezero, self.cols_100, symmetrize=False,
                              remove_mean=False,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol, weights=self.weights)
        # weighted test, mean-free, sparse
        self._test_moments_XY(self.X_10_sparsezero, self.Y_10_sparsezero, self.cols_10, symmetrize=False,
                              remove_mean=True,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol, weights=self.weights)
        self._test_moments_XY(self.X_100_sparsezero, self.Y_100_sparsezero, self.cols_100, symmetrize=False,
                              remove_mean=True,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol, weights=self.weights)

    def test_moments_XY_sym_sparsezero(self):
        # simple test, sparse, symmetric
        self._test_moments_XY(self.X_10_sparsezero, self.Y_10_sparsezero, self.cols_10, symmetrize=True,
                              remove_mean=False,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol)
        self._test_moments_XY(self.X_100_sparsezero, self.Y_100_sparsezero, self.cols_100, symmetrize=True,
                              remove_mean=False,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol)
        # mean-free, sparse, symmetric
        self._test_moments_XY(self.X_10_sparsezero, self.Y_10_sparsezero, self.cols_10, symmetrize=True,
                              remove_mean=True,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol)
        self._test_moments_XY(self.X_100_sparsezero, self.Y_100_sparsezero, self.cols_100, symmetrize=True,
                              remove_mean=True,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol)

    def test_moments_XY_weighted_sym_sparsezero(self):
        # simple test, sparse, symmetric
        self._test_moments_XY(self.X_10_sparsezero, self.Y_10_sparsezero, self.cols_10, symmetrize=True,
                              remove_mean=False,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol, weights=self.weights)
        self._test_moments_XY(self.X_100_sparsezero, self.Y_100_sparsezero, self.cols_100, symmetrize=True,
                              remove_mean=False,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol, weights=self.weights)
        # mean-free, sparse, symmetric
        self._test_moments_XY(self.X_10_sparsezero, self.Y_10_sparsezero, self.cols_10, symmetrize=True,
                              remove_mean=True,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol, weights=self.weights)
        self._test_moments_XY(self.X_100_sparsezero, self.Y_100_sparsezero, self.cols_100, symmetrize=True,
                              remove_mean=True,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol, weights=self.weights)

    def test_moments_XY_sparseconst(self):
        # simple test, sparse
        self._test_moments_XY(self.X_10_sparseconst, self.Y_10_sparseconst, self.cols_10, symmetrize=False,
                              remove_mean=False,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol)
        self._test_moments_XY(self.X_100_sparseconst, self.Y_10_sparseconst, self.cols_10, symmetrize=False,
                              remove_mean=False,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol)
        self._test_moments_XY(self.X_100_sparseconst, self.Y_100_sparseconst, self.cols_100, symmetrize=False,
                              remove_mean=False,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol)
        # mean-free, sparse
        self._test_moments_XY(self.X_10_sparseconst, self.Y_10_sparseconst, self.cols_10, symmetrize=False,
                              remove_mean=True,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol)
        self._test_moments_XY(self.X_100_sparseconst, self.Y_10_sparseconst, self.cols_10, symmetrize=False,
                              remove_mean=True,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol)
        self._test_moments_XY(self.X_100_sparseconst, self.Y_100_sparseconst, self.cols_100, symmetrize=False,
                              remove_mean=True,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol)

    def test_moments_XY_weighted_sparseconst(self):
        # weighted test, sparse
        self._test_moments_XY(self.X_10_sparseconst, self.Y_10_sparseconst, self.cols_10, symmetrize=False,
                              remove_mean=False,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol, weights=self.weights)
        self._test_moments_XY(self.X_100_sparseconst, self.Y_100_sparseconst, self.cols_100, symmetrize=False,
                              remove_mean=False,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol, weights=self.weights)
        # weighted test, mean-free, sparse
        self._test_moments_XY(self.X_10_sparseconst, self.Y_10_sparseconst, self.cols_10, symmetrize=False,
                              remove_mean=True,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol, weights=self.weights)
        self._test_moments_XY(self.X_100_sparseconst, self.Y_100_sparseconst, self.cols_100, symmetrize=False,
                              remove_mean=True,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol, weights=self.weights)

    def test_moments_XY_sym_sparseconst(self):
        # simple test, sparse, symmetric
        self._test_moments_XY(self.X_10_sparseconst, self.Y_10_sparseconst, self.cols_10, symmetrize=True,
                              remove_mean=False,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol)
        self._test_moments_XY(self.X_100_sparseconst, self.Y_100_sparseconst, self.cols_100, symmetrize=True,
                              remove_mean=False,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol)
        # mean-free, sparse, symmetric
        self._test_moments_XY(self.X_10_sparseconst, self.Y_10_sparseconst, self.cols_10, symmetrize=True,
                              remove_mean=True,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol)
        self._test_moments_XY(self.X_100_sparseconst, self.Y_100_sparseconst, self.cols_100, symmetrize=True,
                              remove_mean=True,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol)

    def test_moments_XY_weighted_sym_sparseconst(self):
        # simple test, sparse, symmetric
        self._test_moments_XY(self.X_10_sparseconst, self.Y_10_sparseconst, self.cols_10, symmetrize=True,
                              remove_mean=False,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol, weights=self.weights)
        self._test_moments_XY(self.X_100_sparseconst, self.Y_100_sparseconst, self.cols_100, symmetrize=True,
                              remove_mean=False,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol, weights=self.weights)
        # mean-free, sparse, symmetric
        self._test_moments_XY(self.X_10_sparseconst, self.Y_10_sparseconst, self.cols_10, symmetrize=True,
                              remove_mean=True,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol, weights=self.weights)
        self._test_moments_XY(self.X_100_sparseconst, self.Y_100_sparseconst, self.cols_100, symmetrize=True,
                              remove_mean=True,
                              sparse_mode='sparse', sparse_tol=self.sparse_tol, weights=self.weights)

    def _test_moments_block(self, X, Y, column_selection, remove_mean=False, sparse_mode='auto', sparse_tol=0.0):
        w1, s, C = moments_block(X, Y, remove_mean=remove_mean, modify_data=False,
                                 sparse_mode=sparse_mode, sparse_tol=sparse_tol)
        # reference
        w = X.shape[0]
        s_X_ref = X.sum(axis=0)
        s_Y_ref = Y.sum(axis=0)
        if remove_mean:
            X = X - s_X_ref / float(w)
            Y = Y - s_Y_ref / float(w)
        C_XX_ref = np.dot(X.T, X)
        C_XY_ref = np.dot(X.T, Y)
        C_YX_ref = np.dot(Y.T, X)
        C_YY_ref = np.dot(Y.T, Y)
        # test
        assert np.allclose(w1, w)
        assert np.allclose(s[0], s_X_ref)
        assert np.allclose(s[1], s_Y_ref)
        assert np.allclose(C[0][0], C_XX_ref)
        assert np.allclose(C[0][1], C_XY_ref)
        assert np.allclose(C[1][0], C_YX_ref)
        assert np.allclose(C[1][1], C_YY_ref)
        # column subsets
        w1, s, C = moments_block(X, Y, remove_mean=remove_mean, modify_data=False,
                                         sparse_mode=sparse_mode, sparse_tol=sparse_tol,
                                         column_selection=column_selection)
        assert np.allclose(C[0][0], C_XX_ref[:, column_selection])
        assert np.allclose(C[0][1], C_XY_ref[:, column_selection])
        assert np.allclose(C[1][0], C_YX_ref[:, column_selection])
        assert np.allclose(C[1][1], C_YY_ref[:, column_selection])
        # diagonal only
        if sparse_mode != 'sparse' and X.shape[1] == Y.shape[1]:
            w1, s, C = moments_block(X, Y, remove_mean=remove_mean, modify_data=False,
                                             sparse_mode=sparse_mode, sparse_tol=sparse_tol, diag_only=True)
            assert np.allclose(C[0][0], np.diag(C_XX_ref))
            assert np.allclose(C[0][1], np.diag(C_XY_ref))
            assert np.allclose(C[1][0], np.diag(C_YX_ref))
            assert np.allclose(C[1][1], np.diag(C_YY_ref))

    def test_moments_block(self):
        # simple test, dense
        self._test_moments_block(self.X_10, self.Y_10, self.cols_10, remove_mean=False, sparse_mode='dense')
        self._test_moments_block(self.X_100, self.Y_100, self.cols_100, remove_mean=False, sparse_mode='dense')
        # mean-free, dense
        self._test_moments_block(self.X_10, self.Y_10, self.cols_10, remove_mean=True, sparse_mode='dense')
        self._test_moments_block(self.X_100, self.Y_100, self.cols_100, remove_mean=True, sparse_mode='dense')

    def test_moments_block_sparsezero(self):
        # simple test, sparse
        self._test_moments_block(self.X_10_sparsezero, self.Y_10_sparsezero, self.cols_10, remove_mean=False,
                                 sparse_mode='sparse', sparse_tol=self.sparse_tol)
        self._test_moments_block(self.X_100_sparsezero, self.Y_100_sparsezero, self.cols_100, remove_mean=False,
                                 sparse_mode='sparse', sparse_tol=self.sparse_tol)
        # mean-free, sparse
        self._test_moments_block(self.X_10_sparsezero, self.Y_10_sparsezero, self.cols_10, remove_mean=True,
                                 sparse_mode='sparse', sparse_tol=self.sparse_tol)
        self._test_moments_block(self.X_100_sparsezero, self.Y_100_sparsezero, self.cols_100, remove_mean=True,
                                 sparse_mode='sparse', sparse_tol=self.sparse_tol)

    def test_moments_block_sparseconst(self):
        # simple test, sparse
        self._test_moments_block(self.X_10_sparseconst, self.Y_10_sparseconst, self.cols_10, remove_mean=False,
                                 sparse_mode='sparse', sparse_tol=self.sparse_tol)
        self._test_moments_block(self.X_100_sparseconst, self.Y_100_sparseconst, self.cols_100, remove_mean=False,
                                 sparse_mode='sparse', sparse_tol=self.sparse_tol)
        # mean-free, sparse
        self._test_moments_block(self.X_10_sparseconst, self.Y_10_sparseconst, self.cols_10, remove_mean=True,
                                 sparse_mode='sparse', sparse_tol=self.sparse_tol)
        self._test_moments_block(self.X_100_sparseconst, self.Y_100_sparseconst, self.cols_100, remove_mean=True,
                                 sparse_mode='sparse', sparse_tol=self.sparse_tol)
