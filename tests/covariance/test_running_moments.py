import unittest
import numpy as np

__author__ = 'noe'

from deeptime.covariance.util import RunningCovar


class TestRunningMoments(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X = np.random.rand(10000, 2)
        cls.Y = np.random.rand(10000, 2)
        cls.T = cls.X.shape[0]
        # Chunk size:
        cls.L = 1000
        # Number of chunks:
        cls.nchunks = cls.T // cls.L
        # Set a lag time for time-lagged tests:
        # cls.lag = 50
        # Weights references:
        cls.weights = np.random.rand(10000)
        # Trajectory weights:
        cls.trajweights = 3 * np.random.rand(cls.nchunks)
        # bias the first part
        cls.X[:2000] += 1.0
        cls.Y[:2000] -= 1.0
        # column subsets
        cls.cols_2 = np.array([0])

        # direct calculation, moments of X and Y
        cls.w = np.shape(cls.X)[0]
        cls.wsym = 2 * np.shape(cls.X)[0]
        cls.sx = cls.X.sum(axis=0)
        cls.sy = cls.Y.sum(axis=0)
        cls.Mxx = np.dot(cls.X.T, cls.X)
        cls.Mxy = np.dot(cls.X.T, cls.Y)
        cls.Myy = np.dot(cls.Y.T, cls.Y)
        cls.mx = cls.sx / float(cls.w)
        cls.my = cls.sy / float(cls.w)
        cls.X0 = cls.X - cls.mx
        cls.Y0 = cls.Y - cls.my
        cls.Mxx0 = np.dot(cls.X0.T, cls.X0)
        cls.Mxy0 = np.dot(cls.X0.T, cls.Y0)
        cls.Myy0 = np.dot(cls.Y0.T, cls.Y0)

        # direct calculation, symmetric moments
        cls.s_sym = cls.sx + cls.sy
        cls.Mxx_sym = np.dot(cls.X.T, cls.X) + np.dot(cls.Y.T, cls.Y)
        cls.Mxy_sym = np.dot(cls.X.T, cls.Y) + np.dot(cls.Y.T, cls.X)
        cls.m_sym = cls.s_sym / float(cls.wsym)
        cls.X0_sym = cls.X - cls.m_sym
        cls.Y0_sym = cls.Y - cls.m_sym
        cls.Mxx0_sym = np.dot(cls.X0_sym.T, cls.X0_sym) + np.dot(cls.Y0_sym.T, cls.Y0_sym)
        cls.Mxy0_sym = np.dot(cls.X0_sym.T, cls.Y0_sym) + np.dot(cls.Y0_sym.T, cls.X0_sym)

        # direct calculation, weighted moments:
        cls.wesum = np.sum(cls.weights)
        cls.sx_w = (cls.weights[:, None] * cls.X).sum(axis=0)
        cls.sy_w = (cls.weights[:, None] * cls.Y).sum(axis=0)
        cls.Mxx_w = np.dot((cls.weights[:, None] * cls.X).T, cls.X)
        cls.Mxy_w = np.dot((cls.weights[:, None] * cls.X).T, cls.Y)
        cls.mx_w = cls.sx_w / float(cls.wesum)
        cls.my_w = cls.sy_w / float(cls.wesum)
        cls.X0_w = cls.X - cls.mx_w
        cls.Y0_w = cls.Y - cls.my_w
        cls.Mxx0_w = np.dot((cls.weights[:, None] * cls.X0_w).T, cls.X0_w)
        cls.Mxy0_w = np.dot((cls.weights[:, None] * cls.X0_w).T, cls.Y0_w)

        # direct calculation, weighted symmetric moments
        cls.s_sym_w = cls.sx_w + cls.sy_w
        cls.Mxx_sym_w = np.dot((cls.weights[:, None] * cls.X).T, cls.X) + np.dot((cls.weights[:, None] * cls.Y).T,
                                                                                 cls.Y)
        cls.Mxy_sym_w = np.dot((cls.weights[:, None] * cls.X).T, cls.Y) + np.dot((cls.weights[:, None] * cls.Y).T,
                                                                                 cls.X)
        cls.m_sym_w = cls.s_sym_w / float(2 * cls.wesum)
        cls.X0_sym_w = cls.X - cls.m_sym_w
        cls.Y0_sym_w = cls.Y - cls.m_sym_w
        cls.Mxx0_sym_w = np.dot((cls.weights[:, None] * cls.X0_sym_w).T, cls.X0_sym_w) + np.dot(
            (cls.weights[:, None] * cls.Y0_sym_w).T, cls.Y0_sym_w)
        cls.Mxy0_sym_w = np.dot((cls.weights[:, None] * cls.X0_sym_w).T, cls.Y0_sym_w) + np.dot(
            (cls.weights[:, None] * cls.Y0_sym_w).T, cls.X0_sym_w)

        return cls

    def test_XX_withmean(self):
        # many passes
        cc = RunningCovar(remove_mean=False)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i + self.L])
        np.testing.assert_allclose(cc.weight_XX(), self.T)
        np.testing.assert_allclose(cc.sum_X(), self.sx)
        np.testing.assert_allclose(cc.moments_XX(), self.Mxx)
        cc = RunningCovar(remove_mean=False)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i + self.L], column_selection=self.cols_2)
        np.testing.assert_allclose(cc.moments_XX(), self.Mxx[:, self.cols_2])
        cc = RunningCovar(remove_mean=False, diag_only=True)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i + self.L])
        np.testing.assert_allclose(cc.moments_XX(), np.diag(self.Mxx))

    def test_XX_meanfree(self):
        # many passes
        cc = RunningCovar(remove_mean=True)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i + self.L])
        np.testing.assert_allclose(cc.weight_XX(), self.T)
        np.testing.assert_allclose(cc.sum_X(), self.sx)
        np.testing.assert_allclose(cc.moments_XX(), self.Mxx0)
        cc = RunningCovar(remove_mean=True)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i + self.L], column_selection=self.cols_2)
        np.testing.assert_allclose(cc.moments_XX(), self.Mxx0[:, self.cols_2])
        cc = RunningCovar(remove_mean=True, diag_only=True)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i + self.L])
        np.testing.assert_allclose(cc.moments_XX(), np.diag(self.Mxx0))

    def test_XXXY_withmean(self):
        # many passes
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i + self.L], self.Y[i:i + self.L])
        np.testing.assert_allclose(cc.weight_XY(), self.T)
        np.testing.assert_allclose(cc.sum_X(), self.sx)
        np.testing.assert_allclose(cc.moments_XX(), self.Mxx)
        np.testing.assert_allclose(cc.moments_XY(), self.Mxy)
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i + self.L], self.Y[i:i + self.L], column_selection=self.cols_2)
        np.testing.assert_allclose(cc.moments_XX(), self.Mxx[:, self.cols_2])
        np.testing.assert_allclose(cc.moments_XY(), self.Mxy[:, self.cols_2])
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False, diag_only=True)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i + self.L], self.Y[i:i + self.L])
        np.testing.assert_allclose(cc.moments_XX(), np.diag(self.Mxx))
        np.testing.assert_allclose(cc.moments_XY(), np.diag(self.Mxy))

    def test_XXXY_meanfree(self):
        # many passes
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i + self.L], self.Y[i:i + self.L])
        np.testing.assert_allclose(cc.weight_XY(), self.T)
        np.testing.assert_allclose(cc.sum_X(), self.sx)
        np.testing.assert_allclose(cc.moments_XX(), self.Mxx0)
        np.testing.assert_allclose(cc.moments_XY(), self.Mxy0)
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i + self.L], self.Y[i:i + self.L], column_selection=self.cols_2)
        np.testing.assert_allclose(cc.moments_XX(), self.Mxx0[:, self.cols_2])
        np.testing.assert_allclose(cc.moments_XY(), self.Mxy0[:, self.cols_2])
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True, diag_only=True)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i + self.L], self.Y[i:i + self.L])
        np.testing.assert_allclose(cc.moments_XX(), np.diag(self.Mxx0))
        np.testing.assert_allclose(cc.moments_XY(), np.diag(self.Mxy0))

    def test_XXXY_weighted_withmean(self):
        # many passes
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False)
        for i in range(0, self.T, self.L):
            iX = self.X[i:i + self.L, :]
            iY = self.Y[i:i + self.L, :]
            iwe = self.weights[i:i + self.L]
            cc.add(iX, iY, weights=iwe)
        np.testing.assert_allclose(cc.weight_XY(), self.wesum)
        np.testing.assert_allclose(cc.sum_X(), self.sx_w)
        np.testing.assert_allclose(cc.moments_XX(), self.Mxx_w)
        np.testing.assert_allclose(cc.moments_XY(), self.Mxy_w)
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False)
        for i in range(0, self.T, self.L):
            iX = self.X[i:i + self.L, :]
            iY = self.Y[i:i + self.L, :]
            iwe = self.weights[i:i + self.L]
            cc.add(iX, iY, weights=iwe, column_selection=self.cols_2)
        np.testing.assert_allclose(cc.moments_XX(), self.Mxx_w[:, self.cols_2])
        np.testing.assert_allclose(cc.moments_XY(), self.Mxy_w[:, self.cols_2])
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False, diag_only=True)
        for i in range(0, self.T, self.L):
            iX = self.X[i:i + self.L, :]
            iY = self.Y[i:i + self.L, :]
            iwe = self.weights[i:i + self.L]
            cc.add(iX, iY, weights=iwe)
        np.testing.assert_allclose(cc.moments_XX(), np.diag(self.Mxx_w))
        np.testing.assert_allclose(cc.moments_XY(), np.diag(self.Mxy_w))

    def test_XXXY_weighted_meanfree(self):
        # many passes
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True)
        for i in range(0, self.T, self.L):
            iX = self.X[i:i + self.L, :]
            iY = self.Y[i:i + self.L, :]
            iwe = self.weights[i:i + self.L]
            cc.add(iX, iY, weights=iwe)
        np.testing.assert_allclose(cc.weight_XY(), self.wesum)
        np.testing.assert_allclose(cc.sum_X(), self.sx_w)
        np.testing.assert_allclose(cc.moments_XX(), self.Mxx0_w)
        np.testing.assert_allclose(cc.moments_XY(), self.Mxy0_w)
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True)
        for i in range(0, self.T, self.L):
            iX = self.X[i:i + self.L, :]
            iY = self.Y[i:i + self.L, :]
            iwe = self.weights[i:i + self.L]
            cc.add(iX, iY, weights=iwe, column_selection=self.cols_2)
        np.testing.assert_allclose(cc.moments_XX(), self.Mxx0_w[:, self.cols_2])
        np.testing.assert_allclose(cc.moments_XY(), self.Mxy0_w[:, self.cols_2])
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True, diag_only=True)
        for i in range(0, self.T, self.L):
            iX = self.X[i:i + self.L, :]
            iY = self.Y[i:i + self.L, :]
            iwe = self.weights[i:i + self.L]
            cc.add(iX, iY, weights=iwe)
        np.testing.assert_allclose(cc.moments_XX(), np.diag(self.Mxx0_w))
        np.testing.assert_allclose(cc.moments_XY(), np.diag(self.Mxy0_w))

    def test_XXXY_sym_withmean(self):
        # many passes
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False, symmetrize=True)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i + self.L], self.Y[i:i + self.L])
        np.testing.assert_allclose(cc.weight_XY(), 2 * self.T)
        np.testing.assert_allclose(cc.sum_X(), self.s_sym)
        np.testing.assert_allclose(cc.moments_XX(), self.Mxx_sym)
        np.testing.assert_allclose(cc.moments_XY(), self.Mxy_sym)
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False, symmetrize=True)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i + self.L], self.Y[i:i + self.L], column_selection=self.cols_2)
        np.testing.assert_allclose(cc.moments_XX(), self.Mxx_sym[:, self.cols_2])
        np.testing.assert_allclose(cc.moments_XY(), self.Mxy_sym[:, self.cols_2])
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False, symmetrize=True, diag_only=True)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i + self.L], self.Y[i:i + self.L])
        np.testing.assert_allclose(cc.moments_XX(), np.diag(self.Mxx_sym))
        np.testing.assert_allclose(cc.moments_XY(), np.diag(self.Mxy_sym))

    def test_XXXY_sym_meanfree(self):
        # many passes
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True, symmetrize=True)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i + self.L], self.Y[i:i + self.L])
        np.testing.assert_allclose(cc.weight_XY(), 2 * self.T)
        np.testing.assert_allclose(cc.sum_X(), self.s_sym)
        np.testing.assert_allclose(cc.moments_XX(), self.Mxx0_sym)
        np.testing.assert_allclose(cc.moments_XY(), self.Mxy0_sym)
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True, symmetrize=True)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i + self.L], self.Y[i:i + self.L], column_selection=self.cols_2)
        np.testing.assert_allclose(cc.moments_XX(), self.Mxx0_sym[:, self.cols_2])
        np.testing.assert_allclose(cc.moments_XY(), self.Mxy0_sym[:, self.cols_2])
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True, symmetrize=True, diag_only=True)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i + self.L], self.Y[i:i + self.L])
        np.testing.assert_allclose(cc.moments_XX(), np.diag(self.Mxx0_sym))
        np.testing.assert_allclose(cc.moments_XY(), np.diag(self.Mxy0_sym))

    def test_XXXY_weighted_sym_withmean(self):
        # many passes
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False, symmetrize=True)
        for i in range(0, self.T, self.L):
            iwe = self.weights[i:i + self.L]
            cc.add(self.X[i:i + self.L], self.Y[i:i + self.L], weights=iwe)
        np.testing.assert_allclose(cc.weight_XY(), 2 * self.wesum)
        np.testing.assert_allclose(cc.sum_X(), self.s_sym_w)
        np.testing.assert_allclose(cc.moments_XX(), self.Mxx_sym_w)
        np.testing.assert_allclose(cc.moments_XY(), self.Mxy_sym_w)
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False, symmetrize=True)
        for i in range(0, self.T, self.L):
            iwe = self.weights[i:i + self.L]
            cc.add(self.X[i:i + self.L], self.Y[i:i + self.L], weights=iwe, column_selection=self.cols_2)
        np.testing.assert_allclose(cc.moments_XX(), self.Mxx_sym_w[:, self.cols_2])
        np.testing.assert_allclose(cc.moments_XY(), self.Mxy_sym_w[:, self.cols_2])
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False, symmetrize=True, diag_only=True)
        for i in range(0, self.T, self.L):
            iwe = self.weights[i:i + self.L]
            cc.add(self.X[i:i + self.L], self.Y[i:i + self.L], weights=iwe)
        np.testing.assert_allclose(cc.moments_XX(), np.diag(self.Mxx_sym_w))
        np.testing.assert_allclose(cc.moments_XY(), np.diag(self.Mxy_sym_w))

    def test_XXXY_weighted_sym_meanfree(self):
        # many passes
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True, symmetrize=True)
        for i in range(0, self.T, self.L):
            iwe = self.weights[i:i + self.L]
            cc.add(self.X[i:i + self.L], self.Y[i:i + self.L], weights=iwe)
        np.testing.assert_allclose(cc.weight_XY(), 2 * self.wesum)
        np.testing.assert_allclose(cc.sum_X(), self.s_sym_w)
        np.testing.assert_allclose(cc.moments_XX(), self.Mxx0_sym_w)
        np.testing.assert_allclose(cc.moments_XY(), self.Mxy0_sym_w)
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True, symmetrize=True)
        for i in range(0, self.T, self.L):
            iwe = self.weights[i:i + self.L]
            cc.add(self.X[i:i + self.L], self.Y[i:i + self.L], weights=iwe, column_selection=self.cols_2)
        np.testing.assert_allclose(cc.moments_XX(), self.Mxx0_sym_w[:, self.cols_2])
        np.testing.assert_allclose(cc.moments_XY(), self.Mxy0_sym_w[:, self.cols_2])
        cc = RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True, symmetrize=True, diag_only=True)
        for i in range(0, self.T, self.L):
            iwe = self.weights[i:i + self.L]
            cc.add(self.X[i:i + self.L], self.Y[i:i + self.L], weights=iwe)
        np.testing.assert_allclose(cc.moments_XX(), np.diag(self.Mxx0_sym_w))
        np.testing.assert_allclose(cc.moments_XY(), np.diag(self.Mxy0_sym_w))

    def test_XXYY_meanfree(self):
        # many passes
        cc = RunningCovar(compute_XX=True, compute_XY=True, compute_YY=True, remove_mean=True)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i + self.L], self.Y[i:i + self.L])
        np.testing.assert_allclose(cc.weight_XY(), self.T)
        np.testing.assert_allclose(cc.sum_X(), self.sx)
        np.testing.assert_allclose(cc.moments_XX(), self.Mxx0)
        np.testing.assert_allclose(cc.moments_XY(), self.Mxy0)
        np.testing.assert_allclose(cc.moments_YY(), self.Myy0)
        cc = RunningCovar(compute_XX=True, compute_XY=True, compute_YY=True, remove_mean=True)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i + self.L], self.Y[i:i + self.L], column_selection=self.cols_2)
        np.testing.assert_allclose(cc.moments_XX(), self.Mxx0[:, self.cols_2])
        np.testing.assert_allclose(cc.moments_XY(), self.Mxy0[:, self.cols_2])
        np.testing.assert_allclose(cc.moments_YY(), self.Myy0[:, self.cols_2])
        cc = RunningCovar(compute_XX=True, compute_XY=True, compute_YY=True, remove_mean=True,
                          diag_only=True)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i + self.L], self.Y[i:i + self.L])
        np.testing.assert_allclose(cc.moments_XX(), np.diag(self.Mxx0))
        np.testing.assert_allclose(cc.moments_XY(), np.diag(self.Mxy0))
        np.testing.assert_allclose(cc.moments_YY(), np.diag(self.Myy0))
