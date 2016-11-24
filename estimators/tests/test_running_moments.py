from __future__ import absolute_import
import unittest
import numpy as np
from .. import running_moments

__author__ = 'noe'


class TestRunningMoments(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X = np.random.rand(10000, 2)
        cls.Y = np.random.rand(10000, 2)
        cls.T = cls.X.shape[0]
        # Chunk size:
        cls.L = 1000
        # Number of chunks:
        cls.nchunks = cls.T / cls.L
        # Set a lag time for time-lagged tests:
        #cls.lag = 50
        # Weights references:
        cls.weights = np.random.rand(10000)
        # Trajectory weights:
        cls.trajweights = 3*np.random.rand(cls.nchunks)
        # bias the first part
        cls.X[:2000] += 1.0
        cls.Y[:2000] -= 1.0
        # direct calculation, moments of X and Y
        cls.w = np.shape(cls.X)[0]
        cls.wsym = 2*np.shape(cls.X)[0]
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
        cls.Mxx_sym_w = np.dot((cls.weights[:, None] * cls.X).T, cls.X) + np.dot((cls.weights[:, None] * cls.Y).T, cls.Y)
        cls.Mxy_sym_w = np.dot((cls.weights[:, None] * cls.X).T, cls.Y) + np.dot((cls.weights[:, None] * cls.Y).T, cls.X)
        cls.m_sym_w = cls.s_sym_w / float(2 * cls.wesum)
        cls.X0_sym_w = cls.X - cls.m_sym_w
        cls.Y0_sym_w = cls.Y - cls.m_sym_w
        cls.Mxx0_sym_w = np.dot((cls.weights[:, None] *cls.X0_sym_w).T, cls.X0_sym_w) + np.dot((cls.weights[:, None] *cls.Y0_sym_w).T, cls.Y0_sym_w)
        cls.Mxy0_sym_w = np.dot((cls.weights[:, None] *cls.X0_sym_w).T, cls.Y0_sym_w) + np.dot((cls.weights[:, None] *cls.Y0_sym_w).T, cls.X0_sym_w)

        return cls

    def test_XX_withmean(self):
        # many passes
        cc = running_moments.RunningCovar(remove_mean=False)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i+self.L])
        assert np.allclose(cc.weight_XX(), self.T)
        assert np.allclose(cc.sum_X(), self.sx)
        assert np.allclose(cc.moments_XX(), self.Mxx)

    def test_XX_meanfree(self):
        # many passes
        cc = running_moments.RunningCovar(remove_mean=True)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i+self.L])
        assert np.allclose(cc.weight_XX(), self.T)
        assert np.allclose(cc.sum_X(), self.sx)
        assert np.allclose(cc.moments_XX(), self.Mxx0)

    def test_XXXY_withmean(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i+self.L], self.Y[i:i+self.L])
        assert np.allclose(cc.weight_XY(), self.T)
        assert np.allclose(cc.sum_X(), self.sx)
        assert np.allclose(cc.moments_XX(), self.Mxx)
        assert np.allclose(cc.moments_XY(), self.Mxy)

    def test_XXXY_meanfree(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True)
        L = 1000
        for i in range(0, self.X.shape[0], L):
            cc.add(self.X[i:i+L], self.Y[i:i+L])
        assert np.allclose(cc.weight_XY(), self.T)
        assert np.allclose(cc.sum_X(), self.sx)
        assert np.allclose(cc.moments_XX(), self.Mxx0)
        assert np.allclose(cc.moments_XY(), self.Mxy0)

    def test_XXXY_weighted_withmean(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False)
        for i in range(0, self.T, self.L):
            iX = self.X[i:i+self.L, :]
            iY = self.Y[i:i+self.L, :]
            iwe = self.weights[i:i+self.L]
            cc.add(iX, iY, weights=iwe)
        assert np.allclose(cc.weight_XY(), self.wesum)
        assert np.allclose(cc.sum_X(), self.sx_w)
        assert np.allclose(cc.moments_XX(), self.Mxx_w)
        assert np.allclose(cc.moments_XY(), self.Mxy_w)

    def test_XXXY_weighted_meanfree(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True)
        for i in range(0, self.T, self.L):
            iX = self.X[i:i+self.L, :]
            iY = self.Y[i:i+self.L, :]
            iwe = self.weights[i:i+self.L]
            cc.add(iX, iY, weights=iwe)
        assert np.allclose(cc.weight_XY(), self.wesum)
        assert np.allclose(cc.sum_X(), self.sx_w)
        assert np.allclose(cc.moments_XX(), self.Mxx0_w)
        assert np.allclose(cc.moments_XY(), self.Mxy0_w)

    def test_XXXY_sym_withmean(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False, symmetrize=True)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i+self.L], self.Y[i:i+self.L])
        assert np.allclose(cc.weight_XY(), 2*self.T)
        assert np.allclose(cc.sum_X(), self.s_sym)
        assert np.allclose(cc.moments_XX(), self.Mxx_sym)
        assert np.allclose(cc.moments_XY(), self.Mxy_sym)

    def test_XXXY_sym_meanfree(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True, symmetrize=True)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i+self.L], self.Y[i:i+self.L])
        assert np.allclose(cc.weight_XY(), 2*self.T)
        assert np.allclose(cc.sum_X(), self.s_sym)
        assert np.allclose(cc.moments_XX(), self.Mxx0_sym)
        assert np.allclose(cc.moments_XY(), self.Mxy0_sym)

    def test_XXXY_weighted_sym_withmean(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False, symmetrize=True)
        for i in range(0, self.T, self.L):
            iwe = self.weights[i:i+self.L]
            cc.add(self.X[i:i+self.L], self.Y[i:i+self.L], weights=iwe)
        assert np.allclose(cc.weight_XY(), 2 * self.wesum)
        assert np.allclose(cc.sum_X(), self.s_sym_w)
        assert np.allclose(cc.moments_XX(), self.Mxx_sym_w)
        assert np.allclose(cc.moments_XY(), self.Mxy_sym_w)

    def test_XXXY_weighted_sym_meanfree(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True, symmetrize=True)
        for i in range(0, self.T, self.L):
            iwe = self.weights[i:i+self.L]
            cc.add(self.X[i:i+self.L], self.Y[i:i+self.L], weights=iwe)
        assert np.allclose(cc.weight_XY(), 2*self.wesum)
        assert np.allclose(cc.sum_X(), self.s_sym_w)
        assert np.allclose(cc.moments_XX(), self.Mxx0_sym_w)
        assert np.allclose(cc.moments_XY(), self.Mxy0_sym_w)

if __name__ == "__main__":
    unittest.main()