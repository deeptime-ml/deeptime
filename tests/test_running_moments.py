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
        cls.lag = 50
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

        # direct calculation, time-lagged moments of X:
        # Standard moments:
        cls.s_tlx = np.zeros(2)
        cls.s_tly = np.zeros(2)
        cls.Mxx_tl = np.zeros((2, 2))
        cls.Mxy_tl = np.zeros((2, 2))
        # Mean free moments:
        cls.Mxx0_tl = np.zeros((2, 2))
        cls.Mxy0_tl = np.zeros((2, 2))
        # Symmetric moments:
        cls.s_tl_sym = np.zeros(2)
        cls.Mxx_tl_sym = np.zeros((2, 2))
        cls.Mxy_tl_sym = np.zeros((2, 2))
        # Symmetric mean free moments:
        cls.Mxx0_tl_sym = np.zeros((2, 2))
        cls.Mxy0_tl_sym = np.zeros((2, 2))
        # Weighted moments:
        cls.s_tlx_w = np.zeros(2)
        cls.s_tly_w = np.zeros(2)
        cls.Mxx_tl_w = np.zeros((2, 2))
        cls.Mxy_tl_w = np.zeros((2, 2))
        cls.wesum_tl = 0
        # Weighted moments, integer trajectory weights:
        cls.s_tl_wtraj = np.zeros(2)
        cls.Mxx_tl_wtraj = np.zeros((2, 2))
        cls.Mxy_tl_wtraj = np.zeros((2, 2))
        cls.wesum_tl_traj = 0
        # Weighted mean free moments:
        cls.Mxx0_tl_w = np.zeros((2, 2))
        cls.Mxy0_tl_w = np.zeros((2, 2))
        # Symmetric weighted moments:
        cls.s_tl_w_sym = np.zeros(2)
        cls.Mxx_tl_w_sym = np.zeros((2, 2))
        cls.Mxy_tl_w_sym = np.zeros((2, 2))
        # Symmetric weighted mean free moments:
        cls.Mxx0_tl_w_sym = np.zeros((2, 2))
        cls.Mxy0_tl_w_sym = np.zeros((2, 2))


        q = 0
        for i in range(0, cls.T, cls.L):
            # Non-symmetric version:
            iX = cls.X[i:i+cls.L, :]
            iwe = cls.weights[i:i+cls.L]
            cls.Mxx_tl += np.dot(iX[:-cls.lag, :].T, iX[:-cls.lag, :])
            cls.Mxy_tl += np.dot(iX[:-cls.lag, :].T, iX[cls.lag:, :])
            cls.s_tlx += np.sum(iX[:-cls.lag, :], axis=0)
            cls.s_tly += np.sum(iX[cls.lag:, :], axis=0)
            # Symmetric version:
            cls.Mxx_tl_sym += np.dot(iX[:-cls.lag, :].T, iX[:-cls.lag, :]) + np.dot(iX[cls.lag:, :].T, iX[cls.lag:, :])
            cls.Mxy_tl_sym += np.dot(iX[:-cls.lag, :].T, iX[cls.lag:, :]) + np.dot(iX[cls.lag:, :].T, iX[:-cls.lag, :])
            cls.s_tl_sym += np.sum(iX[:-cls.lag, :], axis=0) + np.sum(iX[cls.lag:, :], axis=0)

            # Weighted version:
            cls.Mxx_tl_w += np.dot((iwe[:-cls.lag, None] * iX[:-cls.lag, :]).T, iX[:-cls.lag, :])
            cls.Mxy_tl_w += np.dot((iwe[:-cls.lag, None] * iX[:-cls.lag, :]).T, iX[cls.lag:, :])
            cls.s_tlx_w += np.sum((iwe[:-cls.lag, None] * iX[:-cls.lag, :]), axis=0)
            cls.s_tly_w += np.sum((iwe[:-cls.lag, None] * iX[cls.lag:, :]), axis=0)
            cls.wesum_tl += np.sum(iwe[:-cls.lag])
            # Weighted symmetric version:
            cls.s_tl_w_sym += np.sum((iwe[:-cls.lag, None] * iX[:-cls.lag, :]), axis=0) +\
                              np.sum((iwe[:-cls.lag, None] * iX[cls.lag:, :]), axis=0)
            cls.Mxx_tl_w_sym += np.dot((iwe[:-cls.lag, None] * iX[:-cls.lag, :]).T, iX[:-cls.lag, :]) +\
                                np.dot((iwe[:-cls.lag, None] * iX[cls.lag:, :]).T, iX[cls.lag:, :])
            cls.Mxy_tl_w_sym += np.dot((iwe[:-cls.lag, None] * iX[:-cls.lag, :]).T, iX[cls.lag:, :]) +\
                                np.dot((iwe[:-cls.lag, None] * iX[cls.lag:, :]).T, iX[:-cls.lag, :])
            # Weighted version, integer weights for trajectories:
            cls.s_tl_wtraj += np.sum((cls.trajweights[q] * iX[:-cls.lag, :]), axis=0)
            cls.Mxx_tl_wtraj += np.dot((cls.trajweights[q] * iX[:-cls.lag, :]).T, iX[:-cls.lag, :])
            cls.Mxy_tl_wtraj += np.dot((cls.trajweights[q] * iX[:-cls.lag, :]).T, iX[cls.lag:, :])
            cls.wesum_tl_traj += cls.trajweights[q] * (iX.shape[0] - cls.lag)
            q += 1

        # Compute the mean for the mean free versions and substract it:
        cls.m_tlx = cls.s_tlx / (cls.T - cls.nchunks*cls.lag)
        cls.m_tly = cls.s_tly / (cls.T - cls.nchunks*cls.lag)
        cls.X0_tl = cls.X - cls.m_tlx
        cls.Y0_tl = cls.X - cls.m_tly
        cls.m_tl_sym = cls.s_tl_sym / (2*(cls.T - cls.nchunks*cls.lag))
        cls.X0_tl_sym = cls.X - cls.m_tl_sym
        # Weighted means:
        cls.m_tlx_w = cls.s_tlx_w / cls.wesum_tl
        cls.m_tly_w = cls.s_tly_w / cls.wesum_tl
        cls.X0_tl_w = cls.X - cls.m_tlx_w
        cls.Y0_tl_w = cls.X - cls.m_tly_w
        cls.m_tl_w_sym = cls.s_tl_w_sym / (2*cls.wesum_tl)
        cls.X0_tl_w_sym = cls.X - cls.m_tl_w_sym

        for i in range(0, cls.T, cls.L):
            # Standard, mean-free:
            iX = cls.X0_tl[i:i+cls.L, :]
            iY = cls.Y0_tl[i:i+cls.L, :]
            iwe = cls.weights[i:i+cls.L]
            cls.Mxx0_tl += np.dot(iX[:-cls.lag, :].T, iX[:-cls.lag, :])
            cls.Mxy0_tl += np.dot(iX[:-cls.lag, :].T, iY[cls.lag:, :])
            # Symmetric, mean-free
            iX = cls.X0_tl_sym[i:i+cls.L, :]
            cls.Mxx0_tl_sym += np.dot(iX[:-cls.lag, :].T, iX[:-cls.lag, :]) + np.dot(iX[cls.lag:, :].T, iX[cls.lag:, :])
            cls.Mxy0_tl_sym += np.dot(iX[:-cls.lag, :].T, iX[cls.lag:, :]) + np.dot(iX[cls.lag:, :].T, iX[:-cls.lag, :])
            # Weighted, mean-free
            iX = cls.X0_tl_w[i:i+cls.L, :]
            iY = cls.Y0_tl_w[i:i+cls.L, :]
            cls.Mxx0_tl_w += np.dot((iwe[:-cls.lag, None] * iX[:-cls.lag, :]).T, iX[:-cls.lag, :])
            cls.Mxy0_tl_w += np.dot((iwe[:-cls.lag, None] * iX[:-cls.lag, :]).T, iY[cls.lag:, :])
            # Weighted, symmetric, mean-free
            iX = cls.X0_tl_w_sym[i:i+cls.L, :]
            cls.Mxx0_tl_w_sym += np.dot((iwe[:-cls.lag, None] * iX[:-cls.lag, :]).T, iX[:-cls.lag, :]) + \
                                 np.dot((iwe[:-cls.lag, None] * iX[cls.lag:, :]).T, iX[cls.lag:, :])
            cls.Mxy0_tl_w_sym += np.dot((iwe[:-cls.lag, None] * iX[:-cls.lag, :]).T, iX[cls.lag:, :]) + \
                                 np.dot((iwe[:-cls.lag, None] * iX[cls.lag:, :]).T, iX[:-cls.lag, :])
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

    def test_XXXY_time_lagged_withmean(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False, symmetrize=False,
                                          time_lagged=True, lag=self.lag)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i+self.L, :])
        assert np.allclose(cc.weight_XY(), self.T - self.nchunks*self.lag)
        assert np.allclose(cc.sum_X(), self.s_tlx)
        assert np.allclose(cc.sum_Y(), self.s_tly)
        assert np.allclose(cc.moments_XX(), self.Mxx_tl)
        assert np.allclose(cc.moments_XY(), self.Mxy_tl)

    def test_XXXY_time_lagged_meanfree(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True, symmetrize=False,
                                          time_lagged=True, lag=self.lag)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i+self.L, :])
        assert np.allclose(cc.weight_XY(), self.T - self.nchunks*self.lag)
        assert np.allclose(cc.sum_X(), self.s_tlx)
        assert np.allclose(cc.sum_Y(), self.s_tly)
        assert np.allclose(cc.moments_XX(), self.Mxx0_tl)
        assert np.allclose(cc.moments_XY(), self.Mxy0_tl)

    def test_XXXY_weighted_time_lagged_withmean(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False, symmetrize=False,
                                          time_lagged=True, lag=self.lag)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i+self.L, :], weights=self.weights[i:i+self.L])
        assert np.allclose(cc.weight_XY(), self.wesum_tl)
        assert np.allclose(cc.sum_X(), self.s_tlx_w)
        assert np.allclose(cc.sum_Y(), self.s_tly_w)
        assert np.allclose(cc.moments_XX(), self.Mxx_tl_w)
        assert np.allclose(cc.moments_XY(), self.Mxy_tl_w)

    def test_XXXY_weighted_time_lagged_meanfree(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True, symmetrize=False,
                                          time_lagged=True, lag=self.lag)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i+self.L, :], weights=self.weights[i:i+self.L])
        assert np.allclose(cc.weight_XY(), self.wesum_tl)
        assert np.allclose(cc.sum_X(), self.s_tlx_w)
        assert np.allclose(cc.sum_Y(), self.s_tly_w)
        assert np.allclose(cc.moments_XX(), self.Mxx0_tl_w)
        assert np.allclose(cc.moments_XY(), self.Mxy0_tl_w)

    def test_XXXY_weighted_trajs_time_lagged_withmean(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False, symmetrize=False,
                                          time_lagged=True, lag=self.lag)
        q = 0
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i+self.L, :], weights=self.trajweights[q])
            q += 1
        assert np.allclose(cc.weight_XY(), self.wesum_tl_traj)
        assert np.allclose(cc.sum_X(), self.s_tl_wtraj)
        assert np.allclose(cc.moments_XX(), self.Mxx_tl_wtraj)
        assert np.allclose(cc.moments_XY(), self.Mxy_tl_wtraj)

    def test_XXXY_sym_time_lagged_withmean(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False, symmetrize=True,
                                          time_lagged=True, lag=self.lag)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i+self.L, :])
        assert np.allclose(cc.weight_XY(), 2*(self.T - self.nchunks*self.lag))
        assert np.allclose(cc.sum_X(), self.s_tl_sym)
        assert np.allclose(cc.moments_XX(), self.Mxx_tl_sym)
        assert np.allclose(cc.moments_XY(), self.Mxy_tl_sym)

    def test_XXXY_sym_time_lagged_meanfree(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True, symmetrize=True,
                                          time_lagged=True, lag=self.lag)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i+self.L, :])
        assert np.allclose(cc.weight_XY(), 2*(self.T - self.nchunks*self.lag))
        assert np.allclose(cc.sum_X(), self.s_tl_sym)
        assert np.allclose(cc.moments_XX(), self.Mxx0_tl_sym)
        assert np.allclose(cc.moments_XY(), self.Mxy0_tl_sym)

    def test_XXXY_weighted_sym_time_lagged_withmean(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False, symmetrize=True,
                                          time_lagged=True, lag=self.lag)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i+self.L, :], weights=self.weights[i:i+self.L])
        assert np.allclose(cc.weight_XY(), 2*self.wesum_tl)
        assert np.allclose(cc.sum_X(), self.s_tl_w_sym)
        assert np.allclose(cc.moments_XX(), self.Mxx_tl_w_sym)
        assert np.allclose(cc.moments_XY(), self.Mxy_tl_w_sym)

    def test_XXXY_weighted_sym_time_lagged_meanfree(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True, symmetrize=True,
                                          time_lagged=True, lag=self.lag)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i+self.L, :], weights=self.weights[i:i+self.L])
        assert np.allclose(cc.weight_XY(), 2*self.wesum_tl)
        assert np.allclose(cc.sum_X(), self.s_tl_w_sym)
        assert np.allclose(cc.moments_XX(), self.Mxx0_tl_w_sym)
        assert np.allclose(cc.moments_XY(), self.Mxy0_tl_w_sym)

if __name__ == "__main__":
    unittest.main()