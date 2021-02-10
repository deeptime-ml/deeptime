import numpy as np

from ..base import Model
from ..kernels import GaussianKernel
from ..numeric import spd_inv_sqrt, spd_eig


class KVADModel(Model):

    def __init__(self, K, U, score, fX, fY):
        super().__init__()
        self.K = K
        self.U = U
        self.score = score
        self.fX = fX
        self.fY = fY


def whiten(X, epsilon=1e-10):
    X_meanfree = X - X.mean(axis=0, keepdims=True)

    cov = 1 / (X.shape[0] - 1) * X_meanfree.T @ X_meanfree
    cov_sqrt_inv = spd_inv_sqrt(cov, epsilon=epsilon)

    return X_meanfree @ cov_sqrt_inv


def kvad(chi_X, chi_Y, Y, kernel=GaussianKernel(1.)):
    N = Y.shape[0]
    M = chi_X.shape[1]

    assert chi_X.shape == (N, M)
    assert chi_Y.shape == (N, M)

    Gyy = kernel.gram(Y)
    assert Gyy.shape == (N, N)

    chi_X_w = whiten(chi_X)
    chi_Y_w = whiten(chi_Y)

    xGx = chi_X_w.T @ Gyy @ chi_X_w

    s, U = spd_eig(xGx)

    U = U[:, :-1]
    fX = np.ones((chi_X_w.shape[0], 1 + U.shape[1]))
    fX[:, 1:] = chi_X_w @ U

    fY = np.ones((chi_Y_w.shape[0], 1 + U.shape[1]))
    fY[:, 1:] = chi_Y_w @ U

    K = 1 / N * fX.T @ fY

    score = 1/(N*N) * (np.sum(s) + np.sum(Gyy))
    return KVADModel(K, U, score, fX, fY)
