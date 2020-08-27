import numpy as np

from ..base import Model
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


def gramian_gauss(Y, sigma=1.):
    Y = Y.T
    ri = np.expand_dims(Y, axis=-2)
    rj = np.expand_dims(Y, axis=-1)
    rij = ri - rj
    D = np.add.reduce(np.square(rij), axis=0, keepdims=False)
    return np.exp(-D / (2. * sigma ** 2))


def kvad(chi_X, chi_Y, Y, kernel=lambda x: gramian_gauss(x, 1.)):
    N = Y.shape[0]
    M = chi_X.shape[1]

    print(f"n_frames={N}, chi n_dims={M}")

    assert chi_X.shape == (N, M)
    assert chi_Y.shape == (N, M)

    Gyy = kernel(Y)
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
    # score = 1 / (N * N) * (np.trace(U.T @ chi_X_w.T @ Gyy @ chi_X_w @ U) + np.sum(Gyy))
    # print(f"K shape {K.shape}, M={M}, m={m}, N={N}, score={score:.5f}")
    # score2 = 1/(N*N) * (np.sum(s) + np.sum(Gyy))
    # print("score2", score2)

    return KVADModel(K, U, score, fX, fY)
