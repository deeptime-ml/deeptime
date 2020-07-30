import numpy as np
from .._mle_bindings import RevSamplerFloat32, RevSamplerFloat64
from ....analysis import stationary_distribution, is_connected


class SamplerRev(object):
    def __init__(self, C, P0=None, seed: int = -1):
        from sktime.markov.tools.estimation import tmatrix
        self.C = 1.0 * C

        """Set up initial state of the chain"""
        if P0 is None:
            # only do a few iterations to get close to the MLE and suppress not converged warning
            P0 = tmatrix(C, reversible=True, maxiter=100, warn_not_converged=False)
        pi0 = stationary_distribution(P0)
        V0 = pi0[:, np.newaxis] * P0

        self.V = V0
        # self.v = self.V.sum(axis=1)
        self.c = self.C.sum(axis=1)

        """Check for valid input"""
        self.check_input()

        """Get nonzero indices"""
        self.I, self.J = np.where((self.C + self.C.T) > 0.0)

        """Init Vsampler"""
        self._sampler = RevSamplerFloat32(seed) if self.C.dtype == np.float32 else RevSamplerFloat64(seed)

    def check_input(self):
        if not self.C.dtype in (np.float32, np.float64):
            raise ValueError("Only supports float32 and float64 dtype.")
        if not np.all(self.C >= 0):
            raise ValueError("Count matrix contains negative elements")
        if not is_connected(self.C):
            raise ValueError("Count matrix is not connected")
        if not np.all(self.V >= 0.0):
            raise ValueError("P0 contains negative entries")
        if not np.allclose(self.V, self.V.T):
            raise ValueError("P0 is not reversible")
        """Check sparsity pattern"""
        iC, jC = np.where((self.C + self.C.T + np.eye(self.C.shape[0])) > 0)
        iV, jV = np.where((self.V + self.V.T + np.eye(self.V.shape[0])) > 0)
        if not np.array_equal(iC, iV):
            raise ValueError('Sparsity patterns of C and X are different.')
        if not np.array_equal(jC, jV):
            raise ValueError('Sparsity patterns of C and X are different.')

    def update(self, N=1):
        self._sampler.update(self.C, self.c, self.V, self.I, self.J, N)

    def sample(self, N=1, return_statdist=False):
        self.update(N=N)
        Vsum = self.V.sum(axis=1)
        if np.any(Vsum == 0):
            raise ValueError('...')
        else:
            pass # raise ValueError('...')
        self.V /= Vsum[..., np.newaxis]
        P = self.V  #  / Vsum[..., np.newaxis]
        if return_statdist:
            nu = 1.0 * self.V.sum(axis=1)
            pi = nu / nu.sum()
            return P, pi
        else:
            return P
