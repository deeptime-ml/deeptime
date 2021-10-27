import numpy as np


class SamplerRev:
    def __init__(self, C, P0=None, seed: int = -1):
        from deeptime.markov.tools.estimation import transition_matrix as tmatrix
        from deeptime.markov.tools.analysis import stationary_distribution
        from .._mle_bindings import RevSampler32, RevSampler64, RevSampler128

        if C.dtype not in (np.float32, np.float64, np.longdouble):
            dtype = np.float64
        else:
            dtype = C.dtype

        self.C = C.astype(dtype)

        """Set up initial state of the chain"""
        if P0 is None:
            # only do a few iterations to get close to the MLE and suppress not converged warning
            P0 = tmatrix(self.C, reversible=True, maxiter=100, warn_not_converged=False)
            assert P0.dtype == self.C.dtype
        else:
            P0 = P0.astype(self.C.dtype)
        pi0 = stationary_distribution(P0).astype(self.C.dtype)
        V0 = pi0[:, np.newaxis] * P0

        self.V = V0
        self.c = self.C.sum(axis=1)

        """Check for valid input"""
        self.check_input()

        """Get nonzero indices"""
        self.I, self.J = np.where((self.C + self.C.T) > 0.0)

        """Init Vsampler"""
        if self.C.dtype == np.float32:
            self._sampler = RevSampler32(seed)
        elif self.C.dtype == np.float64:
            self._sampler = RevSampler64(seed)
        elif self.C.dtype == np.longdouble:
            self._sampler = RevSampler128(seed)
        else:
            raise ValueError(f"Unknown dtype {self.C.dtype}")

    def check_input(self):
        from deeptime.markov.tools.analysis import is_connected
        if self.C.dtype not in (np.float32, np.float64, np.longdouble):
            raise ValueError("Only supports float32, float64, and longdouble dtype.")
        if not np.all(self.C >= 0):
            raise ValueError("Count matrix contains negative elements")
        if not is_connected(self.C):
            raise ValueError("Count matrix is not connected")
        if not np.all(self.V >= 0.0):
            raise ValueError("P0 contains negative entries")
        if not np.allclose(self.V, self.V.T, atol=1e-6):
            raise ValueError("P0 is not reversible")
        """Check sparsity pattern"""
        iC, jC = np.where((self.C + self.C.T + np.eye(self.C.shape[0])) > 0)
        iV, jV = np.where((self.V + self.V.T + np.eye(self.V.shape[0])) > 0)
        if not np.array_equal(iC, iV):
            raise ValueError('Sparsity patterns of C and X are different.')
        if not np.array_equal(jC, jV):
            raise ValueError('Sparsity patterns of C and X are different.')

    def update(self, N=1):
        self._sampler.update(self.C, self.c, self.V, self.I, self.J, int(N))

    def sample(self, N=1, return_statdist=False):
        self.update(N=N)
        Vsum = self.V.sum(axis=1)
        P = self.V / Vsum[..., np.newaxis]
        if return_statdist:
            nu = self.V.sum(axis=1)
            pi = nu / nu.sum()
            return P, pi
        else:
            return P
