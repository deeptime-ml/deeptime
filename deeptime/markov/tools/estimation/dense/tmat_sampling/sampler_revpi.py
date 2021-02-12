import numpy as np


class SamplerRevPi:

    def __init__(self, C, pi, P0=None, P_mle=None, eps=0.1, seed=-1):
        from deeptime.markov.tools.estimation.dense.mle import mle_trev_given_pi
        from .._mle_bindings import RevPiSampler32, RevPiSampler64, RevPiSampler128

        dtype = C.dtype
        if dtype not in (np.float32, np.float64, np.longdouble):
            dtype = np.float64
        self.C = C.astype(dtype)
        self.pi = pi.astype(dtype)

        if P_mle is None:
            P_mle = mle_trev_given_pi(C, pi)

        if P0 is None:
            cdiag = np.diag(C)
            """Entries with cii=0"""
            ind = (cdiag == 0)
            """Add counts, s.t. cii+bii>0 for all i"""
            bdiag = np.zeros_like(cdiag)
            bdiag[ind] = 1.0
            B = np.diag(bdiag)
            P0 = mle_trev_given_pi(C + B, pi)

        """Diagonal prior parameters"""
        b = np.zeros(C.shape[0])

        cii = C[np.diag_indices(C.shape[0])]

        """Zero diagonal entries of C"""
        ind1 = np.isclose(cii, 0.0)
        b[ind1] = eps

        """Non-zero diagonal entries of P0"""
        pii0 = P_mle[np.diag_indices(P_mle.shape[0])]
        ind2 = (pii0 > 0.0)

        """Find elements pii0>0 and cii=0"""
        ind3 = np.logical_and(ind1, ind2)
        b[ind3] = 1.0

        self.b = b

        """Initial state of the chain"""
        self.X = pi[:, np.newaxis] * P0
        self.X /= self.X.sum()

        """Check for valid input"""
        self.check_input()

        """Set up index arrays"""
        self.I, self.J = np.where((self.C + self.C.T) > 0.0)

        """Init the xsampler"""
        if self.C.dtype == np.float32:
            self.xsampler = RevPiSampler32(seed)
            self.X = self.X.astype(np.float32)
            self.b = self.b.astype(np.float32)
            self.pi = self.pi.astype(np.float32)
        elif self.C.dtype == np.float64:
            self.xsampler = RevPiSampler64(seed)
            self.X = self.X.astype(np.float64)
            self.b = self.b.astype(np.float64)
            self.pi = self.pi.astype(np.float64)
        elif self.C.dtype == np.longdouble:
            self.xsampler = RevPiSampler128(seed)
            self.X = self.X.astype(np.longdouble)
            self.b = self.b.astype(np.longdouble)
            self.pi = self.pi.astype(np.longdouble)
        else:
            raise ValueError(f"Unknown dtype {self.C.dtype}")

    def check_input(self):
        from deeptime.markov.tools.analysis import is_connected

        if not np.all(self.C >= 0):
            raise ValueError("Count matrix contains negative elements")
        if not is_connected(self.C, directed=False):
            raise ValueError("Count matrix is not connected")
        if not np.all(self.X >= 0.0):
            raise ValueError("P0 contains negative entries")
        if not np.allclose(self.X, self.X.T):
            raise ValueError("P0 is not reversible")
        """Check sparsity pattern - ignore diagonal"""
        C_sym = self.C + self.C.T
        X_sym = self.X + self.X.T
        ind = np.diag_indices(C_sym.shape[0])
        C_sym[ind] = 0.0
        X_sym[ind] = 0.0
        iC, jC = np.where(C_sym > 0.0)
        iV, jV = np.where(X_sym > 0.0)
        if not np.array_equal(iC, iV):
            raise ValueError('Sparsity patterns of C and X are different.')
        if not np.array_equal(jC, jV):
            raise ValueError('Sparsity patterns of C and X are different.')

    def update(self, N=1):
        for i in range(N):
            self.xsampler.update(self.C, self.X, self.b)

    def sample(self, N=1, return_statdist=False):
        self.update(N=N)
        P = self.X / self.pi[:, np.newaxis]
        if return_statdist:
            return P, self.pi
        else:
            return P
