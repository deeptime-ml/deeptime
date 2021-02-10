r"""Transition matrix sampling for non-reversible stochastic matrices.

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import numpy as np


class SamplerNonRev:
    def __init__(self, Z, seed: int = -1):
        """Posterior counts"""
        self.Z = Z
        """Alpha parameters for dirichlet sampling"""
        self.alpha = Z + 1.0
        """Initial state from single sample"""
        self.P = np.zeros_like(Z)
        self.rnd = np.random.RandomState(seed if isinstance(seed, int) and seed >= 0 else None)
        self.update()

    def update(self, N=1):
        N = self.alpha.shape[0]
        for i in range(N):
            # only pass positive alphas to dirichlet sampling.
            positive = self.alpha[i, :] > 0
            self.P[i, positive] = np.random.dirichlet(self.alpha[i, positive])

    def sample(self, N=1, return_statdist=False):
        from ....analysis import stationary_distribution

        self.update(N=N)
        if return_statdist:
            pi = stationary_distribution(self.P)
            return self.P, pi
        else:
            return self.P
