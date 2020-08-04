# This file is part of scikit-time and MSMTools.
#
# Copyright (c) 2020, 2015, 2014 AI4Science Group, Freie Universitaet Berlin (GER)
#
# scikit-time and MSMTools is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

r"""Transition matrix sampling for non-reversible stochastic matrices.

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import numpy as np


class SamplerNonRev(object):
    def __init__(self, Z, seed: int = -1):
        """Posterior counts"""
        self.Z = Z
        """Alpha parameters for dirichlet sampling"""
        self.alpha = Z + 1.0
        """Initial state from single sample"""
        self.P = np.zeros_like(Z)
        self.rnd = np.random.RandomState(seed)
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
