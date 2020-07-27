
# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# MSMTools is free software: you can redistribute it and/or modify
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

r"""Transition matrix sampling module. Provides a common class for sampling of

i) non-reversible transition matrices
ii) reverisble transition matrices
iii) reversible transition matrices with fixed stationary vector

from given data

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>
.. moduleauthor:: Frank Noe <frank DOT noe AT fu-berlin DOT de>

"""

import math
import numpy as np

from . sampler_nrev import SamplerNonRev
from . sampler_rev import SamplerRev
from . sampler_revpi import SamplerRevPi


class TransitionMatrixSampler(object):

    def __init__(self, C, reversible=False, mu=None, P0=None, nsteps=1, prior='sparse'):

        if not prior == 'sparse':
            raise ValueError("Only Sparse prior is currenty implemented")

        self.C = C

        # distinguish the sampling cases and initialize accordingly
        if reversible:
            if mu is None:
                if nsteps is None:
                    # use sqrt(n) as a rough guess for the decorrelation time
                    nsteps = math.sqrt(np.shape(C)[0])
                self.sampler = SamplerRev(C, P0=P0)
            else:
                if nsteps is None:
                    nsteps = 6  # because we have observed autocorrelation times of about 3.
                self.sampler = SamplerRevPi(C, mu, P0=P0)
        else:
            if mu is None:
                nsteps = 1  # just force to 1, because this is independent sampling
                self.sampler = SamplerNonRev(C-1.0)
            else:
                raise ValueError('Non reversible sampling with fixed stationary vector not implemented')

        # remember number of steps to decorrelate between samples
        self.nsteps = nsteps

    def sample(self, nsamples=1, return_statdist=False, call_back=None):
        if nsamples==1:
            return self.sampler.sample(N=self.nsteps, return_statdist=return_statdist)
        else:
            N = self.C.shape[0]
            P_samples = np.zeros((nsamples, N, N))
            if return_statdist:
                pi_samples = np.zeros((nsamples, N))
                for i in range(nsamples):
                    P_samples[i, :, :], pi_samples[i, :] = self.sampler.sample(N=self.nsteps, return_statdist=True)
                    if call_back is not None:
                        call_back()
                return P_samples, pi_samples
            else:
                for i in range(nsamples):
                    P_samples[i, :, :] = self.sampler.sample(N=self.nsteps, return_statdist=False)
                    if call_back is not None:
                        call_back()
                return P_samples
