
# This file is part of BHMM (Bayesian Hidden Markov Models).
#
# Copyright (c) 2016 Frank Noe (Freie Universitaet Berlin)
# and John D. Chodera (Memorial Sloan-Kettering Cancer Center, New York)
#
# BHMM is free software: you can redistribute it and/or modify
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

from __future__ import print_function
__author__ = 'noe'

import unittest

import numpy as np
import time

from bhmm import hidden
from bhmm.output_models.gaussian import GaussianOutputModel

print_speedup = True


class BenchmarkHidden(object):

    def __init__(self, nrep=10, kernel='c'):
        self.kernel = kernel
        self.nrep = nrep

        # variables
        self.nexamples = 0
        self.A = []
        self.pi = []
        self.pobs = []
        self.T = []
        self.N = []
        self.alpha = []
        self.beta = []
        self.gamma = []
        self.time_alpha = []
        self.time_beta = []
        self.time_gamma = []
        self.time_c = []
        self.time_C = []
        self.time_vpath = []
        self.alpha_mem = []
        self.beta_mem = []
        self.gamma_mem = []
        self.C_mem = []

        # second example
        A = np.array([[0.97, 0.02, 0.01],
                      [0.1,  0.8,  0.1],
                      [0.01, 0.02, 0.97]])
        pi = np.array([0.45, 0.1, 0.45])
        T = 1000000
        means = np.array([-1.0, 0.0, 1.0])
        sigmas = np.array([0.5,  0.5, 0.5])
        gom = GaussianOutputModel(3, means=means, sigmas=sigmas)
        obs = np.random.randint(3, size=T)
        pobs = gom.p_obs(obs)
        self.append_example(A, pi, pobs)

    def append_example(self, A, pi, pobs):
        i = len(self.A)
        self.A.append(A)
        self.pi.append(pi)
        self.pobs.append(pobs)
        self.N.append(A.shape[0])
        self.T.append(pobs.shape[0])
        # compute intermediates
        _, alpha, _ = self.run_forward(i, None)
        self.alpha.append(alpha)
        beta, _ = self.run_backward(i, None)
        self.beta.append(beta)
        gamma, _ = self.run_gamma(i, None)
        self.gamma.append(gamma)
        #
        self.alpha_mem.append(np.zeros((pobs.shape[0], A.shape[0])))
        self.beta_mem.append(np.zeros((pobs.shape[0], A.shape[0])))
        self.gamma_mem.append(np.zeros((pobs.shape[0], A.shape[0])))
        self.C_mem.append(np.zeros((A.shape[0], A.shape[0])))

        self.nexamples += 1

    def run_forward(self, i, out):
        logprob = 0
        alpha = None
        hidden.set_implementation(self.kernel)
        time1 = time.time()
        for k in range(self.nrep):
            logprob, alpha = hidden.forward(self.A[i], self.pobs[i], self.pi[i], alpha_out=out)
        # compare
        time2 = time.time()
        d = (time2-time1)/(1.0*self.nrep)
        return logprob, alpha, d

    def run_backward(self, i, out):
        beta = None
        hidden.set_implementation(self.kernel)
        time1 = time.time()
        for k in range(self.nrep):
            beta = hidden.backward(self.A[i], self.pobs[i], beta_out=out)
        # compare
        time2 = time.time()
        d = (time2-time1)/(1.0*self.nrep)
        return beta, d

    def run_gamma(self, i, out):
        gamma = None
        hidden.set_implementation(self.kernel)
        time1 = time.time()
        for k in range(self.nrep):
            gamma = hidden.state_probabilities(self.alpha[i], self.beta[i], gamma_out=out)
        # compare
        time2 = time.time()
        d = (time2-time1)/(1.0*self.nrep)
        return gamma, d

    def run_state_counts(self, i, out):
        c = None
        hidden.set_implementation(self.kernel)
        time1 = time.time()
        for k in range(self.nrep):
            c = hidden.state_counts(self.gamma[i], self.T[i])
        # compare
        time2 = time.time()
        d = (time2-time1)/(1.0*self.nrep)
        return c, d

    def run_transition_counts(self, i, out):
        C = None
        hidden.set_implementation(self.kernel)
        time1 = time.time()
        for k in range(self.nrep):
            C = hidden.transition_counts(self.alpha[i], self.beta[i], self.A[i], self.pobs[i], out=out)
        # compare
        time2 = time.time()
        d = (time2-time1) / (1.0*self.nrep)
        return C, d

    def run_viterbi(self, i, out):
        vpath = None
        hidden.set_implementation(self.kernel)
        time1 = time.time()
        for k in range(self.nrep):
            vpath = hidden.viterbi(self.A[i], self.pobs[i], self.pi[i])
        # compare
        time2 = time.time()
        d = (time2-time1) / (1.0*self.nrep)
        return vpath, d

    def run_comp(self, call, outs):
        """
        Reference. Just computes the time
        """
        for i in range(self.nexamples):
            if outs is None:
                res = call(i, None)
            else:
                res = call(i, outs[i])
            pkernel = 'mem'
            if print_speedup:
                print('\t' + str(call.__name__) + '\t Impl = ' + pkernel + ' Time = ' + str(int(1000.0*res[-1])) + ' ms')


def main():
    bh = BenchmarkHidden()

    # from scratch
    bh.run_comp(bh.run_forward, None)
    bh.run_comp(bh.run_backward, None)
    bh.run_comp(bh.run_gamma, None)
    bh.run_comp(bh.run_state_counts, None)
    bh.run_comp(bh.run_transition_counts, None)
    bh.run_comp(bh.run_viterbi, None)

    print()

    # in memory
    bh.run_comp(bh.run_forward, bh.alpha_mem)
    bh.run_comp(bh.run_backward, bh.beta_mem)
    bh.run_comp(bh.run_gamma, bh.gamma_mem)
    bh.run_comp(bh.run_state_counts, None)
    bh.run_comp(bh.run_transition_counts, bh.C_mem)
    bh.run_comp(bh.run_viterbi, None)


if __name__ == "__main__":
    main()
