
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
import unittest
import numpy as np
import time
from bhmm import hidden
from bhmm.output_models.gaussian import GaussianOutputModel

print_speedup = False


class TestHidden(unittest.TestCase):

    def setUp(self):
        self.nexamples = 0
        self.A = []
        self.pi = []
        self.pobs = []
        self.T = []
        self.N = []
        self.logprob = []
        self.alpha = []
        self.time_alpha = []
        self.beta = []
        self.time_beta = []
        self.gamma = []
        self.time_gamma = []
        self.c = []
        self.time_c = []
        self.C = []
        self.time_C = []
        self.vpath = []
        self.time_vpath = []
        self.alpha_mem = []
        self.beta_mem = []
        self.gamma_mem = []
        self.C_mem = []

        # first toy example
        A = np.array([[0.9, 0.1],
                      [0.1, 0.9]])
        pi = np.array([0.5, 0.5])
        pobs = np.array([[0.1, 0.9],
                         [0.1, 0.9],
                         [0.1, 0.9],
                         [0.1, 0.9],
                         [0.5, 0.5],
                         [0.9, 0.1],
                         [0.9, 0.1],
                         [0.9, 0.1],
                         [0.9, 0.1],
                         [0.9, 0.1]])
        self.append_example(A, pi, pobs)

        # second example
        A = np.array([[0.97, 0.02, 0.01],
                      [0.1,  0.8,  0.1],
                      [0.01, 0.02, 0.97]])
        pi = np.array([0.45, 0.1, 0.45])
        T = 10000
        means = np.array([-1.0, 0.0, 1.0])
        sigmas = np.array([0.5, 0.5, 0.5])
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
        logprob, alpha, t = self.run_forward(i, 'python', None)
        self.logprob.append(logprob)
        self.alpha.append(alpha)
        self.time_alpha.append(t)
        beta, t = self.run_backward(i, 'python', None)
        self.beta.append(beta)
        self.time_beta.append(t)
        gamma, t = self.run_gamma(i, 'python', None)
        self.gamma.append(gamma)
        self.time_gamma.append(t)
        c, t = self.run_state_counts(i, 'python', None)
        self.c.append(c)
        self.time_c.append(t)
        C, t = self.run_transition_counts(i, 'python', None)
        self.C.append(C)
        self.time_C.append(t)
        vpath, t = self.run_viterbi(i, 'python', None)
        self.vpath.append(vpath)
        self.time_vpath.append(t)
        #
        self.alpha_mem.append(np.zeros((pobs.shape[0], A.shape[0])))
        self.beta_mem.append(np.zeros((pobs.shape[0], A.shape[0])))
        self.gamma_mem.append(np.zeros((pobs.shape[0], A.shape[0])))
        self.C_mem.append(np.zeros((A.shape[0], A.shape[0])))

        self.nexamples += 1

    def run_all(self, A, pobs, pi):
        # forward
        logprob, alpha = hidden.forward(A, pobs, pi)
        # backward
        beta = hidden.backward(A, pobs)
        # gamma
        gamma = hidden.state_probabilities(alpha, beta)
        # state counts
        T = pobs.shape[0]
        statecount = hidden.state_counts(gamma, T)
        # transition counts
        C = hidden.transition_counts(alpha, beta, A, pobs)
        # viterbi path
        vpath = hidden.viterbi(A, pobs, pi)
        # return
        return logprob, alpha, beta, gamma, statecount, C, vpath

    def run_all_mem(self, A, pobs, pi):
        T = pobs.shape[0]
        N = A.shape[0]
        alpha = np.zeros((T, N))
        beta = np.zeros((T, N))
        gamma = np.zeros((T, N))
        C = np.zeros((N, N))
        logprob, alpha = hidden.forward(A, pobs, pi, alpha_out=alpha)
        # backward
        hidden.backward(A, pobs, beta_out=beta)
        # gamma
        hidden.state_probabilities(alpha, beta, gamma_out=gamma)
        # state counts
        statecount = hidden.state_counts(gamma, T)
        # transition counts
        hidden.transition_counts(alpha, beta, A, pobs, out=self.C)
        # viterbi path
        vpath = hidden.viterbi(A, pobs, pi)
        # return
        return logprob, alpha, beta, gamma, statecount, C, vpath

    def tearDown(self):
        pass

    def run_forward(self, i, kernel, out):
        nrep = max(1, int(10000/self.T[i]))
        logprob = 0
        alpha = None
        hidden.set_implementation(kernel)
        time1 = time.time()
        for k in range(nrep):
            logprob, alpha = hidden.forward(self.A[i], self.pobs[i], self.pi[i], alpha_out=out)
        # compare
        time2 = time.time()
        d = (time2-time1) / (1.0*nrep)
        return logprob, alpha, d

    def run_backward(self, i, kernel, out):
        nrep = max(1, int(10000/self.T[i]))
        beta = None
        hidden.set_implementation(kernel)
        time1 = time.time()
        for k in range(nrep):
            beta = hidden.backward(self.A[i], self.pobs[i], beta_out=out)
        # compare
        time2 = time.time()
        d = (time2-time1)/(1.0*nrep)
        return beta, d

    def run_gamma(self, i, kernel, out):
        nrep = max(1, int(10000/self.T[i]))
        gamma = None
        hidden.set_implementation(kernel)
        time1 = time.time()
        for k in range(nrep):
            gamma = hidden.state_probabilities(self.alpha[i], self.beta[i], gamma_out=out)
        # compare
        time2 = time.time()
        d = (time2-time1) / (1.0*nrep)
        return gamma, d

    def run_state_counts(self, i, kernel, out):
        nrep = max(1, int(10000/self.T[i]))
        c = None
        hidden.set_implementation(kernel)
        time1 = time.time()
        for k in range(nrep):
            c = hidden.state_counts(self.gamma[i], self.T[i])
        # compare
        time2 = time.time()
        d = (time2-time1)/(1.0*nrep)
        return c, d

    def run_transition_counts(self, i, kernel, out):
        nrep = max(1, int(10000/self.T[i]))
        C = None
        hidden.set_implementation(kernel)
        time1 = time.time()
        for k in range(nrep):
            C = hidden.transition_counts(self.alpha[i], self.beta[i], self.A[i], self.pobs[i], out=out)
        # compare
        time2 = time.time()
        d = (time2-time1) / (1.0*nrep)
        return C, d

    def run_viterbi(self, i, kernel, out):
        nrep = max(1, int(10000/self.T[i]))
        vpath = None
        hidden.set_implementation(kernel)
        time1 = time.time()
        for k in range(nrep):
            vpath = hidden.viterbi(self.A[i], self.pobs[i], self.pi[i])
        # compare
        time2 = time.time()
        d = (time2-time1) / (1.0*nrep)
        return vpath, d

    def run_abs(self, call, kernel):
        """
        Reference. Just computes the time
        """
        for i in range(self.nexamples):
            res = call(i, kernel, None)
            if print_speedup:
                print('\t' + str(call.__name__) + '\t Example ' + str(i)
                      + '\t Impl = ' + str(kernel) + ' Time = ' + str(res[-1]))

    def run_comp(self, call, kernel, outs, refs, reftime):
        """
        Reference. Just computes the time
        """
        for i in range(self.nexamples):
            if outs is None:
                res = call(i, kernel, None)
            else:
                res = call(i, kernel, outs[i])
            for j in range(len(res)-1):
                myres = res[j]
                refres = refs[j][i]
                self.assertTrue(np.allclose(myres, refres))
            if outs is None:
                pkernel = kernel
            else:
                pkernel = kernel + ' mem'
            if print_speedup:
                print('\t' + str(call.__name__) + '\t Example ' + str(i) + '\t Impl = ' + pkernel
                      + ' Speedup = ' + str(reftime[i]/res[-1]))

    def test_forward_p(self):
        self.run_abs(self.run_forward, 'python')

    def test_forward_p_mem(self):
        self.run_comp(self.run_forward, 'python', self.alpha_mem, [self.logprob, self.alpha], self.time_alpha)

    def test_forward_c(self):
        self.run_comp(self.run_forward, 'c', None, [self.logprob, self.alpha], self.time_alpha)

    def test_forward_c_mem(self):
        self.run_comp(self.run_forward, 'c', self.alpha_mem, [self.logprob, self.alpha], self.time_alpha)

    def test_backward_p(self):
        self.run_abs(self.run_backward, 'python')

    def test_backward_p_mem(self):
        self.run_comp(self.run_backward, 'python', self.beta_mem, [self.beta], self.time_beta)

    def test_backward_c(self):
        self.run_comp(self.run_backward, 'c', None, [self.beta], self.time_beta)

    def test_backward_c_mem(self):
        self.run_comp(self.run_backward, 'c', self.beta_mem, [self.beta], self.time_beta)

    def test_gamma_p(self):
        self.run_abs(self.run_gamma, 'python')

    def test_gamma_p_mem(self):
        self.run_comp(self.run_gamma, 'python', self.gamma_mem, [self.gamma], self.time_gamma)

    def test_gamma_c(self):
        self.run_comp(self.run_gamma, 'c', None, [self.gamma], self.time_gamma)

    def test_gamma_c_mem(self):
        self.run_comp(self.run_gamma, 'c', self.gamma_mem, [self.gamma], self.time_gamma)

    def test_state_counts_p(self):
        self.run_abs(self.run_state_counts, 'python')

    def test_state_counts_p_mem(self):
        self.run_comp(self.run_state_counts, 'python', None, [self.c], self.time_c)

    def test_state_counts_c(self):
        self.run_comp(self.run_state_counts, 'c', None, [self.c], self.time_c)

    def test_state_counts_c_mem(self):
        self.run_comp(self.run_state_counts, 'c', None, [self.c], self.time_c)

    def test_transition_counts_p(self):
        self.run_abs(self.run_transition_counts, 'python')

    def test_transition_counts_p_mem(self):
        self.run_comp(self.run_transition_counts, 'python', self.C_mem, [self.C], self.time_C)

    def test_transition_counts_c(self):
        self.run_comp(self.run_transition_counts, 'c', None, [self.C], self.time_C)

    def test_transition_counts_c_mem(self):
        self.run_comp(self.run_transition_counts, 'c', self.C_mem, [self.C], self.time_C)

    def test_viterbi_p(self):
        self.run_abs(self.run_viterbi, 'python')

    def test_viterbi_p_mem(self):
        self.run_comp(self.run_viterbi, 'python', None, [self.vpath], self.time_vpath)

    def test_viterbi_c(self):
        self.run_comp(self.run_viterbi, 'c', None, [self.vpath], self.time_vpath)

    def test_viterbi_c_mem(self):
        self.run_comp(self.run_viterbi, 'c', None, [self.vpath], self.time_vpath)

    def test_fbtime_p_mem(self):
        for i in range(self.nexamples):
            ttot = 0.0
            logprob, alpha, t = self.run_forward(i, 'python', self.alpha_mem[i])
            ttot += t
            beta, t = self.run_backward(i, 'python', self.beta_mem[i])
            ttot += t
            gamma, t = self.run_gamma(i, 'python', self.gamma_mem[i])
            ttot += t
            c, t = self.run_state_counts(i, 'python', None)
            ttot += t
            C, t = self.run_transition_counts(i, 'python', self.C_mem[i])
            ttot += t
            tref = self.time_alpha[i] + self.time_beta[i] + self.time_gamma[i] + self.time_c[i] + self.time_C[i]
            if print_speedup:
                print ('TOTAL speedup forward-backward example '+str(i)+'\t impl=python mem: \t'+str(tref/ttot))

    def test_fbtime_c(self):
        for i in range(self.nexamples):
            ttot = 0.0
            logprob, alpha, t = self.run_forward(i, 'c', None)
            ttot += t
            beta, t = self.run_backward(i, 'c', None)
            ttot += t
            gamma, t = self.run_gamma(i, 'c', None)
            ttot += t
            c, t = self.run_state_counts(i, 'c', None)
            ttot += t
            C, t = self.run_transition_counts(i, 'c', None)
            ttot += t
            tref = self.time_alpha[i] + self.time_beta[i] + self.time_gamma[i] + self.time_c[i] + self.time_C[i]
            if print_speedup:
                print ('TOTAL speedup forward-backward example '+str(i)+'\t impl=c: \t'+str(tref/ttot))

    def test_fbtime_c_mem(self):
        for i in range(self.nexamples):
            ttot = 0.0
            logprob, alpha, t = self.run_forward(i, 'c', self.alpha_mem[i])
            ttot += t
            beta, t = self.run_backward(i, 'c', self.beta_mem[i])
            ttot += t
            gamma, t = self.run_gamma(i, 'c', self.gamma_mem[i])
            ttot += t
            c, t = self.run_state_counts(i, 'c', None)
            ttot += t
            C, t = self.run_transition_counts(i, 'c', self.C_mem[i])
            ttot += t
            tref = self.time_alpha[i] + self.time_beta[i] + self.time_gamma[i] + self.time_c[i] + self.time_C[i]
            if print_speedup:
                print ('TOTAL speedup forward-backward example '+str(i)+'\t impl=c mem: \t'+str(tref/ttot))


if __name__ == "__main__":
    unittest.main()
