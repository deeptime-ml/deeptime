
# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group
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

r"""Birth death chain for testing

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import numpy as np

from scipy.sparse import diags


class BirthDeathChain(object):
    """Birth and death chain class

    A general birth and death chain on a d-dimensional state space
    has the following transition matrix

                q_i,    j=i-1 for i>0
        p_ij=   r_i     j=i
                p_i     j=i+1 for i<d-1

    """

    def __init__(self, q, p):
        """Generate a birth and death chain from creation and
        anhilation probabilities.

        Parameters
        ----------
        q : array_like
            Anhilation probabilities for transition from i to i-1
        p : array-like
            Creation probabilities for transition from i to i+1

        """
        if q[0] != 0.0:
            raise ValueError('Probability q[0] must be zero')
        if p[-1] != 0.0:
            raise ValueError('Probability p[-1] must be zero')
        if not np.all(q + p <= 1.0):
            raise ValueError('Probabilities q+p can not exceed one')
        self.q = q
        self.p = p
        self.r = 1 - self.q - self.p
        self.dim = self.r.shape[0]

    def transition_matrix(self):
        """Tridiagonal transition matrix for birth and death chain

        Returns
        -------
        P : (N,N) ndarray
            Transition matrix for birth and death chain with given
            creation and anhilation probabilities.

        """
        P0 = np.diag(self.r, k=0)
        P1 = np.diag(self.p[0:-1], k=1)
        P_1 = np.diag(self.q[1:], k=-1)
        return P0 + P1 + P_1

    def transition_matrix_sparse(self):
        """Tridiagonal transition matrix for birth and death chain

        Returns
        -------
        P : (N,N) scipy.sparse matrix
            Transition matrix for birth and death chain with given
            birth and death probabilities.

        """
        P = diags([self.q[1:], self.r, self.p[0:-1]], [-1, 0, 1])
        return P

    def stationary_distribution(self):
        a = np.zeros(self.dim)
        a[0] = 1.0
        a[1:] = np.cumprod(self.p[0:-1] / self.q[1:])
        mu = a / np.sum(a)
        return mu

    def committor_forward(self, a, b):
        r"""Forward committor for birth-and-death-chain.

        The forward committor is the probability to hit
        state b before hitting state a starting in state x,

            u_x=P_x(T_b<T_a)

        T_i is the first arrival time of the chain to state i,

            T_i = inf( t>0 | X_t=i )

        Parameters
        ----------
        a : int
            State index
        b : int
            State index

        Returns
        -------
        u : (M,) ndarray
            Vector of committor probabilities.

        """
        u = np.zeros(self.dim)
        g = np.zeros(self.dim - 1)
        g[0] = 1.0
        g[1:] = np.cumprod(self.q[1:-1] / self.p[1:-1])

        """If a and b are equal the event T_b<T_a is impossible
           for any starting state x so that the committor is
           zero everywhere"""
        if a == b:
            return u
        elif a < b:
            """Birth-death chain has to hit a before it can hit b"""
            u[0:a + 1] = 0.0
            """Birth-death chain has to hit b before it can hit a"""
            u[b:] = 1.0
            """Intermediate states are given in terms of sums of g"""
            u[a + 1:b] = np.cumsum(g[a:b])[0:-1] / np.sum(g[a:b])
            return u
        else:
            u[0:b + 1] = 1.0
            u[a:] = 0.0
            u[b + 1:a] = (np.cumsum(g[b:a])[0:-1] / np.sum(g[b:a]))[::-1]
            return u

    def committor_backward(self, a, b):
        r"""Backward committor for birth-and-death-chain.

        The backward committor is the probability for a chain in state
        x chain to originate from state a instead of coming from
        state b,

            w_x=P_x(t_a<t_b)

        t_i is the last exit time of the chain from state i,

            t_i = inf( t>0 | X(-t)=i )

        Parameters
        ----------
        a : int
            State index
        b : int
            State index

        Returns
        -------
        w : (M,) ndarray
            Vector of committor probabilities.

        Remark
        ------
        The birth-death chain is time-reversible,

            P(t_a<t_b)=P(T_a<T_b)=1-P(T_b<T_a),

        therefore we can express the backward
        comittor probabilities in terms of the forward
        committor probabilities,

            w=1-u

        """
        return 1.0 - self.committor_forward(a, b)

    def flux(self, a, b):
        r"""The flux network for the reaction from
        A=[0,...,a] => B=[b,...,M].

        Parameters
        ----------
        a : int
            State index
        b : int
            State index

        Returns
        -------
        flux : (M, M) ndarray
            Matrix of flux values between pairs of states.

        """
        # if b<=a:
        # raise ValueError("State index b has to be strictly larger than state index a")
        qminus = self.committor_backward(a, b)
        qplus = self.committor_forward(a, b)
        P = self.transition_matrix()
        pi = self.stationary_distribution()

        flux = pi[:, np.newaxis] * qminus[:, np.newaxis] * P * qplus[np.newaxis, :]
        ind = np.diag_indices(P.shape[0])
        flux[ind] = 0.0
        return flux

    def netflux(self, a, b):
        r"""The netflux network for the reaction from
        A=[0,...,a] => B=[b,...,M].

        Parameters
        ----------
        a : int
            State index
        b : int
            State index

        Returns
        -------
        netflux : (M, M) ndarray
            Matrix of flux values between pairs of states.

        """
        flux = self.flux(a, b)
        netflux = flux - np.transpose(flux)
        ind = (netflux < 0.0)
        netflux[ind] = 0.0
        return netflux

    def totalflux(self, a, b):
        r"""The tiotal flux for the reaction
        A=[0,...,a] => B=[b,...,M].

        Parameters
        ----------
        a : int
            State index
        b : int
            State index

        Returns
        -------
        F : float
            The total flux between reactant and product

        """
        flux = self.flux(a, b)
        A = list(range(a + 1))
        notA = list(range(a + 1, flux.shape[0]))
        F = flux[A, :][:, notA].sum()
        return F

    def rate(self, a, b):
        F = self.totalflux(a, b)
        pi = self.stationary_distribution()
        qminus = self.committor_backward(a, b)
        kAB = F / (pi * qminus).sum()
        return kAB
