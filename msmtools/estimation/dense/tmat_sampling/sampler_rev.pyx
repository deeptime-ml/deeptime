
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

r"""Transition matrix sampling for revrsible stochastic matrices.

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>
.. moduleauthor:: Frank Noe <frank DOT noe AT fu-berlin DOT de>

"""
from __future__ import absolute_import

import numpy as np
import ctypes
cimport numpy as np

from ....analysis import statdist, is_connected


cdef extern from "sample_rev.h":
    void _update(double* C, double* sumC, double* X, int n, int n_step)

    void _update_sparse(double* C, double* sumC, double* X, double* sumX,
                        int* I, int* J, int n, int n_idx, int n_step)

    double _update_step(double v0, double v1, double v2,
                        double c0, double c1, double c2, int random_walk_stepsize)

cdef extern from "rnglib/rnglib.h":
    void initialize()
    void set_initial_seed(int g1, int g2)

cdef class VSampler(object):

    def __init__(self):
        """Seed the generator upon init"""
        initialize()
        set_initial_seed(np.random.randint(1, 2147483563),
                         np.random.randint(1, 2147483399))

    # def update(self, C, sumC, X, nstep):
    #     n = C.shape[0]
    #     pC    = <double*> np.PyArray_DATA(C)
    #     psumC = <double*> np.PyArray_DATA(sumC)
    #     pX    = <double*> np.PyArray_DATA(X)
    #     # call
    #     _update(pC, psumC, pX, n, nstep)

    def update_sparse(self, C, sumC, X, I, J, nstep):
        n = C.shape[0]
        n_idx = len(I)
        sumX = np.zeros( (n), dtype=ctypes.c_double, order='C' )
        sumX[:] = X.sum(axis=1)

        cdef np.ndarray[int, ndim=1, mode="c"] cI
        cI = np.array( I, dtype=ctypes.c_int, order='C' )
        cdef np.ndarray[int, ndim=1, mode="c"] cJ
        cJ = np.array( J, dtype=ctypes.c_int, order='C' )

        pC    = <double*> np.PyArray_DATA(C)
        psumC = <double*> np.PyArray_DATA(sumC)
        pX    = <double*> np.PyArray_DATA(X)
        psumX = <double*> np.PyArray_DATA(sumX)
        pI    = <int*>    np.PyArray_DATA(cI)
        pJ    = <int*>    np.PyArray_DATA(cJ)
        # call
        _update_sparse(pC, psumC, pX, psumX, pI, pJ, n, n_idx, nstep)


class SamplerRev(object):

    def __init__(self, C, P0=None):
        from msmtools.estimation import tmatrix
        self.C = 1.0*C

        """Set up initial state of the chain"""
        if P0 is None:
            # only do a few iterations to get close to the MLE and suppress not converged warning
            P0 = tmatrix(C, reversible=True, maxiter=100, warn_not_converged=False)
        pi0 = statdist(P0)
        V0 = pi0[:,np.newaxis] * P0

        self.V = V0
        # self.v = self.V.sum(axis=1)
        self.c = self.C.sum(axis=1)

        """Check for valid input"""
        self.check_input()

        """Get nonzero indices"""
        self.I, self.J = np.where( (self.C + self.C.T)>0.0 )

        """Init Vsampler"""
        self.vsampler = VSampler()

    def check_input(self):
        if not np.all(self.C>=0):
            raise ValueError("Count matrix contains negative elements")
        if not is_connected(self.C):
            raise ValueError("Count matrix is not connected")
        if not np.all(self.V>=0.0):
            raise ValueError("P0 contains negative entries")
        if not np.allclose(self.V, self.V.T):
            raise ValueError("P0 is not reversible")
        """Check sparsity pattern"""
        iC, jC = np.where( (self.C+self.C.T+np.eye(self.C.shape[0]))>0 )
        iV, jV = np.where( (self.V+self.V.T+np.eye(self.V.shape[0]))>0 )
        #print 'equal perms?', np.array_equal(np.array(sorted([(i,j) for i,j in zip(iC, jC) if i!=j])), np.array(sorted([(i,j) for i,j in  zip(iV, jV) if i!=j])))
        if not np.array_equal(iC, iV):
            raise ValueError('Sparsity patterns of C and X are different.')
        if not np.array_equal(jC, jV):
            raise ValueError('Sparsity patterns of C and X are different.')

    def update(self, N=1):
        self.vsampler.update_sparse(self.C, self.c, self.V, self.I, self.J, N)

    def sample(self, N=1, return_statdist=False):
        self.update(N=N)
        P = self.V/self.V.sum(axis=1)[:, np.newaxis]
        if return_statdist:
            nu = 1.0*self.V.sum(axis=1)
            pi = nu/nu.sum()
            return P, pi
        else:
            return P
