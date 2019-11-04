
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

import numpy
import ctypes
cimport numpy

cdef extern from "_discrete.h":
    void _update_pout(int* obs, double* weights, int T, int N, int M, double* pout)

def update_pout(obs, weights, pout, dtype=numpy.float32):
    T = len(obs)
    N, M = numpy.shape(pout)

    # cdef numpy.ndarray[int, ndim=1, mode="c"] obs
    obs = numpy.ascontiguousarray(obs, dtype=ctypes.c_int)

    # pointers to arrays
    if dtype == numpy.float64:
        pobs     = <int*>    numpy.PyArray_DATA(obs)
        pweights = <double*> numpy.PyArray_DATA(weights)
        ppout    = <double*> numpy.PyArray_DATA(pout)
        # call
        _update_pout(pobs, pweights, T, N, M, ppout)
    else:
        raise TypeError

