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

import ctypes

import numpy

cdef extern from "_hidden.h":
    double _forward(double *alpha, const double *A, const double *pobs, const double *pi, const int N, const int T)
    void _backward(double *beta, const double *A, const double *pobs, const int N, const int T)
    int _compute_transition_counts(double *transition_counts, const double *A, const double *pobs, const double *alpha,
                                   const double *beta, int N, int T)
    int _compute_viterbi(int *path, const double *A, const double *pobs, const double *pi, int N, int T)
    int _sample_path(int *path, const double *alpha, const double *A, const double *pobs, const int N, const int T)
    int _BHMM_ERR_NO_MEM


def cdef_double_array(n1, n2):
    cdef numpy.ndarray[double, ndim=2, mode="c"] out = numpy.zeros((n1, n2), dtype=numpy.double, order='C')
    return out


def forward(A, pobs, pi, T=None, alpha_out=None):
    if T is None:
        T = pobs.shape[0]  # if not set, use the length of pobs as trajectory length
    elif T > pobs.shape[0]:
        raise TypeError('T must be at most the length of pobs.')
    N = A.shape[0]
    if alpha_out is None:
        alpha = cdef_double_array(T, N)
    elif T > alpha_out.shape[0]:
        raise TypeError('alpha_out must at least have length T in order to fit trajectory.')
    else:
        alpha = alpha_out

    palpha = <double*> numpy.PyArray_DATA(alpha)
    pA = <double*> numpy.PyArray_DATA(A)
    ppobs = <double*> numpy.PyArray_DATA(pobs)
    ppi = <double*> numpy.PyArray_DATA(pi)
    # call
    logprob = _forward(palpha, pA, ppobs, ppi, N, T)
    return logprob, alpha

def backward(A, pobs, T=None, beta_out=None):
    if T is None:
        T = pobs.shape[0]  # if not set, use the length of pobs as trajectory length
    elif T > pobs.shape[0]:
        raise ValueError('T must be at most the length of pobs.')
    N = A.shape[0]
    if beta_out is None:
        beta = cdef_double_array(T, N)
    elif T > beta_out.shape[0]:
        raise ValueError('beta_out must at least have length T in order to fit trajectory.')
    else:
        beta = beta_out

    pbeta = <double*> numpy.PyArray_DATA(beta)
    pA = <double*> numpy.PyArray_DATA(A)
    ppobs = <double*> numpy.PyArray_DATA(pobs)
    # call
    _backward(pbeta, pA, ppobs, N, T)
    return beta


def transition_counts(alpha, beta, A, pobs, T=None, out=None):
    if T is None:
        T = pobs.shape[0]  # if not set, use the length of pobs as trajectory length
    elif T > pobs.shape[0]:
        raise ValueError('T must be at most the length of pobs.')
    N = len(A)
    if out is None:
        C = cdef_double_array(N, N)
    else:
        C = out

    pC = <double*> numpy.PyArray_DATA(C)
    pA = <double*> numpy.PyArray_DATA(A)
    ppobs = <double*> numpy.PyArray_DATA(pobs)
    palpha = <double*> numpy.PyArray_DATA(alpha)
    pbeta = <double*> numpy.PyArray_DATA(beta)
    # call
    res = _compute_transition_counts(pC, pA, ppobs, palpha, pbeta, N, T)
    if res == _BHMM_ERR_NO_MEM:
        raise MemoryError()
    return C

def viterbi(A, pobs, pi):
    N = A.shape[0]
    T = pobs.shape[0]
    # prepare path array
    cdef numpy.ndarray[int, ndim=1, mode="c"] path
    path = numpy.zeros(T, dtype=ctypes.c_int, order='C')

    ppath = <int*> numpy.PyArray_DATA(path)
    pA = <double*> numpy.PyArray_DATA(A)
    ppobs = <double*> numpy.PyArray_DATA(pobs)
    ppi = <double*> numpy.PyArray_DATA(pi)
    # call
    res = _compute_viterbi(ppath, pA, ppobs, ppi, N, T)
    if res == _BHMM_ERR_NO_MEM:
        raise MemoryError()
    return path

def sample_path(alpha, A, pobs, T=None):
    N = pobs.shape[1]
    if T is None:
        T = pobs.shape[0]  # if not set, use the length of pobs as trajectory length
    elif T > pobs.shape[0] or T > alpha.shape[0]:
        raise ValueError('T must be at most the length of pobs and alpha.')
    # prepare path array
    cdef numpy.ndarray[int, ndim=1, mode="c"] path
    path = numpy.zeros(T, dtype=ctypes.c_int, order='C')

    ppath = <int*> numpy.PyArray_DATA(path)
    palpha = <double*> numpy.PyArray_DATA(alpha)
    pA = <double*> numpy.PyArray_DATA(A)
    ppobs = <double*> numpy.PyArray_DATA(pobs)
    # call
    res = _sample_path(ppath, palpha, pA, ppobs, N, T)
    if res == _BHMM_ERR_NO_MEM:
        raise MemoryError()
    return path
