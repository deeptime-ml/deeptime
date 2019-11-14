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

cimport numpy as np

cdef extern from "_hidden.h":
    double _forward(double *alpha, const double *A, const double *pobs, const double *pi, const int N, const int T) nogil
    void _backward(double *beta, const double *A, const double *pobs, const int N, const int T) nogil
    int _compute_transition_counts(double *transition_counts, const double *A, const double *pobs, const double *alpha,
                                   const double *beta, int N, int T) nogil
    int _compute_viterbi(int *path, const double *A, const double *pobs, const double *pi, int N, int T) nogil
    int _sample_path(int *path, const double *alpha, const double *A, const double *pobs, const int N, const int T) nogil
    int _BHMM_ERR_NO_MEM


def forward(A, pobs, pi, alpha, int T, int N):
    palpha = <double*> np.PyArray_DATA(alpha)
    pA = <double*> np.PyArray_DATA(A)
    ppobs = <double*> np.PyArray_DATA(pobs)
    ppi = <double*> np.PyArray_DATA(pi)

    with nogil:
        logprob = _forward(palpha, pA, ppobs, ppi, N, T)
    return logprob, alpha


def backward(A, pobs, beta, int T, int N):
    pbeta = <double*> np.PyArray_DATA(beta)
    pA = <double*> np.PyArray_DATA(A)
    ppobs = <double*> np.PyArray_DATA(pobs)
    # call
    with nogil:
        _backward(pbeta, pA, ppobs, N, T)
    return beta


def transition_counts(alpha, beta, A, pobs, int T, int N, C):
    pC = <double*> np.PyArray_DATA(C)
    pA = <double*> np.PyArray_DATA(A)
    ppobs = <double*> np.PyArray_DATA(pobs)
    palpha = <double*> np.PyArray_DATA(alpha)
    pbeta = <double*> np.PyArray_DATA(beta)
    # call
    with nogil:
        res = _compute_transition_counts(pC, pA, ppobs, palpha, pbeta, N, T)
    if res == _BHMM_ERR_NO_MEM:
        raise MemoryError()
    return C

def viterbi(A, pobs, pi, path, int T, int N):
    ppath = <int*> np.PyArray_DATA(path)
    pA = <double*> np.PyArray_DATA(A)
    ppobs = <double*> np.PyArray_DATA(pobs)
    ppi = <double*> np.PyArray_DATA(pi)

    with nogil:
        res = _compute_viterbi(ppath, pA, ppobs, ppi, N, T)
    if res == _BHMM_ERR_NO_MEM:
        raise MemoryError()
    return path

def sample_path(alpha, A, pobs, path, int T, int N):
    ppath = <int*> np.PyArray_DATA(path)
    palpha = <double*> np.PyArray_DATA(alpha)
    pA = <double*> np.PyArray_DATA(A)
    ppobs = <double*> np.PyArray_DATA(pobs)

    with nogil:
        res = _sample_path(ppath, palpha, pA, ppobs, N, T)
    if res == _BHMM_ERR_NO_MEM:
        raise MemoryError()
    return path
