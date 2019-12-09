
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

"""Python implementation of Hidden Markov Model kernel functions

This module is considered to be the reference for checking correctness of other
kernels. All implementations are based on paper Rabiners [1].

.. [1] Lawrence R. Rabiner, "A Tutorial on Hidden Markov Models and
   Selected Applications in Speech Recognition", Proceedings of the IEEE,
   vol. 77, issue 2
"""
import numpy as np

from . import hidden as _impl


def forward(A, pobs, pi, T=None, alpha=None):
    """Compute P( obs | A, B, pi ) and all forward coefficients.

    Parameters
    ----------
    A : ndarray((N,N), dtype = float)
        transition matrix of the hidden states
    pobs : ndarray((T,N), dtype = float)
        pobs[t,i] is the observation probability for observation at time t given hidden state i
    pi : ndarray((N), dtype = float)
        initial distribution of hidden states
    T : int, optional, default = None
        trajectory length. If not given, T = pobs.shape[0] will be used.
    alpha : ndarray((T,N), dtype = float), optional, default = None
        container for the alpha result variables. If None, a new container will be created.

    Returns
    -------
    logprob : float
        The probability to observe the sequence `ob` with the model given
        by `A`, `B` and `pi`.
    alpha : ndarray((T,N), dtype = float), optional, default = None
        alpha[t,i] is the ith forward coefficient of time t. These can be
        used in many different algorithms related to HMMs.

    """
    if T is None:
        T = len(pobs)  # if not set, use the length of pobs as trajectory length
    elif T > len(pobs):
        raise TypeError('T must be at most the length of pobs.')
    N = len(A)
    if alpha is None:
        alpha = np.zeros_like(pobs)
    elif T > len(alpha):
        raise TypeError('alpha must at least have length T in order to fit trajectory.')

    return _impl.forward(A, pobs, pi, alpha, T, N)


def backward(A, pobs, T=None, beta_out=None):
    """Compute all backward coefficients. With scaling!

    Parameters
    ----------
    A : ndarray((N,N), dtype = float)
        transition matrix of the hidden states
    pobs : ndarray((T,N), dtype = float)
        pobs[t,i] is the observation probability for observation at time t given hidden state i
    T : int, optional, default = None
        trajectory length. If not given, T = pobs.shape[0] will be used.
    beta_out : ndarray((T,N), dtype = float), optional, default = None
        container for the beta result variables. If None, a new container will be created.

    Returns
    -------
    beta : ndarray((T,N), dtype = float), optional, default = None
        beta[t,i] is the ith backward coefficient of time t. These can be
        used in many different algorithms related to HMMs.

    """
    if T is None:
        T = len(pobs)  # if not set, use the length of pobs as trajectory length
    elif T > len(pobs):
        raise ValueError('T must be at most the length of pobs.')
    N = len(A)
    if beta_out is None:
        beta_out = np.zeros_like(pobs)
    elif T > len(beta_out):
        raise ValueError('beta_out must at least have length T in order to fit trajectory.')

    return _impl.backward(A, pobs, T=T, N=N, beta=beta_out)


def state_probabilities(alpha, beta, T=None, gamma_out=None):
    """ Calculate the (T,N)-probability matrix for being in state i at time t.

    Parameters
    ----------
    alpha : ndarray((T,N), dtype = float), optional, default = None
        alpha[t,i] is the ith forward coefficient of time t.
    beta : ndarray((T,N), dtype = float), optional, default = None
        beta[t,i] is the ith forward coefficient of time t.
    T : int, optional, default = None
        trajectory length. If not given, gamma_out.shape[0] will be used. If
        gamma_out is neither given, T = alpha.shape[0] will be used.
    gamma_out : ndarray((T,N), dtype = float), optional, default = None
        container for the gamma result variables. If None, a new container will be created.

    Returns
    -------
    gamma : ndarray((T,N), dtype = float), optional, default = None
        gamma[t,i] is the probability at time t to be in state i !


    See Also
    --------
    forward : to calculate `alpha`
    backward : to calculate `beta`

    """
    # get summation helper - we use matrix multiplication with 1's because it's faster than the np.sum function (yes!)
    if state_probabilities.ones_size != alpha.shape[1]:
        state_probabilities.ones = np.ones(alpha.shape[1])[:, None]
        state_probabilities.ones_size = alpha.shape[1]

    if len(alpha) != len(beta):
        raise ValueError('Inconsistent sizes of alpha and beta.')
    # determine T to use
    if T is None:
        if gamma_out is None:
            T = len(alpha)
        else:
            T = len(gamma_out)
    # compute
    if gamma_out is None:
        gamma_out = alpha * beta
        if T < len(gamma_out):
            gamma_out = gamma_out[:T]
    else:
        if len(gamma_out) < len(alpha):
            np.multiply(alpha[:T], beta[:T], gamma_out)
        else:
            np.multiply(alpha, beta, gamma_out)
    # normalize
    np.divide(gamma_out, np.dot(gamma_out, state_probabilities.ones), out=gamma_out)
    # done
    return gamma_out


# function singletons as little helpers
state_probabilities.ones = None
state_probabilities.ones_size = 0


def transition_counts(alpha, beta, A, pobs, T=None, out=None):
    """ Sum for all t the probability to transition from state i to state j.

    Parameters
    ----------
    alpha : ndarray((T,N), dtype = float), optional, default = None
        alpha[t,i] is the ith forward coefficient of time t.
    beta : ndarray((T,N), dtype = float), optional, default = None
        beta[t,i] is the ith forward coefficient of time t.
    A : ndarray((N,N), dtype = float)
        transition matrix of the hidden states
    pobs : ndarray((T,N), dtype = float)
        pobs[t,i] is the observation probability for observation at time t given hidden state i
    T : int
        number of time steps
    out : ndarray((N,N), dtype = float), optional, default = None
        container for the resulting count matrix. If None, a new matrix will be created.

    Returns
    -------
    counts : numpy.array shape (N, N)
         counts[i, j] is the summed probability to transition from i to j in time [0,T)

    See Also
    --------
    forward : calculate forward coefficients `alpha`
    backward : calculate backward coefficients `beta`

    """
    if T is None:
        T = pobs.shape[0]  # if not set, use the length of pobs as trajectory length
    elif T > pobs.shape[0]:
        raise ValueError('T must be at most the length of pobs.')
    N = len(A)
    if out is None:
        C = np.zeros_like(A)
    else:
        C = out

    return _impl.transition_counts(alpha, beta, A, pobs, T=T, N=N, C=C)


def viterbi(A, pobs, pi):
    """ Estimate the hidden pathway of maximum likelihood using the Viterbi algorithm.

    Parameters
    ----------
    A : ndarray((N,N), dtype = float)
        transition matrix of the hidden states
    pobs : ndarray((T,N), dtype = float)
        pobs[t,i] is the observation probability for observation at time t given hidden state i
    pi : ndarray((N), dtype = float)
        initial distribution of hidden states

    Returns
    -------
    q : numpy.array shape (T)
        maximum likelihood hidden path

    """
    # prepare path array
    T = len(pobs)
    N = len(A)
    path = np.zeros(T, dtype=np.int32)

    return _impl.viterbi(A, pobs, pi, path, T, N)


def sample_path(alpha, A, pobs, T=None, seed=-1):
    """ Sample the hidden pathway S from the conditional distribution P ( S | Parameters, Observations )

    Parameters
    ----------
    alpha : ndarray((T,N), dtype = float), optional, default = None
        alpha[t,i] is the ith forward coefficient of time t.
    A : ndarray((N,N), dtype = float)
        transition matrix of the hidden states
    pobs : ndarray((T,N), dtype = float)
        pobs[t,i] is the observation probability for observation at time t given hidden state i
    T : int
        number of time steps
    seed: int, default = -1
        seed for libc.stdlib.srand, if -1 is given, the random number generation will be initialized
        with the current time (seconds).

    Returns
    -------
    S : numpy.array shape (T)
        maximum likelihood hidden path

    """
    N = pobs.shape[1]
    if T is None:
        T = pobs.shape[0]  # if not set, use the length of pobs as trajectory length
    elif T > pobs.shape[0] or T > alpha.shape[0]:
        raise ValueError('T must be at most the length of pobs and alpha.')
    path = np.zeros(T, dtype=np.int32)

    return _impl.sample_path(alpha, A, pobs, path, T, N, seed)
