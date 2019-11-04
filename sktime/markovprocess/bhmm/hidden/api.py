
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

from bhmm.hidden import impl_python as ip
from bhmm.hidden import impl_c as ic
from bhmm.util import config


# implementation codes
__IMPL_PYTHON__ = 0
__IMPL_C__ = 1

# implementation used
__impl__ = __IMPL_PYTHON__


def set_implementation(impl):
    """
    Sets the implementation of this module

    Parameters
    ----------
    impl : str
        One of ["python", "c"]

    """
    global __impl__
    if impl.lower() == 'python':
        __impl__ = __IMPL_PYTHON__
    elif impl.lower() == 'c':
        __impl__ = __IMPL_C__
    else:
        import warnings
        warnings.warn('Implementation '+impl+' is not known. Using the fallback python implementation.')
        __impl__ = __IMPL_PYTHON__


def forward(A, pobs, pi, T=None, alpha_out=None):
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
    alpha_out : ndarray((T,N), dtype = float), optional, default = None
        containter for the alpha result variables. If None, a new container will be created.

    Returns
    -------
    logprob : float
        The probability to observe the sequence `ob` with the model given
        by `A`, `B` and `pi`.
    alpha : ndarray((T,N), dtype = float), optional, default = None
        alpha[t,i] is the ith forward coefficient of time t. These can be
        used in many different algorithms related to HMMs.

    """
    if __impl__ == __IMPL_PYTHON__:
        return ip.forward(A, pobs, pi, T=T, alpha_out=alpha_out, dtype=config.dtype)
    elif __impl__ == __IMPL_C__:
        return ic.forward(A, pobs, pi, T=T, alpha_out=alpha_out, dtype=config.dtype)
    else:
        raise RuntimeError('Nonexisting implementation selected: '+str(__impl__))


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
        containter for the beta result variables. If None, a new container will be created.

    Returns
    -------
    beta : ndarray((T,N), dtype = float), optional, default = None
        beta[t,i] is the ith backward coefficient of time t. These can be
        used in many different algorithms related to HMMs.

    """
    if __impl__ == __IMPL_PYTHON__:
        return ip.backward(A, pobs, T=T, beta_out=beta_out, dtype=config.dtype)
    elif __impl__ == __IMPL_C__:
        return ic.backward(A, pobs, T=T, beta_out=beta_out, dtype=config.dtype)
    else:
        raise RuntimeError('Nonexisting implementation selected: '+str(__impl__))


# global singletons as little helpers
ones = None
ones_size = 0


def state_probabilities(alpha, beta, T=None, gamma_out=None):
    """ Calculate the (T,N)-probabilty matrix for being in state i at time t.

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
        containter for the gamma result variables. If None, a new container will be created.

    Returns
    -------
    gamma : ndarray((T,N), dtype = float), optional, default = None
        gamma[t,i] is the probabilty at time t to be in state i !


    See Also
    --------
    forward : to calculate `alpha`
    backward : to calculate `beta`

    """
    # get summation helper - we use matrix multiplication with 1's because it's faster than the np.sum function (yes!)
    global ones_size
    if ones_size != alpha.shape[1]:
        global ones
        ones = np.ones(alpha.shape[1])[:, None]
        ones_size = alpha.shape[1]
    #
    if alpha.shape[0] != beta.shape[0]:
        raise ValueError('Inconsistent sizes of alpha and beta.')
    # determine T to use
    if T is None:
        if gamma_out is None:
            T = alpha.shape[0]
        else:
            T = gamma_out.shape[0]
    # compute
    if gamma_out is None:
        gamma_out = alpha * beta
        if T < gamma_out.shape[0]:
            gamma_out = gamma_out[:T]
    else:
        if gamma_out.shape[0] < alpha.shape[0]:
            np.multiply(alpha[:T], beta[:T], gamma_out)
        else:
            np.multiply(alpha, beta, gamma_out)
    # normalize
    np.divide(gamma_out, np.dot(gamma_out, ones), out=gamma_out)
    # done
    return gamma_out


def state_counts(gamma, T, out=None):
    """ Sum the probabilities of being in state i to time t

    Parameters
    ----------
    gamma : ndarray((T,N), dtype = float), optional, default = None
        gamma[t,i] is the probabilty at time t to be in state i !
    T : int
        number of time steps

    Returns
    -------
    count : numpy.array shape (N)
            count[i] is the summed probabilty to be in state i !

    See Also
    --------
    state_probabilities : to calculate `gamma`

    """
    return np.sum(gamma[0:T], axis=0, out=out)


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
        containter for the resulting count matrix. If None, a new matrix will be created.

    Returns
    -------
    counts : numpy.array shape (N, N)
         counts[i, j] is the summed probability to transition from i to j in time [0,T)

    See Also
    --------
    forward : calculate forward coefficients `alpha`
    backward : calculate backward coefficients `beta`

    """
    if __impl__ == __IMPL_PYTHON__:
        return ip.transition_counts(alpha, beta, A, pobs, T=T, out=out, dtype=config.dtype)
    elif __impl__ == __IMPL_C__:
        return ic.transition_counts(alpha, beta, A, pobs, T=T, out=out, dtype=config.dtype)
    else:
        raise RuntimeError('Nonexisting implementation selected: '+str(__impl__))


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
    if __impl__ == __IMPL_PYTHON__:
        return ip.viterbi(A, pobs, pi, dtype=config.dtype)
    elif __impl__ == __IMPL_C__:
        return ic.viterbi(A, pobs, pi, dtype=config.dtype)
    else:
        raise RuntimeError('Nonexisting implementation selected: '+str(__impl__))


def sample_path(alpha, A, pobs, T=None):
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

    Returns
    -------
    S : numpy.array shape (T)
        maximum likelihood hidden path

    """
    if __impl__ == __IMPL_PYTHON__:
        return ip.sample_path(alpha, A, pobs, T=T, dtype=config.dtype)
    elif __impl__ == __IMPL_C__:
        return ic.sample_path(alpha, A, pobs, T=T, dtype=config.dtype)
    else:
        raise RuntimeError('Nonexisting implementation selected: '+str(__impl__))
