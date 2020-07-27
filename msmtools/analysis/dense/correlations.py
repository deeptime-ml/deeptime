
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

'''
Created on 29.11.2013

.. moduleauthor:: marscher, noe
'''

import numpy as np

from .decomposition import rdl_decomposition


def time_correlation_by_diagonalization(P, pi, obs1, obs2=None, time=1, rdl=None):
    """
    calculates time correlation. Raises P to power 'times' by diagonalization.
    If rdl tuple (R, D, L) is given, it will be used for
    further calculation.
    """
    if rdl is None:
        raise ValueError("no rdl decomposition")
    R, D, L = rdl

    d_times = np.diag(D) ** time
    diag_inds = np.diag_indices_from(D)
    D_time = np.zeros(D.shape, dtype=d_times.dtype)
    D_time[diag_inds] = d_times
    P_time = np.dot(np.dot(R, D_time), L)

    # multiply element-wise obs1 and pi. this is obs1' diag(pi)
    l = np.multiply(obs1, pi)
    m = np.dot(P_time, obs2)
    result = np.dot(l, m)
    return result


def time_correlation_direct_by_mtx_vec_prod(P, mu, obs1, obs2=None, time=1, start_values=None, return_P_k_obs=False):
    r"""Compute time-correlation of obs1, or time-cross-correlation with obs2.

    The time-correlation at time=k is computed by the matrix-vector expression:
    cor(k) = obs1' diag(pi) P^k obs2


    Parameters
    ----------
    P : ndarray, shape=(n, n) or scipy.sparse matrix
        Transition matrix
    obs1 : ndarray, shape=(n)
        Vector representing observable 1 on discrete states
    obs2 : ndarray, shape=(n)
        Vector representing observable 2 on discrete states. If not given,
        the autocorrelation of obs1 will be computed
    mu : ndarray, shape=(n)
        stationary distribution vector.
    time : int
        time point at which the (auto)correlation will be evaluated.
    start_values : (time, ndarray <P, <P, obs2>>_t)
        start iteration of calculation of matrix power product, with this values.
        only useful when calling this function out of a loop over times.
    return_P_k_obs : bool
        if True, the dot product <P^time, obs2> will be returned for further
        calculations.

    Returns
    -------
    cor(k) : float
           correlation between observations
    """
    # input checks
    if not (type(time) == int):
        if not (type(time) == np.int64):
            raise TypeError("given time (%s) is not an integer, but has type: %s"
                            % (str(time), type(time)))
    if obs1.shape[0] != P.shape[0]:
        raise ValueError("observable shape not compatible with given matrix")
    if obs2 is None:
        obs2 = obs1
    # multiply element-wise obs1 and pi. this is obs1' diag(pi)
    l = np.multiply(obs1, mu)
    # raise transition matrix to power of time by substituting dot product
    # <Pk, obs2> with something like <P, <P, obs2>>.
    # This saves a lot of matrix matrix multiplications.
    if start_values:  # begin with a previous calculated val
        P_i_obs = start_values[1]
        # calculate difference properly!
        time_prev = start_values[0]
        t_diff = time - time_prev
        r = range(t_diff)
    else:
        if time >= 2:
            P_i_obs = np.dot(P, np.dot(P, obs2))  # vector <P, <P, obs2> := P^2 * obs
            r = range(time - 2)
        elif time == 1:
            P_i_obs = np.dot(P, obs2)  # P^1 = P*obs
            r = range(0)
        elif time == 0:  # P^0 = I => I*obs2 = obs2
            P_i_obs = obs2
            r = range(0)

    for k in r:  # since we already substituted started with 0
        P_i_obs = np.dot(P, P_i_obs)
    corr = np.dot(l, P_i_obs)
    if return_P_k_obs:
        return corr, (time, P_i_obs)
    else:
        return corr


def time_correlations_direct(P, pi, obs1, obs2=None, times=[1]):
    r"""Compute time-correlations of obs1, or time-cross-correlation with obs2.

    The time-correlation at time=k is computed by the matrix-vector expression:
    cor(k) = obs1' diag(pi) P^k obs2


    Parameters
    ----------
    P : ndarray, shape=(n, n) or scipy.sparse matrix
        Transition matrix
    obs1 : ndarray, shape=(n)
        Vector representing observable 1 on discrete states
    obs2 : ndarray, shape=(n)
        Vector representing observable 2 on discrete states. If not given,
        the autocorrelation of obs1 will be computed
    pi : ndarray, shape=(n)
        stationary distribution vector. Will be computed if not given
    times : array-like, shape(n_t)
        Vector of time points at which the (auto)correlation will be evaluated

    Returns
    -------

    """
    n_t = len(times)
    times = np.sort(times)  # sort it to use caching of previously computed correlations
    f = np.zeros(n_t)

    # maximum time > number of rows?
    if times[-1] > P.shape[0]:
        use_diagonalization = True
        R, D, L = rdl_decomposition(P)
        # discard imaginary part, if all elements i=0
        if not np.any(np.iscomplex(R)):
            R = np.real(R)
        if not np.any(np.iscomplex(D)):
            D = np.real(D)
        if not np.any(np.iscomplex(L)):
            L = np.real(L)
        rdl = (R, D, L)

    if use_diagonalization:
        for i in range(n_t):
            f[i] = time_correlation_by_diagonalization(P, pi, obs1, obs2, times[i], rdl)
    else:
        start_values = None
        for i in range(n_t):
            f[i], start_values = \
                time_correlation_direct_by_mtx_vec_prod(P, pi, obs1, obs2,
                                                        times[i], start_values, True)
    return f


def time_relaxation_direct_by_mtx_vec_prod(P, p0, obs, time=1, start_values=None, return_pP_k=False):
    r"""Compute time-relaxations of obs with respect of given initial distribution.

    relaxation(k) = p0 P^k obs

    Parameters
    ----------
    P : ndarray, shape=(n, n) or scipy.sparse matrix
        Transition matrix
    p0 : ndarray, shape=(n)
        initial distribution
    obs : ndarray, shape=(n)
        Vector representing observable on discrete states.
    time : int or array like
        time point at which the (auto)correlation will be evaluated.

    start_values = (time,
    Returns
    -------
    relaxation : float
    """
    # input checks
    if not type(time) == int:
        if not type(time) == np.int64:
            raise TypeError("given time (%s) is not an integer, but has type: %s"
                            % (str(time), type(time)))
    if obs.shape[0] != P.shape[0]:
        raise ValueError("observable shape not compatible with given matrix")
    if p0.shape[0] != P.shape[0]:
        raise ValueError("shape of init dist p0 (%s) not compatible with given matrix (shape=%s)"
                         % (p0.shape[0], P.shape))
    # propagate in time
    if start_values:  # begin with a previous calculated val
        pk_i = start_values[1]
        time_prev = start_values[0]
        t_diff = time - time_prev
        r = range(t_diff)
    else:
        if time >= 2:
            pk_i = np.dot(np.dot(p0, P), P)  # pk_2
            r = range(time - 2)
        elif time == 1:
            pk_i = np.dot(p0, P)  # propagate once
            r = range(0)
        elif time == 0:  # P^0 = I => p0*I = p0
            pk_i = p0
            r = range(0)

    for k in r:  # perform the rest of the propagations p0 P^t_diff
        pk_i = np.dot(pk_i, P)

    # result
    l = np.dot(pk_i, obs)
    if return_pP_k:
        return l, (time, pk_i)
    else:
        return l


def time_relaxation_direct_by_diagonalization(P, p0, obs, time, rdl=None):
    if rdl is None:
        raise ValueError("no rdl decomposition")
    R, D, L = rdl

    d_times = np.diag(D) ** time
    diag_inds = np.diag_indices_from(D)
    D_time = np.zeros(D.shape, dtype=d_times.dtype)
    D_time[diag_inds] = d_times
    P_time = np.dot(np.dot(R, D_time), L)

    result = np.dot(np.dot(p0, P_time), obs)
    return result


def time_relaxations_direct(P, p0, obs, times=[1]):
    r"""Compute time-relaxations of obs with respect of given initial distribution.

    relaxation(k) = p0 P^k obs

    Parameters
    ----------
    P : ndarray, shape=(n, n) or scipy.sparse matrix
        Transition matrix
    p0 : ndarray, shape=(n)
        initial distribution
    obs : ndarray, shape=(n)
        Vector representing observable on discrete states.
    times : array-like, shape(n_t)
        Vector of time points at which the (auto)correlation will be evaluated

    Returns
    -------
    relaxations : ndarray, shape(n_t)
    """
    n_t = len(times)
    times = np.sort(times)

    # maximum time > number of rows?
    if times[-1] > P.shape[0]:
        use_diagonalization = True
        R, D, L = rdl_decomposition(P)
        # discard imaginary part, if all elements i=0
        if not np.any(np.iscomplex(R)):
            R = np.real(R)
        if not np.any(np.iscomplex(D)):
            D = np.real(D)
        if not np.any(np.iscomplex(L)):
            L = np.real(L)
        rdl = (R, D, L)

    f = np.empty(n_t, dtype=D.dtype)

    if use_diagonalization:
        for i in range(n_t):
            f[i] = time_relaxation_direct_by_diagonalization(P, p0, obs, times[i], rdl)
    else:
        start_values = None
        for i in range(n_t):
            f[i], start_values = time_relaxation_direct_by_mtx_vec_prod(P, p0, obs, times[i], start_values, True)
    return f
