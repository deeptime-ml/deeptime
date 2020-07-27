
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

r"""This module provides dense implementation for the computation of
dynamical fingerprints, expectations and correlations

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>
.. moduleauthor:: M.Scherer <m.scherer AT fu-berlin DOT de>

"""

import numpy as np

from .decomposition import rdl_decomposition, timescales_from_eigenvalues
from .stationary_vector import stationary_distribution as statdist

################################################################################
# Fingerprints
################################################################################


def fingerprint_correlation(P, obs1, obs2=None, tau=1, k=None):
    r"""Dynamical fingerprint for equilibrium correlation experiment.

    The dynamical fingerprint is given by the implied time-scale
    spectrum together with the corresponding amplitudes.

    Parameters
    ----------
    P : (M, M) ndarray
        Transition matrix
    obs1 : (M,) ndarray
        Observable, represented as vector on state space
    obs2 : (M,) ndarray (optional)
        Second observable, for cross-correlations
    tau : int (optional)
        Lag time of given transition matrix, for correct time-scales
    k : int (optional)
        Number of time-scales and amplitudes to compute

    Returns
    -------
    timescales : (N,) ndarray
        Time-scales of the transition matrix
    amplitudes : (N,) ndarray
        Amplitudes for the correlation experiment

    """
    return fingerprint(P, obs1, obs2=obs2, k=k, tau=tau)


def fingerprint_relaxation(P, p0, obs, tau=1, k=None):
    r"""Dynamical fingerprint for relaxation experiment.

    The dynamical fingerprint is given by the implied time-scale
    spectrum together with the corresponding amplitudes.

    Parameters
    ----------
    P : (M, M) ndarray
        Transition matrix
    obs1 : (M,) ndarray
        Observable, represented as vector on state space
    obs2 : (M,) ndarray (optional)
        Second observable, for cross-correlations
    tau : int (optional)
        Lag time of given transition matrix, for correct time-scales
    k : int (optional)
        Number of time-scales and amplitudes to compute

    Returns
    -------
    timescales : (N,) ndarray
        Time-scales of the transition matrix
    amplitudes : (N,) ndarray
        Amplitudes for the relaxation experiment

    """
    one_vec = np.ones(P.shape[0])
    return fingerprint(P, one_vec, obs2=obs, p0=p0, k=k, tau=tau)

def fingerprint(P, obs1, obs2=None, p0=None, tau=1, k=None):
    r"""Dynamical fingerprint for equilibrium or relaxation experiment

    The dynamical fingerprint is given by the implied time-scale
    spectrum together with the corresponding amplitudes.

    Parameters
    ----------
    P : (M, M) ndarray
        Transition matrix
    obs1 : (M,) ndarray
        Observable, represented as vector on state space
    obs2 : (M,) ndarray (optional)
        Second observable, for cross-correlations
    p0 : (M,) ndarray (optional)
        Initial distribution for a relaxation experiment
    tau : int (optional)
        Lag time of given transition matrix, for correct time-scales
    k : int (optional)
        Number of time-scales and amplitudes to compute

    Returns
    -------
    timescales : (N,) ndarray
        Time-scales of the transition matrix
    amplitudes : (N,) ndarray
        Amplitudes for the given observable(s)

    """
    if obs2 is None:
        obs2 = obs1
    R, D, L = rdl_decomposition(P, k=k)
    """Stationary vector"""
    mu = L[0, :]
    """Extract diagonal"""
    w = np.diagonal(D)
    """Compute time-scales"""
    timescales = timescales_from_eigenvalues(w, tau)
    if p0 is None:
        """Use stationary distribution - we can not use only left
        eigenvectors since the system might be non-reversible"""
        amplitudes = np.dot(mu * obs1, R) * np.dot(L, obs2)
    else:
        """Use initial distribution"""
        amplitudes = np.dot(p0 * obs1, R) * np.dot(L, obs2)
    return timescales, amplitudes


################################################################################
# Expectation
################################################################################

def expectation(P, obs):
    r"""Equilibrium expectation of given observable.

    Parameters
    ----------
    P : (M, M) ndarray
        Transition matrix
    obs : (M,) ndarray
        Observable, represented as vector on state space

    Returns
    -------
    x : float
        Expectation value

    """
    pi = statdist(P)
    return np.dot(pi, obs)


################################################################################
# Correlation
################################################################################

def correlation(P, obs1, obs2=None, times=[1], k=None):
    r"""Time-correlation for equilibrium experiment.

    Parameters
    ----------
    P : (M, M) ndarray
        Transition matrix
    obs1 : (M,) ndarray
        Observable, represented as vector on state space
    obs2 : (M,) ndarray (optional)
        Second observable, for cross-correlations
    times : list of int (optional)
        List of times (in tau) at which to compute correlation
    k : int (optional)
        Number of eigenvectors and eigenvalues to use for computation

    Returns
    -------
    correlations : ndarray
        Correlation values at given times

    """
    M = P.shape[0]
    T = np.asarray(times).max()
    if T < M:
        return correlation_matvec(P, obs1, obs2=obs2, times=times)
    else:
        return correlation_decomp(P, obs1, obs2=obs2, times=times, k=k)


def correlation_decomp(P, obs1, obs2=None, times=[1], k=None):
    r"""Time-correlation for equilibrium experiment - via decomposition.

    Parameters
    ----------
    P : (M, M) ndarray
        Transition matrix
    obs1 : (M,) ndarray
        Observable, represented as vector on state space
    obs2 : (M,) ndarray (optional)
        Second observable, for cross-correlations
    times : list of int (optional)
        List of times (in tau) at which to compute correlation
    k : int (optional)
        Number of eigenvalues and eigenvectors to use for computation

    Returns
    -------
    correlations : ndarray
        Correlation values at given times

    """
    if obs2 is None:
        obs2 = obs1
    R, D, L = rdl_decomposition(P, k=k)
    """Stationary vector"""
    mu = L[0, :]
    """Extract eigenvalues"""
    ev = np.diagonal(D)
    """Amplitudes"""
    amplitudes = np.dot(mu * obs1, R) * np.dot(L, obs2)
    """Propgate eigenvalues"""
    times = np.asarray(times)
    ev_t = ev[np.newaxis, :] ** times[:, np.newaxis]
    """Compute result"""
    res = np.dot(ev_t, amplitudes)
    """Truncate imaginary part - should be zero anyways"""
    res = res.real
    return res


def correlation_matvec(P, obs1, obs2=None, times=[1]):
    r"""Time-correlation for equilibrium experiment - via matrix vector products.

    Parameters
    ----------
    P : (M, M) ndarray
        Transition matrix
    obs1 : (M,) ndarray
        Observable, represented as vector on state space
    obs2 : (M,) ndarray (optional)
        Second observable, for cross-correlations
    times : list of int (optional)
        List of times (in tau) at which to compute correlation

    Returns
    -------
    correlations : ndarray
        Correlation values at given times

    """
    if obs2 is None:
        obs2 = obs1

    """Compute stationary vector"""
    mu = statdist(P)
    obs1mu = mu * obs1

    times = np.asarray(times)
    """Sort in increasing order"""
    ind = np.argsort(times)
    times = times[ind]

    if times[0] < 0:
        raise ValueError("Times can not be negative")
    dt = times[1:] - times[0:-1]

    nt = len(times)

    correlations = np.zeros(nt)

    """Propagate obs2 to initial time"""
    obs2_t = 1.0 * obs2
    obs2_t = propagate(P, obs2_t, times[0])
    correlations[0] = np.dot(obs1mu, obs2_t)
    for i in range(nt - 1):
        obs2_t = propagate(P, obs2_t, dt[i])
        correlations[i + 1] = np.dot(obs1mu, obs2_t)

    """Cast back to original order of time points"""
    correlations = correlations[ind]

    return correlations


################################################################################
# Relaxation
################################################################################


def relaxation(P, p0, obs, times=[1], k=None):
    r"""Relaxation experiment.

    The relaxation experiment describes the time-evolution
    of an expectation value starting in a non-equilibrium
    situation.

    Parameters
    ----------
    P : (M, M) ndarray
        Transition matrix
    p0 : (M,) ndarray (optional)
        Initial distribution for a relaxation experiment
    obs : (M,) ndarray
        Observable, represented as vector on state space
    times : list of int (optional)
        List of times at which to compute expectation
    k : int (optional)
        Number of eigenvalues and eigenvectors to use for computation

    Returns
    -------
    res : ndarray
        Array of expectation value at given times

    """
    M = P.shape[0]
    T = np.asarray(times).max()
    if T < M:
        return relaxation_matvec(P, p0, obs, times=times)
    else:
        return relaxation_decomp(P, p0, obs, times=times, k=k)


def relaxation_decomp(P, p0, obs, times=[1], k=None):
    r"""Relaxation experiment.

    The relaxation experiment describes the time-evolution
    of an expectation value starting in a non-equilibrium
    situation.

    Parameters
    ----------
    P : (M, M) ndarray
        Transition matrix
    p0 : (M,) ndarray (optional)
        Initial distribution for a relaxation experiment
    obs : (M,) ndarray
        Observable, represented as vector on state space
    times : list of int (optional)
        List of times at which to compute expectation
    k : int (optional)
        Number of eigenvalues and eigenvectors to use for computation

    Returns
    -------
    res : ndarray
        Array of expectation value at given times

    """
    R, D, L = rdl_decomposition(P, k=k)
    """Extract eigenvalues"""
    ev = np.diagonal(D)
    """Amplitudes"""
    amplitudes = np.dot(p0, R) * np.dot(L, obs)
    """Propgate eigenvalues"""
    times = np.asarray(times)
    ev_t = ev[np.newaxis, :] ** times[:, np.newaxis]
    """Compute result"""
    res = np.dot(ev_t, amplitudes)
    """Truncate imaginary part - should be zero anyways"""
    res = res.real
    return res


def relaxation_matvec(P, p0, obs, times=[1]):
    r"""Relaxation experiment.

    The relaxation experiment describes the time-evolution
    of an expectation value starting in a non-equilibrium
    situation.

    Parameters
    ----------
    P : (M, M) ndarray
        Transition matrix
    p0 : (M,) ndarray (optional)
        Initial distribution for a relaxation experiment
    obs : (M,) ndarray
        Observable, represented as vector on state space
    times : list of int (optional)
        List of times at which to compute expectation

    Returns
    -------
    res : ndarray
        Array of expectation value at given times

    """
    times = np.asarray(times)
    """Sort in increasing order"""
    ind = np.argsort(times)
    times = times[ind]

    if times[0] < 0:
        raise ValueError("Times can not be negative")
    dt = times[1:] - times[0:-1]

    nt = len(times)

    relaxations = np.zeros(nt)

    """Propagate obs to initial time"""
    obs_t = 1.0 * obs
    obs_t = propagate(P, obs_t, times[0])
    relaxations[0] = np.dot(p0, obs_t)
    for i in range(nt - 1):
        obs_t = propagate(P, obs_t, dt[i])
        relaxations[i + 1] = np.dot(p0, obs_t)

    """Cast back to original order of time points"""
    relaxations = relaxations[ind]

    return relaxations


################################################################################
# Helper functions
################################################################################

def propagate(A, x, N):
    r"""Use matrix A to propagate vector x.

    Parameters
    ----------
    A : (M, M) ndarray
        Matrix of propagator
    x : (M, ) ndarray
        Vector to propagate
    N : int
        Number of steps to propagate

    Returns
    -------
    y : (M, ) ndarray
        Propagated vector

    """
    y = 1.0 * x
    for i in range(N):
        y = np.dot(A, y)
    return y
