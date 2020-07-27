
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
Created on Jan 8, 2014

@author: noe
'''

import warnings
import math
import numpy as np
import scipy.sparse
from ..util import types

__all__ = ['transition_matrix_metropolis_1d',
           'generate_traj',
           'generate_trajs']


class MarkovChainSampler(object):
    """
    Class for generation of trajectories from a transition matrix P.
    If many trajectories will be sampled from P, using this class is much more
    efficient than individual calls to generate_traj because that avoid costly
    multiple construction of random variable objects.

    """

    def __init__(self, P, dt=1, random_state=None):
        """
        Constructs a sampling object with transition matrix P. The results will be produced every dt'th time step

        Parameters
        ----------
        P : (n, n) ndarray
            transition matrix
        dt : int
            trajectory will be saved every dt time steps.
            Internally, the dt'th power of P is taken to ensure a more efficient simulation.

        """
        if scipy.sparse.issparse(P):
            warnings.warn("Markov Chain sampler not implemented for sparse matrices. "
                          "Converting transition matrix to dense array")
            P = P.toarray()
        # process input
        if dt > 1:
            # take a power of P if requested
            self.P = np.linalg.matrix_power(P, dt)
        else:
            # create local copy and transform to ndarray if in a different format
            self.P = np.array(P)
        self.n = self.P.shape[0]

        if random_state is None:
            random_state = np.random.RandomState()
        self.random_state = random_state

    def _get_start_state(self):
        # compute mu, the stationary distribution of P
        from ..analysis import stationary_distribution

        mu = stationary_distribution(self.P)
        start = self.random_state.choice(self.n, p=mu)

        return start

    def trajectory(self, N, start=None, stop=None):
        """
        Generates a trajectory realization of length N, starting from state s

        Parameters
        ----------
        N : int
            trajectory length
        start : int, optional, default = None
            starting state. If not given, will sample from the stationary distribution of P
        stop : int or int-array-like, optional, default = None
            stopping set. If given, the trajectory will be stopped before N steps
            once a state of the stop set is reached

        """
        # check input
        stop = types.ensure_int_vector_or_None(stop, require_order=False)

        if start is None:
          start = self._get_start_state()

        # result
        traj = np.zeros(N, dtype=int)
        traj[0] = start
        # already at stopping state?
        if traj[0] == stop:
            return traj[:1]
        # else run until end or stopping state
        for t in range(1, N):
            traj[t] = self.random_state.choice(self.n, p=self.P[traj[t - 1]])
            if traj[t] == stop:
                traj = np.resize(traj, t + 1)
                break

        return traj

    def trajectories(self, M, N, start=None, stop=None):
        """
        Generates M trajectories, each of length N, starting from state s

        Parameters
        ----------
        M : int
            number of trajectories
        N : int
            trajectory length
        start : int, optional, default = None
            starting state. If not given, will sample from the stationary distribution of P
        stop : int or int-array-like, optional, default = None
            stopping set. If given, the trajectory will be stopped before N steps
            once a state of the stop set is reached

        """
        trajs = [self.trajectory(N, start=start, stop=stop) for _ in range(M)]
        return trajs


def generate_traj(P, N, start=None, stop=None, dt=1, random_state=None):
    """
    Generates a realization of the Markov chain with transition matrix P.

    Parameters
    ----------
    P : (n, n) ndarray
        transition matrix
    N : int
        trajectory length
    start : int, optional, default = None
        starting state. If not given, will sample from the stationary distribution of P
    stop : int or int-array-like, optional, default = None
        stopping set. If given, the trajectory will be stopped before N steps
        once a state of the stop set is reached
    dt : int
        trajectory will be saved every dt time steps.
        Internally, the dt'th power of P is taken to ensure a more efficient simulation.
    random_state : None or int or numpy.random.RandomState instance, optional
        This parameter defines the RandomState object to use for drawing random variates.
        If None, the global np.random state is used. If integer, it is used to seed the local RandomState instance.
        Default is None.

    Returns
    -------
    traj_sliced : (N/dt, ) ndarray
        A discrete trajectory with length N/dt

    """
    sampler = MarkovChainSampler(P, dt=dt, random_state=random_state)
    return sampler.trajectory(N, start=start, stop=stop)


def generate_trajs(P, M, N, start=None, stop=None, dt=1, random_state=None):
    """
    Generates multiple realizations of the Markov chain with transition matrix P.

    Parameters
    ----------
    P : (n, n) ndarray
        transition matrix
    M : int
        number of trajectories
    N : int
        trajectory length
    start : int, optional, default = None
        starting state. If not given, will sample from the stationary distribution of P
    stop : int or int-array-like, optional, default = None
        stopping set. If given, the trajectory will be stopped before N steps
        once a state of the stop set is reached
    dt : int
        trajectory will be saved every dt time steps.
        Internally, the dt'th power of P is taken to ensure a more efficient simulation.
    random_state : None or int or numpy.random.RandomState instance, optional
        This parameter defines the RandomState object to use for drawing random variates.
        If None, the global np.random state is used. If integer, it is used to seed the local RandomState instance.
        Default is None.

    Returns
    -------
    traj_sliced : (N/dt, ) ndarray
        A discrete trajectory with length N/dt

    """
    sampler = MarkovChainSampler(P, dt=dt, random_state=random_state)
    return sampler.trajectories(M, N, start=start, stop=stop)


def transition_matrix_metropolis_1d(E, d=1.0):
    r"""Transition matrix describing the Metropolis chain jumping
    between neighbors in a discrete 1D energy landscape.

    Parameters
    ----------
    E : (M,) ndarray
        Energies in units of kT
    d : float (optional)
        Diffusivity of the chain, d in (0, 1]

    Returns
    -------
    P : (M, M) ndarray
        Transition matrix of the Markov chain

    Notes
    -----
    Transition probabilities are computed as
    .. math::
        p_{i,i-1} &=& 0.5 d \min \left{ 1.0, \mathrm{e}^{-(E_{i-1} - E_i)} \right}, \\
        p_{i,i+1} &=& 0.5 d \min \left{ 1.0, \mathrm{e}^{-(E_{i+1} - E_i)} \right}, \\
        p_{i,i}   &=& 1.0 - p_{i,i-1} - p_{i,i+1}.

    """
    # check input
    if d <= 0 or d > 1:
        raise ValueError('Diffusivity must be in (0,1]. Trying to set the invalid value', str(d))
    # init
    n = len(E)
    P = np.zeros((n, n))
    # set off diagonals
    P[0, 1] = 0.5 * d * min(1.0, math.exp(-(E[1] - E[0])))
    for i in range(1, n - 1):
        P[i, i - 1] = 0.5 * d * min(1.0, math.exp(-(E[i - 1] - E[i])))
        P[i, i + 1] = 0.5 * d * min(1.0, math.exp(-(E[i + 1] - E[i])))
    P[n - 1, n - 2] = 0.5 * d * min(1.0, math.exp(-(E[n - 2] - E[n - 1])))
    # normalize
    P += np.diag(1.0 - np.sum(P, axis=1))
    # done
    return P
