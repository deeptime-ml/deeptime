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

import msmtools.estimation as msmest
import numpy as np

from sktime.base import Model
from sktime.markovprocess.bhmm import hidden
from sktime.markovprocess.bhmm.estimators import _tmatrix_disconnected
# TODO: this seems somehow duplicated from pyemma.msm.HMM class + some extra features.
from sktime.markovprocess.bhmm.output_models.outputmodel import OutputModel


class HMM(Model):
    r""" Hidden Markov model (HMM).

    This class is used to represent an HMM. This could be a maximum-likelihood HMM or a sampled HMM from a
    Bayesian posterior.

    Parameters
    ----------
    transition_matrix : np.array with shape (n_states, n_states), optional, default=None
        Row-stochastic transition matrix among states.
    output_model : :class:`sktime.markovprocess.bhmm.OutputModel`
        The output model for the states.
    lag : int, optional, default=1
        Lag time (optional) used to estimate the HMM. Used to compute relaxation timescales.
    initial_distribution : np.array with shape (n_states), optional, default=None
        The initial state vector. Required when stationary=False

    Examples
    --------

    >>> # Gaussian HMM
    >>> n_states = 2
    >>> pi = np.array([0.5, 0.5])
    >>> Tij = np.array([[0.8, 0.2], [0.5, 0.5]])
    >>> from sktime.markovprocess.bhmm.output_models import GaussianOutputModel
    >>> output_model = GaussianOutputModel(n_states, means=[-1, +1], sigmas=[1, 1])
    >>> model = HMM(pi, Tij, output_model)

    >>> # Discrete HMM
    >>> n_states = 2
    >>> pi = np.array([0.5, 0.5])
    >>> Tij = np.array([[0.8, 0.2], [0.5, 0.5]])
    >>> from sktime.markovprocess.bhmm.output_models import DiscreteOutputModel
    >>> output_model = DiscreteOutputModel([[0.5, 0.1, 0.4], [0.2, 0.3, 0.5]])
    >>> model = HMM(pi, Tij, output_model)

    """

    def __init__(self, initial_distribution=None, transition_matrix=None, output_model=None, lag=1):
        self._stationary = None
        self._likelihoods = None
        self._lag = lag
        self._output_model = output_model
        self.hidden_state_trajectories = None
        self._gammas = None

        if initial_distribution is not None and transition_matrix is not None:
            # update numbers
            self.update(initial_distribution, transition_matrix)

    def update(self, Pi, Tij):
        r""" Updates the transition matrix and recomputes all derived quantities """
        # update transition matrix by copy
        # TODO: why copy here?
        self._Tij = np.array(Tij)
        # set number of states
        self._n_states = len(Tij)
        # assert is_transition_matrix(self._Tij), 'Given transition matrix is not a stochastic matrix'
        # assert self._Tij.shape[0] == self.n_states, 'Given transition matrix has unexpected number of states '
        # reset spectral decomposition
        self._spectral_decomp_available = False

        # check initial distribution
        # assert np.all(Pi >= 0), 'Given initial distribution contains negative elements.'
        # assert np.any(Pi > 0), 'Given initial distribution is zero'
        self._Pi = np.array(Pi) / np.sum(Pi)  # ensure normalization and make a copy
        # TODO: why copy here?
        #self._Pi = Pi / np.sum(Pi)

    def _do_spectral_decomposition(self):
        self._R, self._D, self._L = _tmatrix_disconnected.rdl_decomposition(self._Tij, reversible=self.reversible)
        self._eigenvalues = np.diag(self._D)
        self._spectral_decomp_available = True

    def _ensure_spectral_decomposition(self):
        """
        """
        if not self._spectral_decomp_available:
            self._do_spectral_decomposition()

    @property
    def lag(self):
        """ Lag time of the model, i.e. the number of observed trajectory steps made by the transition matrix """
        return self._lag

    @property
    def is_strongly_connected(self):
        """ Whether the HMM transition matrix is strongly connected """
        return msmest.is_connected(self._Tij, directed=True)

    @property
    def strongly_connected_sets(self):
        return msmest.connected_sets(self._Tij, directed=True)

    @property
    def is_weakly_connected(self):
        """ Whether the HMM transition matrix is weakly connected """
        return msmest.is_connected(self._Tij, directed=False)

    @property
    def weakly_connected_sets(self):
        return msmest.connected_sets(self._Tij, directed=False)

    @property
    def reversible(self):
        """ Whether the HMM is reversible """
        return _tmatrix_disconnected.is_reversible(self._Tij)

    @property
    def is_stationary(self):
        """ Whether the MSM is stationary, i.e. whether the initial distribution is the stationary distribution
         of the hidden transition matrix. """
        # for disconnected matrices, the stationary distribution depends on the estimator, so we can't compute
        # it directly. Therefore we test whether the initial distribution is stationary.
        return np.allclose(np.dot(self._Pi, self._Tij), self._Pi)

    @property
    def n_states(self):
        r""" The number of hidden states """
        return self._n_states

    @property
    def initial_distribution(self):
        r""" The initial distribution of the hidden states """
        return self._Pi

    @property
    def stationary_distribution(self):
        r""" Compute stationary distribution of hidden states if possible.

        Raises
        ------
        ValueError if the HMM is not stationary

        """
        from msmtools.analysis import is_connected, stationary_distribution
        if not is_connected(self.transition_matrix, directed=False):
            raise RuntimeError('No unique stationary distribution because transition matrix is not connected')
        return stationary_distribution(self._Tij)

    @property
    def transition_matrix(self):
        r""" The hidden transition matrix """
        return self._Tij

    @property
    def eigenvalues(self):
        r""" Hidden transition matrix eigenvalues

        Returns
        -------
        ts : ndarray(m)
            transition matrix eigenvalues :math:`\lambda_i, i = 1,...,k`., sorted by descending norm.

        """
        self._ensure_spectral_decomposition()
        return self._eigenvalues

    @property
    def eigenvectors_left(self):
        r""" Left eigenvectors of the hidden transition matrix

        Returns
        -------
        L : ndarray(n_states,n_states)
            left eigenvectors in a row matrix. l_ij is the j'th component of the i'th left eigenvector

        """
        self._ensure_spectral_decomposition()
        return self._L

    @property
    def eigenvectors_right(self):
        r""" Right eigenvectors of the hidden transition matrix

        Returns
        -------
        R : ndarray(n_states,n_states)
            right eigenvectors in a column matrix. r_ij is the i'th component of the j'th right eigenvector

        """
        self._ensure_spectral_decomposition()
        return self._R

    @property
    def timescales(self):
        r""" Relaxation timescales of the hidden transition matrix

        Returns
        -------
        ts : ndarray(m)
            relaxation timescales in units of the input trajectory time step,
            defined by :math:`-tau / ln | \lambda_i |, i = 2,...,n_states`, where
            :math:`\lambda_i` are the hidden transition matrix eigenvalues.

        """
        from msmtools.analysis.dense.decomposition import timescales_from_eigenvalues as ts

        self._ensure_spectral_decomposition()
        ts = ts(self._eigenvalues, tau=self._lag)
        return ts[1:]

    @property
    def lifetimes(self):
        r""" Lifetimes of states of the hidden transition matrix

        Returns
        -------
        l : ndarray(n_states)
            state lifetimes in units of the input trajectory time step,
            defined by :math:`-tau / ln | p_{ii} |, i = 1,...,n_states`, where
            :math:`p_{ii}` are the diagonal entries of the hidden transition matrix.

        """
        return -self._lag / np.log(np.diag(self.transition_matrix))

    @property
    def likelihood(self):
        r""" Estimated HMM likelihood """
        return self._likelihoods[-1]

    @property
    def likelihoods(self):
        r""" Sequence of likelihoods generated from the iteration """
        return self._likelihoods

    @property
    def hidden_state_probabilities(self):
        r""" Probabilities of hidden states at every trajectory and time point """
        return self._gammas

    @property
    def output_model(self) -> OutputModel:
        r""" The HMM output model """
        return self._output_model

    @property
    def initial_probability(self):
        r""" Initial probability """
        return self._Pi

    @property
    def stationary_probability(self):
        r""" Stationary probability, if the model is stationary """
        assert self._stationary, 'Estimator is not stationary'
        return self._Pi

    def sub_hmm(self, states):
        r""" Returns HMM on a subset of states

        Returns the HMM restricted to the selected subset of states.
        Will raise exception if the hidden transition matrix cannot be normalized on this subset

        """
        # restrict initial distribution
        pi_sub = self._Pi[states]
        pi_sub /= pi_sub.sum()

        # restrict transition matrix
        P_sub = self._Tij[states, :][:, states]
        # checks if this selection is possible
        if not np.all(P_sub.sum(axis=1) > 0):
            raise ValueError(f'Illegal sub_hmm request: transition matrix cannot be normalized on {states}')
        P_sub /= P_sub.sum(axis=1)[:, None]

        # restrict output model
        out_sub = self.output_model.sub_output_model(states)

        return HMM(pi_sub, P_sub, out_sub, lag=self.lag)

    def count_matrix(self):
        # TODO: does this belong here or to the BHMM sampler, or in a subclass containing HMM with data?
        """Compute the transition count matrix from hidden state trajectory.

        Returns
        -------
        C : numpy.array with shape (n_states,n_states)
            C[i,j] is the number of transitions observed from state i to state j

        Raises
        ------
        RuntimeError
            A RuntimeError is raised if the HMM model does not yet have a hidden state trajectory associated with it.

        Examples
        --------

        """
        if self.hidden_state_trajectories is None:
            raise RuntimeError('HMM model does not have a hidden state trajectory.')

        C = msmest.count_matrix(self.hidden_state_trajectories, 1, nstates=self._n_states, sparse_return=False)
        return C

    def count_init(self):
        """Compute the counts at the first time step

        Returns
        -------
        n : ndarray(n_states)
            n[i] is the number of trajectories starting in state i

        """
        if self.hidden_state_trajectories is None:
            raise RuntimeError('HMM model does not have a hidden state trajectory.')

        n = [traj[0] for traj in self.hidden_state_trajectories]
        return np.bincount(n, minlength=self.n_states)

    def collect_observations_in_state(self, observations, state_index):
        # TODO: this would work well in a subclass with data
        """Collect a vector of all observations belonging to a specified hidden state.

        Parameters
        ----------
        observations : list of numpy.array
            List of observed trajectories.
        state_index : int
            The index of the hidden state for which corresponding observations are to be retrieved.

        Returns
        -------
        collected_observations : numpy.array with shape (nsamples,)
            The collected vector of observations belonging to the specified hidden state.

        Raises
        ------
        RuntimeError
            A RuntimeError is raised if the HMM model does not yet have a hidden state trajectory associated with it.

        """
        if not self.hidden_state_trajectories:
            raise RuntimeError('HMM model does not have a hidden state trajectory.')
        dtype = observations[0].dtype
        collected_observations = np.array([], dtype=dtype)
        for (s_t, o_t) in zip(self.hidden_state_trajectories, observations):
            indices = np.where(s_t == state_index)[0]
            collected_observations = np.append(collected_observations, o_t[indices])

        return collected_observations
        collected_observations = (o_t[np.where(s_t == state_index)[0]]
                                  for s_t, o_t in zip(self.hidden_state_trajectories, observations)
                                  )
        return np.hstack(collected_observations)

    def generate_synthetic_state_trajectory(self, nsteps, initial_Pi=None, start=None, stop=None, dtype=np.int32):
        """Generate a synthetic state trajectory.

        Parameters
        ----------
        nsteps : int
            Number of steps in the synthetic state trajectory to be generated.
        initial_Pi : np.array of shape (n_states,), optional, default=None
            The initial probability distribution, if samples are not to be taken from the intrinsic
            initial distribution.
        start : int
            starting state. Exclusive with initial_Pi
        stop : int
            stopping state. Trajectory will terminate when reaching the stopping state before length number of steps.
        dtype : numpy.dtype, optional, default=numpy.int32
            The numpy dtype to use to store the synthetic trajectory.

        Returns
        -------
        states : np.array of shape (n_states,) of dtype=np.int32
            The trajectory of hidden states, with each element in range(0,n_states).

        Examples
        --------

        Generate a synthetic state trajectory of a specified length.

        >>> from sktime.markovprocess.bhmm import testsystems
        >>> model = testsystems.dalton_model()
        >>> states = model.generate_synthetic_state_trajectory(nsteps=100)

        """
        # consistency check
        if initial_Pi is not None and start is not None:
            raise ValueError('Arguments initial_Pi and start are exclusive. Only set one of them.')

        # Generate first state sample.
        if start is None:
            if initial_Pi is not None:
                start = np.random.choice(self._n_states, size=1, p=initial_Pi)
            else:
                start = np.random.choice(self._n_states, size=1, p=self._Pi)

        # Generate and return trajectory
        from msmtools import generation as msmgen
        traj = msmgen.generate_traj(self.transition_matrix, nsteps, start=start, stop=stop, dt=1)
        return traj.astype(dtype)

    def generate_synthetic_observation(self, state):
        """Generate a synthetic observation from a given state.

        Parameters
        ----------
        state : int
            The index of the state from which the observable is to be sampled.

        Returns
        -------
        observation : float
            The observation from the given state.

        Examples
        --------

        Generate a synthetic observation from a single state.

        >>> from sktime.markovprocess.bhmm import testsystems
        >>> model = testsystems.dalton_model()
        >>> observation = model.generate_synthetic_observation(0)

        """
        return self.output_model.generate_observation_from_state(state)

    def generate_synthetic_observation_trajectory(self, length, initial_Pi=None):
        """Generate a synthetic realization of observables.

        Parameters
        ----------
        length : int
            Length of synthetic state trajectory to be generated.
        initial_Pi : np.array of shape (n_states,), optional, default=None
            The initial probability distribution, if samples are not to be taken from equilibrium.

        Returns
        -------
        o_t : np.array of shape (n_states,) of dtype=np.float32
            The trajectory of observations.
        s_t : np.array of shape (n_states,) of dtype=np.int32
            The trajectory of hidden states, with each element in range(0,n_states).

        Examples
        --------

        Generate a synthetic observation trajectory for an equilibrium realization.

        >>> from sktime.markovprocess.bhmm import testsystems
        >>> model = testsystems.dalton_model()
        >>> [o_t, s_t] = model.generate_synthetic_observation_trajectory(length=100)

        Use an initial nonequilibrium distribution.

        >>> model = testsystems.dalton_model()
        >>> [o_t, s_t] = model.generate_synthetic_observation_trajectory(length=100, initial_Pi=np.array([1,0,0]))

        """
        # First, generate synthetic state trajetory.
        s_t = self.generate_synthetic_state_trajectory(length, initial_Pi=initial_Pi)

        # Next, generate observations from these states.
        o_t = self.output_model.generate_observation_trajectory(s_t)

        return [o_t, s_t]

    def generate_synthetic_observation_trajectories(self, ntrajectories, length, initial_Pi=None):
        """Generate a number of synthetic realization of observables from this model.

        Parameters
        ----------
        ntrajectories : int
            The number of trajectories to be generated.
        length : int
            Length of synthetic state trajectory to be generated.
        initial_Pi : np.array of shape (n_states,), optional, default=None
            The initial probability distribution, if samples are not to be taken from equilibrium.

        Returns
        -------
        O : list of np.array of shape (n_states,) of dtype=np.float32
            The trajectories of observations
        S : list of np.array of shape (n_states,) of dtype=np.int32
            The trajectories of hidden states

        Examples
        --------

        Generate a number of synthetic trajectories.

        >>> from sktime.markovprocess.bhmm import testsystems
        >>> model = testsystems.dalton_model()
        >>> O, S = model.generate_synthetic_observation_trajectories(ntrajectories=10, length=100)

        Use an initial nonequilibrium distribution.

        >>> model = testsystems.dalton_model(n_states=3)
        >>> O, S = model.generate_synthetic_observation_trajectories(ntrajectories=10, length=100, initial_Pi=np.array([1,0,0]))

        """
        O = []  # observations
        S = []  # state trajectories
        for trajectory_index in range(ntrajectories):
            o_t, s_t = self.generate_synthetic_observation_trajectory(length=length, initial_Pi=initial_Pi)
            O.append(o_t)
            S.append(s_t)

        return O, S

    def compute_viterbi_paths(self, observations):
        """Computes the Viterbi paths using the current HMM model"""
        A = self.transition_matrix
        pi = self.initial_distribution
        p_obs = self.output_model.p_obs
        paths = [
            # compute output probability matrix
            hidden.viterbi(A, p_obs(obs), pi)
            for obs in observations
        ]

        return paths
