
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

import numpy as np
import msmtools.estimation as msmest
from bhmm.estimators import _tmatrix_disconnected


class HMM(object):
    r""" Hidden Markov model (HMM).

    This class is used to represent an HMM. This could be a maximum-likelihood HMM or a sampled HMM from a
    Bayesian posterior.

    Parameters
    ----------
    Tij : np.array with shape (nstates, nstates), optional, default=None
        Row-stochastic transition matrix among states.
    output_model : :class:`bhmm.OutputModel`
        The output model for the states.
    lag : int, optional, default=1
        Lag time (optional) used to estimate the HMM. Used to compute relaxation timescales.
    Pi : np.array with shape (nstates), optional, default=None
        The initial state vector. Required when stationary=False
    stationary : bool, optional, default=True
        If true, the initial distribution is equal to the stationary distribution of the transition matrix
        If false, the initial distribution must be given as Pi
    reversible : bool, optional, default=True
        If true, the transition matrix is reversible.

    Examples
    --------

    >>> # Gaussian HMM
    >>> nstates = 2
    >>> pi = np.array([0.5, 0.5])
    >>> Tij = np.array([[0.8, 0.2], [0.5, 0.5]])
    >>> from bhmm import GaussianOutputModel
    >>> output_model = GaussianOutputModel(nstates, means=[-1, +1], sigmas=[1, 1])
    >>> model = HMM(pi, Tij, output_model)

    >>> # Discrete HMM
    >>> nstates = 2
    >>> pi = np.array([0.5, 0.5])
    >>> Tij = np.array([[0.8, 0.2], [0.5, 0.5]])
    >>> from bhmm import DiscreteOutputModel
    >>> output_model = DiscreteOutputModel([[0.5, 0.1, 0.4], [0.2, 0.3, 0.5]])
    >>> model = HMM(pi, Tij, output_model)

    """
    def __init__(self, Pi, Tij, output_model, lag=1):
        # set number of states
        self._nstates = np.shape(Tij)[0]
        # lag time
        self._lag = lag
        # output model
        self.output_model = output_model
        # hidden state trajectories are optional
        self.hidden_state_trajectories = None
        # update numbers
        self.update(Pi, Tij)

    def update(self, Pi, Tij):
        r""" Updates the transition matrix and recomputes all derived quantities """
        from msmtools import analysis as msmana

        # update transition matrix by copy
        self._Tij = np.array(Tij)
        assert msmana.is_transition_matrix(self._Tij), 'Given transition matrix is not a stochastic matrix'
        assert self._Tij.shape[0] == self._nstates, 'Given transition matrix has unexpected number of states '
        # reset spectral decomposition
        self._spectral_decomp_available = False

        # check initial distribution
        assert np.all(Pi >= 0), 'Given initial distribution contains negative elements.'
        assert np.any(Pi > 0), 'Given initial distribution is zero'
        self._Pi = np.array(Pi) / np.sum(Pi) # ensure normalization and make a copy

    def __repr__(self):
        from bhmm.output_models import OutputModel
        if issubclass(self.__class__, OutputModel):
            outrepr = repr(OutputModel.__repr__(self))
        else:
            outrepr = repr(self.output_model)
        """ Returns a string representation of the HMM """
        return "HMM(%d, %s, %s, Pi=%s, stationary=%s, reversible=%s)" % (self._nstates,
                                                                         repr(self._Tij),
                                                                         outrepr,
                                                                         repr(self._Pi),
                                                                         repr(self.is_stationary),
                                                                         repr(self.is_reversible))

    def __str__(self):
        """ Returns a human-readable string representation of the HMM """
        output = 'Hidden Markov model\n'
        output += '-------------------\n'
        output += 'nstates: %d\n' % self._nstates
        output += 'Tij:\n'
        output += str(self._Tij) + '\n'
        output += 'Pi:\n'
        output += str(self._Pi) + '\n'
        output += 'output model:\n'
        from bhmm.output_models import OutputModel
        if issubclass(self.__class__, OutputModel):
            output += str(OutputModel.__str__(self))
        else:
            output += str(self.output_model)
        output += '\n'
        return output

    def _do_spectral_decomposition(self):
        self._R, self._D, self._L = _tmatrix_disconnected.rdl_decomposition(self._Tij, reversible=self.is_reversible)
        self._eigenvalues = np.diag(self._D)
        self._spectral_decomp_available = True

    def _ensure_spectral_decomposition(self):
        """
        """
        if not self._spectral_decomp_available:
            self._do_spectral_decomposition()

    @property
    def lag(self):
        r""" Lag time of the model, i.e. the number of observated trajectory steps made by the transition matrix """
        return self._lag

    @property
    def is_strongly_connected(self):
        r""" Whether the HMM transition matrix is strongly connected """
        return _tmatrix_disconnected.is_connected(self._Tij, strong=True)

    @property
    def strongly_connected_sets(self):
        return _tmatrix_disconnected.connected_sets(self._Tij, strong=True)

    @property
    def is_weakly_connected(self):
        r""" Whether the HMM transition matrix is weakly connected """
        return _tmatrix_disconnected.is_connected(self._Tij, strong=False)

    @property
    def weakly_connected_sets(self):
        return _tmatrix_disconnected.connected_sets(self._Tij, strong=False)

    @property
    def is_reversible(self):
        r""" Whether the HMM is reversible """
        return _tmatrix_disconnected.is_reversible(self._Tij)

    @property
    def is_stationary(self):
        r""" Whether the MSM is stationary, i.e. whether the initial distribution is the stationary distribution
         of the hidden transition matrix. """
        # for disconnected matrices, the stationary distribution depends on the estimator, so we can't compute
        # it directly. Therefore we test whether the initial distribution is stationary.
        return np.allclose(np.dot(self._Pi, self._Tij), self._Pi)

    @property
    def nstates(self):
        r""" The number of hidden states """
        return self._nstates

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
        assert _tmatrix_disconnected.is_connected(self._Tij, strong=False), \
            'No unique stationary distribution because transition matrix is not connected'
        import msmtools.analysis as msmana
        return msmana.stationary_distribution(self._Tij)

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
        L : ndarray(nstates,nstates)
            left eigenvectors in a row matrix. l_ij is the j'th component of the i'th left eigenvector

        """
        self._ensure_spectral_decomposition()
        return self._L

    @property
    def eigenvectors_right(self):
        r""" Right eigenvectors of the hidden transition matrix

        Returns
        -------
        R : ndarray(nstates,nstates)
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
            defined by :math:`-tau / ln | \lambda_i |, i = 2,...,nstates`, where
            :math:`\lambda_i` are the hidden transition matrix eigenvalues.

        """
        from msmtools.analysis.dense.decomposition import timescales_from_eigenvalues as _timescales

        self._ensure_spectral_decomposition()
        ts = _timescales(self._eigenvalues, tau=self._lag)
        return ts[1:]

    @property
    def lifetimes(self):
        r""" Lifetimes of states of the hidden transition matrix

        Returns
        -------
        l : ndarray(nstates)
            state lifetimes in units of the input trajectory time step,
            defined by :math:`-tau / ln | p_{ii} |, i = 1,...,nstates`, where
            :math:`p_{ii}` are the diagonal entries of the hidden transition matrix.

        """
        return -self._lag / np.log(np.diag(self.transition_matrix))

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
        assert np.all(P_sub.sum(axis=1) > 0), \
            'Illegal sub_hmm request: transition matrix cannot be normalized on ' + str(states)
        P_sub /= P_sub.sum(axis=1)[:, None]

        # restrict output model
        out_sub = self.output_model.sub_output_model(states)

        return HMM(pi_sub, P_sub, out_sub, lag=self.lag)

    def count_matrix(self):
        # TODO: does this belong here or to the BHMM sampler, or in a subclass containing HMM with data?
        """Compute the transition count matrix from hidden state trajectory.

        Returns
        -------
        C : numpy.array with shape (nstates,nstates)
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

        C = msmest.count_matrix(self.hidden_state_trajectories, 1, nstates=self._nstates)
        return C.toarray()

    def count_init(self):
        """Compute the counts at the first time step

        Returns
        -------
        n : ndarray(nstates)
            n[i] is the number of trajectories starting in state i

        """
        if self.hidden_state_trajectories is None:
            raise RuntimeError('HMM model does not have a hidden state trajectory.')

        n = [traj[0] for traj in self.hidden_state_trajectories]
        return np.bincount(n, minlength=self.nstates)

    # def emission_probability(self, state, observation):
    #     """Compute the emission probability of an observation from a given state.
    #
    #     Parameters
    #     ----------
    #     state : int
    #         The state index for which the emission probability is to be computed.
    #
    #     Returns
    #     -------
    #     Pobs : float
    #         The probability (or probability density, if continuous) of the observation.
    #
    #     TODO
    #     ----
    #     * Vectorize
    #
    #     Examples
    #     --------
    #
    #     Compute the probability of observing an emission of 0 from state 0.
    #
    #     >>> from bhmm import testsystems
    #     >>> model = testsystems.dalton_model(nstates=3)
    #     >>> state_index = 0
    #     >>> observation = 0.0
    #     >>> Pobs = model.emission_probability(state_index, observation)
    #
    #     """
    #     return self.output_model.p_o_i(observation, state)

    # def log_emission_probability(self, state, observation):
    #     """Compute the log emission probability of an observation from a given state.
    #
    #     Parameters
    #     ----------
    #     state : int
    #         The state index for which the emission probability is to be computed.
    #
    #     Returns
    #     -------
    #     log_Pobs : float
    #         The log probability (or probability density, if continuous) of the observation.
    #
    #     TODO
    #     ----
    #     * Vectorize
    #
    #     Examples
    #     --------
    #
    #     Compute the log probability of observing an emission of 0 from state 0.
    #
    #     >>> from bhmm import testsystems
    #     >>> model = testsystems.dalton_model(nstates=3)
    #     >>> state_index = 0
    #     >>> observation = 0.0
    #     >>> log_Pobs = model.log_emission_probability(state_index, observation)
    #
    #     """
    #     return self.output_model.log_p_o_i(observation, state)

    def collect_observations_in_state(self, observations, state_index):
        # TODO: this would work well in a subclass with data
        """Collect a vector of all observations belonging to a specified hidden state.

        Parameters
        ----------
        observations : list of numpy.array
            List of observed trajectories.
        state_index : int
            The index of the hidden state for which corresponding observations are to be retrieved.
        dtype : numpy.dtype, optional, default=numpy.float64
            The numpy dtype to use to store the collected observations.

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

    def generate_synthetic_state_trajectory(self, nsteps, initial_Pi=None, start=None, stop=None, dtype=np.int32):
        """Generate a synthetic state trajectory.

        Parameters
        ----------
        nsteps : int
            Number of steps in the synthetic state trajectory to be generated.
        initial_Pi : np.array of shape (nstates,), optional, default=None
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
        states : np.array of shape (nstates,) of dtype=np.int32
            The trajectory of hidden states, with each element in range(0,nstates).

        Examples
        --------

        Generate a synthetic state trajectory of a specified length.

        >>> from bhmm import testsystems
        >>> model = testsystems.dalton_model()
        >>> states = model.generate_synthetic_state_trajectory(nsteps=100)

        """
        # consistency check
        if initial_Pi is not None and start is not None:
            raise ValueError('Arguments initial_Pi and start are exclusive. Only set one of them.')

        # Generate first state sample.
        if start is None:
            if initial_Pi is not None:
                start = np.random.choice(range(self._nstates), size=1, p=initial_Pi)
            else:
                start = np.random.choice(range(self._nstates), size=1, p=self._Pi)

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

        >>> from bhmm import testsystems
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
        initial_Pi : np.array of shape (nstates,), optional, default=None
            The initial probability distribution, if samples are not to be taken from equilibrium.

        Returns
        -------
        o_t : np.array of shape (nstates,) of dtype=np.float32
            The trajectory of observations.
        s_t : np.array of shape (nstates,) of dtype=np.int32
            The trajectory of hidden states, with each element in range(0,nstates).

        Examples
        --------

        Generate a synthetic observation trajectory for an equilibrium realization.

        >>> from bhmm import testsystems
        >>> model = testsystems.dalton_model()
        >>> [o_t, s_t] = model.generate_synthetic_observation_trajectory(length=100)

        Use an initial nonequilibrium distribution.

        >>> from bhmm import testsystems
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
        initial_Pi : np.array of shape (nstates,), optional, default=None
            The initial probability distribution, if samples are not to be taken from equilibrium.

        Returns
        -------
        O : list of np.array of shape (nstates,) of dtype=np.float32
            The trajectories of observations
        S : list of np.array of shape (nstates,) of dtype=np.int32
            The trajectories of hidden states

        Examples
        --------

        Generate a number of synthetic trajectories.

        >>> from bhmm import testsystems
        >>> model = testsystems.dalton_model()
        >>> O, S = model.generate_synthetic_observation_trajectories(ntrajectories=10, length=100)

        Use an initial nonequilibrium distribution.

        >>> from bhmm import testsystems
        >>> model = testsystems.dalton_model(nstates=3)
        >>> O, S = model.generate_synthetic_observation_trajectories(ntrajectories=10, length=100, initial_Pi=np.array([1,0,0]))

        """
        O = list()  # observations
        S = list()  # state trajectories
        for trajectory_index in range(ntrajectories):
            o_t, s_t = self.generate_synthetic_observation_trajectory(length=length, initial_Pi=initial_Pi)
            O.append(o_t)
            S.append(s_t)

        return O, S
