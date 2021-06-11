from numbers import Integral
from typing import Union, Optional, List, Iterable

import numpy as np

from deeptime.base import Model
from ._output_model import OutputModel, DiscreteOutputModel
from deeptime.markov import sample
from ._hmm_bindings.util import viterbi as viterbi_impl, forward as forward_impl
from ...util.types import ensure_dtraj_list, ensure_array


class HiddenMarkovModel(Model):
    """ Hidden Markov state model consisting of a transition model
    (:class:`MSM <deeptime.markov.msm.MarkovStateModel>`) on the hidden states, an
    :class:`output model <OutputModel>` which maps from the hidden states to a distribution of observable states,
    and optionally an initial distribution on the hidden states. Some properties require a crisp assignment to
    states in the observable space, in which case only a discrete output model can be used.

    Parameters
    ----------
    transition_model : (m,m) ndarray or MarkovStateModel
        Transition matrix for hidden (macro) states
    output_model : (m,n) ndarray or OutputModel
        observation probability matrix from hidden to observable (micro) states or OutputModel instance which yields
        the mapping from hidden to observable state.
    initial_distribution : (m,) ndarray, optional, default=None
        Initial distribution of the hidden (macro) states. Default is uniform.
    likelihoods : (k,) ndarray, optional, default=None
        Likelihood progression of the HMM as it was trained for k iterations with Baum-Welch.
    state_probabilities : list of ndarray, optional, default=None
        List of state probabilities for each trajectory that the model was trained on (gammas).
    initial_count : ndarray, optional, default=None
        Initial counts of the hidden (macro) states, computed from the gamma output of the Baum-Welch algorithm
    hidden_state_trajectories : list of ndarray, optional, default=None
        When estimating the HMM the data's most likely hidden state trajectory is determined and can be saved
        with the model by providing this argument.
    stride : int or str('effective'), optional, default=1
        Stride which was used to subsample discrete trajectories while estimating a HMM. Can either be an integer
        value which determines the offset or 'effective', which makes an estimate of a stride at which subsequent
        discrete trajectory elements are uncorrelated.
    observation_symbols : array_like, optional, default=None
        Sorted unique symbols in observations. If None, it is assumed that all possible observations are made
        and the state symbols are set to an iota range over the number of observation states.
    observation_symbols_full : array_like, optional, default=None
        Full set of symbols in observations. If None, it is assumed to coincide with observation_symbols.

    References
    ----------
    .. bibliography:: /references.bib
        :style: unsrt
        :filter: docname in docnames
        :keyprefix: hmm-

    See Also
    --------
    init.discrete.metastable_from_data : initial guess from data with discrete output model
    init.discrete.metastable_from_msm : initial guess from MSM with discrete output model
    init.gaussian.from_data : initial guess from data with Gaussian output model
    MaximumLikelihoodHMM : maximum likelihood estimation of HMMs
    BayesianHMM : Bayesian sampling of models for confidences.
    """

    def __init__(self, transition_model, output_model: Union[np.ndarray, OutputModel],
                 initial_distribution: Optional[np.ndarray] = None, likelihoods: Optional[np.ndarray] = None,
                 state_probabilities: Optional[List[np.ndarray]] = None, initial_count: Optional[np.ndarray] = None,
                 hidden_state_trajectories: Optional[Iterable[np.ndarray]] = None, stride: Union[int, str] = 1,
                 observation_symbols: Optional[np.ndarray] = None,
                 observation_symbols_full: Optional[np.ndarray] = None):
        super().__init__()
        if isinstance(transition_model, np.ndarray):
            from deeptime.markov.msm import MarkovStateModel
            transition_model = MarkovStateModel(transition_model)
        if isinstance(output_model, np.ndarray):
            output_model = DiscreteOutputModel(output_model)
        if transition_model.n_states != output_model.n_hidden_states:
            raise ValueError("Transition model must describe hidden states")
        if initial_distribution is None:
            # uniform
            initial_distribution = np.ones(transition_model.n_states) / transition_model.n_states
        if initial_distribution.shape[0] != transition_model.n_states:
            raise ValueError("Initial distribution over hidden states must be of length {}"
                             .format(transition_model.n_states))
        self._transition_model = transition_model
        self._output_model = output_model
        self._initial_distribution = initial_distribution
        self._likelihoods = likelihoods
        self._state_probabilities = state_probabilities
        self._initial_count = initial_count
        self._hidden_state_trajectories = hidden_state_trajectories
        if observation_symbols is None and output_model.n_observable_states >= 0:
            observation_symbols = np.arange(output_model.n_observable_states)
            observation_symbols_full = observation_symbols
        self._observation_symbols = observation_symbols
        self._observation_symbols_full = observation_symbols_full
        if not (isinstance(stride, Integral) or (isinstance(stride, str) and stride == 'effective')):
            raise ValueError("Stride argument must either be an integer value or 'effective', "
                             "but was: {}".format(stride))
        self._stride = stride

    def compute_observation_likelihood(self, data: Union[np.ndarray, List[np.ndarray]]):
        r""" Computes the likelihood of observed data under this model.

        Internally, the forward pass of the Baum-Welch algorithm is used.

        Parameters
        ----------
        data : array_like or list of array_like
            The observations

        Returns
        -------
        likelihood : float
            The computed likelihood.
        """
        if not isinstance(data, (list, tuple)):
            data = [data]
        max_n_frames = max(len(obs) for obs in data)

        # get parameters
        A = self.transition_model.transition_matrix
        pi = self.initial_distribution
        alpha = np.zeros((max_n_frames, self.n_hidden_states), dtype=A.dtype)

        # compute output probability matrix
        loglik = 0.
        for obs in data:
            T = len(obs)
            pobs = self.output_model.to_state_probability_trajectory(obs)
            pobs = pobs.astype(A.dtype)
            loglik += forward_impl(A, pobs, pi, alpha_out=alpha, T=T)
        return loglik

    @property
    def lagtime(self) -> int:
        r""" The lagtime this model was estimated at.

        Returns
        -------
        lagtime : int
            The lagtime.
        """
        return self.transition_model.lagtime

    @property
    def stride(self):
        r"""
        The stride parameter which was used to subsample the discrete trajectories when estimating the hidden
        markov state model. Can either be an integer value or 'effective', in which case a stride is estimated at
        which subsequent states are uncorrelated.

        Returns
        -------
        stride : int or str
            The stride parameter.
        """
        return self._stride

    @property
    def observation_symbols(self) -> Optional[np.ndarray]:
        r"""
        The symbols represented by this HMM in observation space. Can be None in case the output model has no
        discrete observations it is None.

        Returns
        -------
        The list of observation symbols or None.
        """
        return self._observation_symbols

    @property
    def observation_symbols_full(self) -> Optional[np.ndarray]:
        r""" All symbols that the original model contained (original before taking any submodel).

        Returns
        -------
        The list of observation symbols or None, if there are no discrete symbols or None was provided.
        """
        return self._observation_symbols_full

    @property
    def n_observation_states(self) -> int:
        r"""
        Property determining the number of observed/macro states. It coincides with the size of the second axis
        of the observation probabilities matrix in case of a discrete output model.

        Returns
        -------
        Number of observed/macro states
        """
        return self.output_model.n_observable_states

    @property
    def count_model(self):
        r"""
        Yields the count model for the micro (hidden) states. The count matrix is estimated from Viterbi paths.

        Returns
        -------
        count_model : deeptime.markov.TransitionCountModel
            The count model for the micro states.
        """
        return self.transition_model.count_model

    @property
    def transition_model(self):
        r""" Yields the transition model for the hidden states.

        Returns
        -------
        model : deeptime.markov.msm.MarkovStateModel
            The transition model.
        """
        return self._transition_model

    @property
    def initial_count(self) -> Optional[np.ndarray]:
        r"""
        The hidden initial counts, can be None.

        Returns
        -------
        Initial counts.
        """
        return self._initial_count

    @property
    def output_model(self) -> OutputModel:
        r"""
        The selected output model for this HMM. The output model can map from the hidden states to observable states
        and can also be fitted to data.

        Returns
        -------
        The output model
        """
        return self._output_model

    @property
    def initial_distribution(self) -> np.ndarray:
        r""" The initial distribution of this HMM over the hidden states.

        Returns
        -------
        The initial distribution.
        """
        return self._initial_distribution

    @property
    def n_hidden_states(self) -> int:
        r"""
        The number of hidden states. Can also be retrieved from the output model as well as from the transition model.

        Returns
        -------
        Number of hidden states
        """
        return self.output_model.n_hidden_states

    @property
    def likelihoods(self) -> Optional[np.ndarray]:
        r"""
        If the model comes from the MaximumLikelihoodHMM estimator, this property contains the sequence of likelihoods
        generated from the fitting iteration.

        Returns
        -------
        Sequence of likelihoods, otherwise None.
        """
        return self._likelihoods

    @property
    def likelihood(self) -> Optional[float]:
        r"""
        The estimated likelihood of this model based on the training data. Only available if the sequence of likelihoods
        is provided.

        Returns
        -------
        The estimated likelihood, otherwise None.
        """
        return self.likelihoods[-1] if self.likelihoods is not None else None

    @property
    def state_probabilities(self) -> Optional[List[np.ndarray]]:
        r"""
        List of state probabilities for each trajectory that the model was trained on (gammas in the Baum-Welch algo).

        Returns
        -------
        List of state probabilities if initially provided in the constructor.
        """
        return self._state_probabilities

    @property
    def transition_counts(self) -> Optional[np.ndarray]:
        r"""
        The transition counts for the hidden states as estimated in the fitting procedure.

        Returns
        -------
        The transition counts, can be None if the transition model has no count model.
        """
        return self.transition_model.count_model.count_matrix if self.transition_model.count_model is not None else None

    @property
    def hidden_state_trajectories(self) -> Optional[List[np.ndarray]]:
        r"""
        Training trajectories mapped to hidden states after estimation.

        Returns
        -------
        hidden state trajectories, can be None if not provided in constructor.
        """
        return self._hidden_state_trajectories

    @property
    def output_probabilities(self) -> np.ndarray:
        r"""
        Returns the probabilities for each hidden state to map to a particular observation state. Only available
        if the underlying output model is a `DiscreteOutputModel`.

        Returns
        -------
        probabilities : np.ndarray
            a (M,N) row-stochastic matrix mapping from each hidden to each observation state
        """
        if isinstance(self.output_model, DiscreteOutputModel):
            return self.output_model.output_probabilities
        raise ValueError("Output probabilities are only available for HMMs with discrete output model.")

    @property
    def stationary_distribution_obs(self):
        r"""
        The stationary distribution in observable space. Only available with a discrete output model.

        Returns
        -------
        stationary distribution in observation space if available
        """
        if isinstance(self.output_model, DiscreteOutputModel):
            return np.dot(self.transition_model.stationary_distribution, self.output_probabilities)
        raise RuntimeError("only available for discrete output model")

    @property
    def eigenvectors_left_obs(self):
        r"""
        Left eigenvectors in observation space. Only available with a discrete output model.

        Returns
        -------
        Left eigenvectors in observation space.
        """
        return np.dot(self.transition_model.eigenvectors_left(), self.output_probabilities)

    @property
    def eigenvectors_right_obs(self):
        r"""
        Right eigenvectors in observation space. Only available with a discrete output model.

        Returns
        -------
        Right eigenvectors in observation space.
        """
        return np.dot(self.metastable_memberships, self.transition_model.eigenvectors_right())

    def transition_matrix_obs(self, k=1) -> np.ndarray:
        r""" Computes the transition matrix between observed states

        Transition matrices for longer lag times than the one used to parametrize this HMM can be obtained by setting
        the k option. Note that a HMM is not Markovian, thus we cannot compute transition matrices at longer lag times
        using the Chapman-Kolmogorow equality. I.e.:

        .. math::
            P (k \tau) \neq P^k (\tau)

        This function computes the correct transition matrix using the metastable (coarse)
        transition matrix :math:`P_c` as:

        .. math::
            P (k \tau) = {\Pi}^{-1} \chi^{\top} ({\Pi}_c) P_c^k (\tau) \chi

        where :math:`\chi` is the output probability matrix, :math:`\Pi_c` is a diagonal matrix with the
        metastable-state (coarse) stationary distribution and :math:`\Pi` is a diagonal matrix with the
        observable-state stationary distribution.

        Parameters
        ----------
        k : int, optional, default=1
            Multiple of the lag time.
            By default (k=1), the transition matrix at the lag time used to
            construct this HMM will be returned. If a higher power is given,

        """
        Pi_c = np.diag(self.transition_model.stationary_distribution)
        P_c = self.transition_model.transition_matrix
        P_c_k = np.linalg.matrix_power(P_c, k)  # take a power if needed
        B = self.output_probabilities
        C = np.dot(np.dot(B.T, Pi_c), np.dot(P_c_k, B))
        P = C / C.sum(axis=1)[:, None]  # row normalization
        return P

    @property
    def lifetimes(self) -> np.ndarray:
        r""" Lifetimes of states of the hidden transition matrix

        Returns
        -------
        l : ndarray(n_states)
            state lifetimes in units of the input trajectory time step,
            defined by :math:`-\tau / \ln \mid p_{ii} \mid, i = 1,...,n_\mathrm{states}`, where
            :math:`p_{ii}` are the diagonal entries of the hidden transition matrix.
        """
        return -self.transition_model.lagtime / np.log(np.diag(self.transition_model.transition_matrix))

    def compute_viterbi_paths(self, observations) -> List[np.ndarray]:
        r"""
        Computes the Viterbi paths using the current HMM model.

        Parameters
        ----------
        observations : list of array_like or array_like
            observations

        Returns
        -------
        paths : list of np.ndarray
            the computed viterbi paths
        """
        observations = ensure_dtraj_list(observations)
        A = self.transition_model.transition_matrix
        pi = self.initial_distribution
        state_probabilities = [
            self.output_model.to_state_probability_trajectory(obs) for obs in observations
        ]
        paths = [viterbi(A, obs, pi) for obs in state_probabilities]
        return paths

    def collect_observations_in_state(self, observations: List[np.ndarray], state_index: int):
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
        observations = ensure_dtraj_list(observations)

        from ._util import observations_in_state
        return observations_in_state(self.hidden_state_trajectories, observations, state_index)

    ################################################################################
    # Generation of trajectories and samples
    ################################################################################

    def simulate(self, n_steps, start=None, stop=None, dt=1):
        """
        Generates a realization of the Hidden Markov Model

        Parameters
        ----------
        n_steps : int
            trajectory length in steps of the lag time
        start : int, optional, default = None
            starting hidden state. If not given, will sample from the stationary
            distribution of the hidden transition matrix.
        stop : int or int-array-like, optional, default = None
            stopping hidden set. If given, the trajectory will be stopped before
            N steps once a hidden state of the stop set is reached
        dt : int
            trajectory will be saved every dt time steps.
            Internally, the dt'th power of P is taken to ensure a more efficient simulation.

        Returns
        -------
        htraj : (N/dt, ) ndarray
            The hidden state trajectory with length N/dt
        otraj : (N/dt, ) ndarray
            The observable state discrete trajectory with length N/dt

        """
        # sample hidden trajectory
        htraj = self.transition_model.simulate(n_steps, start=start, stop=stop, dt=dt)
        otraj = self.output_model.generate_observation_trajectory(htraj)
        return htraj, otraj

    def transform_discrete_trajectories_to_observed_symbols(self, dtrajs):
        r"""A list of integer arrays with the discrete trajectories mapped to the currently used set of observation
        symbols. For example, if there has been a subselection of the model for connectivity='largest', the indices
        will be given within the connected set, frames that do not correspond to a considered symbol are set to -1.

        Parameters
        ----------
        dtrajs : array_like or list of array_like
            discretized trajectories

        Returns
        -------
        array_like or list of array_like
            Curated discretized trajectories so that unconsidered symbols are mapped to -1.
        """
        dtrajs = ensure_dtraj_list(dtrajs)

        max_state = self.observation_symbols_full.max()
        # add one because of 0 indexing and another one so that -1 gets mapped to -1
        mapping = np.full(max_state + 2, -1, dtype=np.int32)
        mapping[self.observation_symbols] = self.observation_symbols

        # map elements which are too large to -1 directly
        transformed_dtrajs = []
        for dtraj in dtrajs:
            transformed_dtrajs.append(dtraj.copy())
            transformed_dtrajs[-1][np.where(dtraj > max_state)[0]] = -1

        # perform mapping
        return [mapping[dtraj] for dtraj in transformed_dtrajs]

    def sample_by_observation_probabilities(self, dtrajs, nsample):
        r"""Generates samples according to the current observation probability distribution.

        Notes
        -----
        Sampling from off-sample-trajectories might yield -1 indices as discrete observable states
        are drawn from output probability distributions and off-sample trajectories might not
        contain all drawn observable states.

        Parameters
        ----------
        dtrajs : discrete trajectory
            Input observation trajectory or list of trajectories
        nsample : int
            Number of samples per distribution.

        Returns
        -------
        indexes : length m list of ndarray( (nsample, 2) )
            List of the sampled indices by distribution.
            Each element is an index array with a number of rows equal to nsample, with rows consisting of a
            tuple (i, t), where i is the index of the trajectory and t is the time index within the trajectory.
        """
        mapped = self.transform_discrete_trajectories_to_observed_symbols(dtrajs)
        if all(np.all(x == -1) for x in mapped):
            raise ValueError("The discrete trajectories contained no elements which are in the observation "
                             "symbols of this HMM.")
        observable_state_indices = sample.compute_index_states(mapped, subset=self.observation_symbols)
        return sample.indices_by_distribution(observable_state_indices, self.output_probabilities, nsample)

    # ================================================================================================================
    # Metastable state stuff based on HMM transition matrix
    # ================================================================================================================

    @property
    def metastable_memberships(self):
        r""" Computes the memberships of observable states to metastable sets by Bayesian inversion.
        :cite:`hmm-noe2013projected`

        Returns
        -------
        M : ndarray((n,m))
            A matrix containing the probability or membership of each
            observable state to be assigned to each metastable or hidden state.
            The row sums of M are 1.
        """
        nonzero = np.nonzero(self.stationary_distribution_obs)[0]
        M = np.zeros((self.n_observation_states, self.transition_model.n_states))
        M[nonzero, :] = np.transpose(np.diag(self.transition_model.stationary_distribution).dot(
            self.output_probabilities[:, nonzero]) / self.stationary_distribution_obs[nonzero])
        # renormalize
        M[nonzero, :] /= M.sum(axis=1)[nonzero, None]
        return M

    @property
    def metastable_distributions(self):
        r""" Returns the output probability distributions. Identical to
            :meth:`output_probabilities`

        Returns
        -------
        Pout : ndarray (m,n)
            output probability matrix from hidden to observable discrete states

        See also
        --------
        output_probabilities

        """
        return self.output_probabilities

    @property
    def metastable_sets(self):
        r""" Computes the metastable sets of observable states within each
            metastable set

        Notes
        -----
        This is only recommended for visualization purposes. You *cannot*
        compute any actual quantity of the coarse-grained kinetics without
        employing the fuzzy memberships!

        Returns
        -------
        sets : list of int-arrays
            A list of length equal to metastable states. Each element is an array
            with observable state indexes contained in it

        """
        res = []
        assignment = self.metastable_assignments
        for i in range(self.transition_model.n_states):
            res.append(np.where(assignment == i)[0])
        return res

    @property
    def metastable_assignments(self):
        r""" Computes the assignment to metastable sets for observable states

        Notes
        -----
        This is only recommended for visualization purposes. You *cannot*
        compute any actual quantity of the coarse-grained kinetics without
        employing the fuzzy memberships!

        Returns
        -------
        ndarray((n) ,dtype=int)
            For each observable state, the metastable state it is located in.

        See Also
        --------
        output_probabilities

        """
        return np.argmax(self.output_probabilities, axis=0)

    # ================================================================================================================
    # Micro- / observable state properties
    # ================================================================================================================

    def _project_to_hidden(self, a: np.ndarray, ndim=None, allow_none=None) -> Optional[np.ndarray]:
        r"""
        Projects an observable state vector to hidden states. Only available for discrete output model.

        Parameters
        ----------
        a : array_like
            observable state vector
        ndim : int, optional, default=None
            requested dimension of observable state vector
        allow_none : bool, optional, default=None
            Whether None is allowed as observable state vector, resulting in a no-op.
        Returns
        -------
        hidden state vector corresponding to observable state vector
        """
        if allow_none and a is None:
            return None
        a = ensure_array(a, ndim=ndim)
        if len(a) != self.n_observation_states:
            raise ValueError("Input array has incompatible shape, needs to have "
                             "length {} but had length {}.".format(len(a), self.n_observation_states))
        return np.dot(self.output_probabilities, a)

    def expectation_obs(self, a):
        r"""Equilibrium expectation value of a given observable state vector.

        See Also
        --------
        deeptime.markov.msm.MarkovStateModel.expectation
        """
        return self.transition_model.expectation(self._project_to_hidden(a))

    def correlation_obs(self, a, b=None, maxtime=None, k=None, ncv=None):
        r""" Time-correlation for equilibrium experiment based on observable state vectors a and b.

        See Also
        --------
        deeptime.markov.msm.MarkovStateModel.correlation
        """
        # basic checks for a and b
        a = self._project_to_hidden(a)
        b = self._project_to_hidden(b, ndim=1, allow_none=True)
        return self.transition_model.correlation(a, b=b, maxtime=maxtime, k=k, ncv=ncv)

    def fingerprint_correlation_obs(self, a, b=None, k=None, ncv=None):
        r""" Dynamical fingerprint for equilibrium time-correlation experiment based on observable state vectors a
        and b.

        See Also
        --------
        deeptime.markov.msm.MarkovStateModel.fingerprint_correlation
        """
        a = self._project_to_hidden(a, ndim=1)
        b = self._project_to_hidden(b, ndim=1, allow_none=True)
        return self.transition_model.fingerprint_correlation(a, b=b, k=k, ncv=ncv)

    def relaxation_obs(self, p0, a, maxtime=None, k=None, ncv=None):
        r""" Simulates a perturbation-relaxation experiment based on observable state vector and distribution.

        See Also
        --------
        deeptime.markov.msm.MarkovStateModel.relaxation
        """
        p0 = self._project_to_hidden(p0, ndim=1)
        a = self._project_to_hidden(a, ndim=1)
        return self.transition_model.relaxation(p0, a, maxtime=maxtime, k=k, ncv=ncv)

    def fingerprint_relaxation_obs(self, p0, a, k=None, ncv=None):
        r""" Dynamical fingerprint for perturbation/relaxation experiment
        based on observable state vector and distribution.

        See Also
        --------
        deeptime.markov.msm.MarkovStateModel.fingerprint_relaxation
        """
        p0 = self._project_to_hidden(p0, ndim=1)
        a = self._project_to_hidden(a, ndim=1)
        return self.transition_model.fingerprint_relaxation(p0, a, k=k, ncv=ncv)

    def propagate(self, p0, k):
        r""" Propagates the initial distribution p0 defined on observable space k times.

        Therefore computes the product

        .. math::

            p_k = p_0^T P^k

        If the lag time of transition matrix :math:`P` is :math:`\tau`, this
        will provide the probability distribution at time :math:`k \tau`.

        Parameters
        ----------
        p0 : ndarray(n)
            Initial distribution. Vector of size of the active set.

        k : int
            Number of time steps

        Returns
        ----------
        pk : ndarray(n)
            Distribution after k steps

        """
        if k == 0:  # simply return p0 normalized
            return p0 / p0.sum()

        p0 = self._project_to_hidden(p0)

        ev_right = self.transition_model.eigenvectors_right(self.n_hidden_states)
        ev_left = self.transition_model.eigenvectors_left(self.n_hidden_states)
        pk = np.linalg.multi_dot([p0.T, ev_right, np.diag(np.power(self.transition_model.eigenvalues(), k)), ev_left])

        pk = np.dot(pk, self.output_probabilities)  # convert back to microstate space

        # normalize to 1.0 and return
        return pk / pk.sum()

    # ================================================================================================================
    # Submodel functionality
    # ================================================================================================================

    def submodel(self, states: Optional[np.ndarray] = None, obs: Optional[np.ndarray] = None):
        """Returns a HMM with restricted state space

        Parameters
        ----------
        states : None or int-array
            Hidden states to restrict the model to. In addition to specifying
            the subset, possible options are:

            * int-array: indices of states to restrict onto
            * None : all states - don't restrict

        obs : None or int-array
            Observed states to restrict the model to. In addition to specifying
            an array with the state labels to be observed, possible options are:

              * int-array: indices of states to restrict onto
              * None : all states - don't restrict
        Returns
        -------
        hmm : HiddenMarkovModel
            The restricted HMM.
        """

        if states is None and obs is None:
            return self  # do nothing
        if states is None:
            states = np.arange(self.n_hidden_states)
        if obs is None:
            obs = np.arange(self.n_observation_states)

        transition_model = self.transition_model.submodel(states)

        if self.initial_count is not None:
            initial_count = self.initial_count[states].copy()
        else:
            initial_count = None
        initial_distribution = self.initial_distribution[states] / np.sum(self.initial_distribution[states])

        # observation matrix
        output_model = self.output_model.submodel(states, obs)

        observation_symbols = self.observation_symbols
        if observation_symbols is not None:
            observation_symbols = observation_symbols[obs]

        model = HiddenMarkovModel(transition_model=transition_model, output_model=output_model,
                                  initial_distribution=initial_distribution, likelihoods=self.likelihoods,
                                  state_probabilities=self.state_probabilities, initial_count=initial_count,
                                  hidden_state_trajectories=self.hidden_state_trajectories,
                                  observation_symbols=observation_symbols,
                                  observation_symbols_full=self.observation_symbols_full)
        return model

    def _select_states(self, connectivity_threshold, states) -> np.ndarray:
        r"""
        Retrieves a selection of states based on the arguments provided.

        Parameters
        ----------
        connectivity_threshold : str or int
            A connectivity threshold that needs to be exceeded so that an edge in the connectivity graph is considered
            to be relevant. In case of a string it can be set to "1/n", where n is the number of hidden states.
        states : array_like
            The states to restrict onto. Can be further reduced by providing a threshold.
        Returns
        -------
        A ndarray containing a subselection of states.
        """
        if str(connectivity_threshold) == '1/n':
            connectivity_threshold = 1.0 / float(self.transition_model.n_states)
        if isinstance(states, str):
            strong = 'strong' in states
            largest = 'largest' in states
            count_model = self.transition_model.count_model
            S = count_model.connected_sets(connectivity_threshold=connectivity_threshold, directed=strong)
            if largest:
                score = np.array([len(s) for s in S])
            else:
                score = np.array([count_model.count_matrix[np.ix_(s, s)].sum() for s in S])
            states = S[np.argmax(score)]
        return states

    def nonempty_obs(self, dtrajs) -> np.ndarray:
        r"""
        Computes the set of visited observable states given a set of discrete trajectories.

        Parameters
        ----------
        dtrajs : array_like
            observable trajectory

        Returns
        -------
        symbols : np.ndarray
            The observation symbols which are visited.
        """
        from deeptime.markov import compute_dtrajs_effective, count_states
        if dtrajs is None:
            raise ValueError("Needs nonempty dtrajs to evaluate nonempty obs.")
        dtrajs = ensure_dtraj_list(dtrajs)
        dtrajs_lagged_strided = compute_dtrajs_effective(
            dtrajs, self.transition_model.lagtime, self.transition_model.count_model.n_states_full, self.stride
        )
        obs = np.where(count_states(dtrajs_lagged_strided) > 0)[0]
        return obs

    def states_largest(self, directed=True, connectivity_threshold='1/n') -> np.ndarray:
        r"""
        Selects hidden states which represent the largest connected set.

        Parameters
        ----------
        directed : bool, optional, default=True
            Whether the connectivity is strong (directed) or weak (undirected)
        connectivity_threshold : str or int, optional, default='1/n'
            A connectivity threshold which can be employed to only consider edges with a certain minimum weight.

        Returns
        -------
        The largest connected set of hidden states
        """
        return self._select_states(connectivity_threshold, 'largest-strong' if directed else 'largest-weak')

    def submodel_largest(self, directed=True, connectivity_threshold='1/n', observe_nonempty=True, dtrajs=None):
        r"""
        Returns the largest connected sub-HMM. By default this means that the largest connected set of hidden states
        and the set of visited observable states is selected.

        Parameters
        ----------
        directed : bool, optional, default=True
            Whether the connectivity is based on a directed graph (strong connectiviy) or undirected (weak connectivity)
        connectivity_threshold : str or int, optional, default='1/n'
            The connectivity threshold required to consider two hidden states connected.
        observe_nonempty : bool, optional, default=True
            Whether the observable state set should be restricted to visited observable states. If True, dtrajs must
            be provided.
        dtrajs : array_like, optional, default=None
            Observable state trajectory or a list thereof to evaluate visited observable states.

        Returns
        -------
        sub_hmm : HiddenMarkovModel
            The restricted HMM.
        """
        states = self.states_largest(directed=directed, connectivity_threshold=connectivity_threshold)
        obs = self.nonempty_obs(dtrajs) if observe_nonempty else None
        return self.submodel(states=states, obs=obs)

    def states_populous(self, strong=True, connectivity_threshold='1/n'):
        r"""
        Retrieves the hidden states which are most populated and connected.

        Parameters
        ----------
        strong : bool, optional, default=True
            Whether the connectivity is evaluated based on a directed or on an undirected graph.
        connectivity_threshold : str or int, optional, default=None
            Minimum weight so that two states are considered connected.

        Returns
        -------
        states : np.ndarray
            Most populated set of states
        """
        return self._select_states(connectivity_threshold, 'populous-strong' if strong else 'populous-weak')

    def submodel_populous(self, directed=True, connectivity_threshold='1/n', observe_nonempty=True, dtrajs=None):
        """ Returns the most populous connected sub-HMM.

        Parameters
        ----------
        directed : bool, optional, default=True
            Whether the connectivity is based on a directed graph (strong connectiviy) or undirected (weak connectivity)
        connectivity_threshold : str or int, optional, default='1/n'
            The connectivity threshold required to consider two hidden states connected.
        observe_nonempty : bool, optional, default=True
            Whether the observable state set should be restricted to visited observable states. If True, dtrajs must
            be provided.
        dtrajs : array_like, optional, default=None
            Observable state trajectory or a list thereof to evaluate visited observable states.

        Returns
        -------
        hmm : HiddenMarkovModel
            The restricted HMM.

        """
        states = self.states_populous(strong=directed, connectivity_threshold=connectivity_threshold)
        obs = self.nonempty_obs(dtrajs) if observe_nonempty else None
        return self.submodel(states=states, obs=obs)

    def submodel_disconnect(self, connectivity_threshold='1/n'):
        """Disconnects sets of hidden states that are barely connected

        Runs a connectivity check excluding all transition counts below
        connectivity_threshold. The transition matrix and stationary distribution
        will be re-estimated. Note that the resulting transition matrix
        may have both strongly and weakly connected subsets.

        Parameters
        ----------
        connectivity_threshold : float or '1/n'
            minimum number of counts to consider a connection between two states.
            Counts lower than that will count zero in the connectivity check and
            may thus separate the resulting transition matrix. The default
            evaluates to 1/n_states.

        Returns
        -------
        hmm : HiddenMarkovModel
            The restricted HMM.
        """
        lcc = self.transition_model.count_model.connected_sets(connectivity_threshold=connectivity_threshold)[0]
        return self.submodel(lcc)

    # ================================================================================================================
    # Properties inherited from transition model
    # ================================================================================================================


def viterbi(transition_matrix: np.ndarray, state_probability_trajectory: np.ndarray, initial_distribution: np.ndarray):
    """ Estimate the hidden pathway of maximum likelihood using the Viterbi algorithm.

    Parameters
    ----------
    transition_matrix : ndarray((N,N), dtype = float)
        transition matrix of the hidden states
    state_probability_trajectory : ndarray((T,N), dtype = float)
        pobs[t,i] is the observation probability for observation at time t given hidden state i
    initial_distribution : ndarray((N), dtype = float)
        initial distribution of hidden states

    Returns
    -------
    q : numpy.array shape (T)
        maximum likelihood hidden path

    """
    if state_probability_trajectory.ndim == 1 and transition_matrix.shape[0] == 1:
        # if there is only one state, pad so that there is an additional dimension
        state_probability_trajectory = state_probability_trajectory[..., None]
    return viterbi_impl(transition_matrix=transition_matrix,
                        state_probability_trajectory=state_probability_trajectory,
                        initial_distribution=initial_distribution)
