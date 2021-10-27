import abc
from typing import Optional, List

import numpy as np
from ._hmm_bindings import output_models as _bindings

from ...base import Model


class OutputModel(Model, metaclass=abc.ABCMeta):
    r""" Output model superclass. Contains basic functionality and interfaces which output models are supposed
    to implement.

    Parameters
    ----------
    n_hidden_states : int
        Number of hidden states.
    n_observable_states : int
        Number of observable states if discrete state space, otherwise -1
    ignore_outliers : bool, optional, default=True
        By outliers observations that have zero probability given the
        current model are meant. :code:`ignore_outliers=True` means that outliers will be treated
        as if no observation was made, which is equivalent to making this
        observation with equal probability from any hidden state.
        :code:`ignore_outliers=False` means that an Exception or in the worst case an
        unhandled crash will occur if an outlier is observed.

    See Also
    --------
    DiscreteOutputModel
    GaussianOutputModel
    """

    def __init__(self, n_hidden_states: int, n_observable_states: int, ignore_outliers: bool = True):
        super().__init__()
        self._n_hidden_states = n_hidden_states
        self._n_observable_states = n_observable_states
        self._ignore_outliers = ignore_outliers

    @property
    def ignore_outliers(self) -> bool:
        r""" By outliers observations that have zero probability given the
        current model are meant. :code:`ignore_outliers=True` means that outliers will be treated
        as if no observation was made, which is equivalent to making this
        observation with equal probability from any hidden state.
        :code:`ignore_outliers=False` means that an Exception or in the worst case an
        unhandled crash will occur if an outlier is observed.
        """
        return self._ignore_outliers

    @ignore_outliers.setter
    def ignore_outliers(self, value: bool):
        self._ignore_outliers = value

    @property
    def n_hidden_states(self) -> int:
        r""" Number of hidden states. """
        return self._n_hidden_states

    @property
    def n_observable_states(self) -> int:
        r""" Number of observable states, can be -1 if not applicable (e.g., in a continous observable space). """
        return self._n_observable_states

    @abc.abstractmethod
    def to_state_probability_trajectory(self, observations: np.ndarray) -> np.ndarray:
        r"""Converts a list of observations to hidden state probabilities, i.e., for each observation :math:`o_t`,
        one obtains a vector :math:`p_t\in\mathbb{R}^{n_\mathrm{hidden}}` denoting how probable that particular
        observation is in one of the hidden states.

        Parameters
        ----------
        observations : (T, d) ndarray
            Array of observations.

        Returns
        -------
        state_probabilities : (T, n_hidden) ndarray
            State probability trajectory.
        """

    @abc.abstractmethod
    def generate_observation_trajectory(self, hidden_state_trajectory: np.ndarray) -> np.ndarray:
        r""" Generates a synthetic trajectory in observation space given a trajectory in hidden state space.

        Parameters
        ----------
        hidden_state_trajectory : (T, 1) ndarray
            Hidden state trajectory.

        Returns
        -------
        observations : (T, d) ndarray
            Observation timeseries :math:`\{o_t\}_t`, where :math:`o_t` is the observation
            associated to hidden state :math:`s_t`.
        """

    @abc.abstractmethod
    def sample(self, observations_per_state: List[np.ndarray]) -> None:
        r""" Samples a new set of parameters for this output model given a list of observations for each hidden
        state.

        Parameters
        ----------
        observations_per_state : list of ndarray
            The observations per state, i.e., :code:`len(observations_per_state)` must evaluate to the value
            of :attr:`n_hidden_states`.
        """

    @abc.abstractmethod
    def submodel(self, states: Optional[np.ndarray] = None, obs: Optional[np.ndarray] = None):
        r"""Restricts this model to a set of hidden states and observable states (if applicable).

        Parameters
        ----------
        states : ndarray, optional, default=None
            The hidden states to restrict to, per default no restriction.
        obs : ndarray, optional, default=None
            The observable states to restrict to (if applicable), per default no restriction.

        Returns
        -------
        submodel : OutputModel
            The restricted output model.
        """

    @abc.abstractmethod
    def fit(self, observations: List[np.ndarray], weights: List[np.ndarray]):
        r""" Fits the output model given the observations and weights.

        Parameters
        ----------
        observations : list of ndarray
            A list of K observation trajectories
        weights : list of ndarray
            A list of K weight matrices, each having length T_k of their corresponding observation trajectory.
            Evaluating weights[k][t,n] should yield the weight assignment from observations[k][t] to state index n.

        Returns
        -------
        self : OutputModel
            Reference to self.
        """

    @staticmethod
    def _handle_outliers(state_probability_trajectory: np.ndarray) -> None:
        r"""Takes a state probability trajectory, shape (T, n_hidden_states), and sets all probabilities which sum up to
        0 to uniform (or rather constant one, the trajectory is re-normalized later to make it row-stochastic again).
        This operation is preformed in-place.

        Parameters
        ----------
        state_probability_trajectory : (T, n_hidden) ndarray
            state probability trajectory: for each time step, the probability to be in a certain hidden state
        """
        _bindings.handle_outliers(state_probability_trajectory)


class DiscreteOutputModel(OutputModel):
    r"""HMM output probability model using discrete symbols. This corresponds to the "standard" HMM that is
    classically used in the literature.

    Parameters
    ----------
    output_probabilities : ((N, M) dtype=float32 or float64) ndarray
        Row-stochastic output probability matrix for :code:`N` hidden states and :code:`M` observable symbols.
    prior : None or type and shape of output_probabilities, optional, default=None
        Prior for the initial distribution of the HMM. Currently implements the Dirichlet prior that is
        conjugate to the Dirichlet distribution of :math:`b_i`, which is sampled from

        .. math:
            b_i \sim \prod_j b_{ij}_i^{a_{ij} + n_{ij} - 1},

        where :math:`n_{ij}` are the number of times symbol :math:`j` has been observed when the hidden trajectory
        was in state :math:`i` and :math:`a_{ij}` is the prior count. The default prior=None corresponds
        to :math:`a_{ij} = 0`. This option ensures coincidence between sample mean an MLE.
    ignore_outliers : bool, optional, default=False
        Whether to ignore outliers, see :attr:`ignore_outliers`.
    """
    def __init__(self, output_probabilities: np.ndarray, prior: Optional[np.ndarray] = None,
                 ignore_outliers: bool = False):
        if output_probabilities.ndim != 2:
            raise ValueError("Discrete output model requires two-dimensional output probability matrix!")
        if np.any(output_probabilities < 0) or not np.allclose(output_probabilities.sum(axis=-1), 1.):
            raise ValueError("Output probabilities must be row-stochastic, i.e., the row-sums must be one and all "
                             "elements must be non-negative.")
        super(DiscreteOutputModel, self).__init__(n_hidden_states=output_probabilities.shape[0],
                                                  n_observable_states=output_probabilities.shape[1],
                                                  ignore_outliers=ignore_outliers)
        if prior is None:
            prior = np.zeros_like(output_probabilities)
        if prior.shape != output_probabilities.shape:
            raise ValueError(f"Prior must have same shape as output probabilities: {prior.shape} "
                             f"!= {output_probabilities.shape}.")
        self._output_probabilities = output_probabilities
        self._prior = prior

    @property
    def prior(self) -> np.ndarray:
        r""" Prior matrix. """
        return self._prior

    @property
    def output_probabilities(self):
        r""" A row-stochastic matrix of shape (:attr:`n_hidden_states`, :attr:`n_observable_states`) describing
        the (conditional) discrete probability distribution of observable states given one hidden state.
        """
        return self._output_probabilities

    def to_state_probability_trajectory(self, observations: np.ndarray) -> np.ndarray:
        r""" Returns the output probabilities for an entire trajectory and all hidden states.

        Parameters
        ----------
        observations : ndarray((T), dtype=int)
            a discrete trajectory of length T

        Return
        ------
        p_o : ndarray (T,N)
            The probability of generating the symbol at time point t from any of the N hidden states.
        """
        state_probabilities = _bindings.discrete.to_output_probability_trajectory(observations,
                                                                                  self.output_probabilities)
        if self.ignore_outliers:
            self._handle_outliers(state_probabilities)
        return state_probabilities

    def generate_observation_trajectory(self, hidden_state_trajectory: np.ndarray) -> np.ndarray:
        return _bindings.discrete.generate_observation_trajectory(hidden_state_trajectory, self.output_probabilities)

    def fit(self, observations: List[np.ndarray], weights: List[np.ndarray]):
        # initialize output probability matrix
        self._output_probabilities.fill(0)
        # update output probability matrix (numerator)
        for obs, w in zip(observations, weights):
            _bindings.discrete.update_p_out(obs, w, self._output_probabilities)
        self.normalize()
        return self

    def normalize(self):
        r""" Normalizes output probabilities so they are row-stochastic. """
        self._output_probabilities /= np.sum(self._output_probabilities, axis=1)[:, None]

    def submodel(self, states: Optional[np.ndarray] = None, obs: Optional[np.ndarray] = None):
        if states is None:
            states = np.arange(self.output_probabilities.shape[0])
        if obs is None:
            obs = np.arange(self.output_probabilities.shape[1])
        B = np.copy(self.output_probabilities[np.ix_(states, obs)])
        B /= B.sum(axis=1)[:, None]
        if self.prior is not None:
            prior = np.copy(self.prior[np.ix_(states, obs)])
        else:
            prior = None
        return DiscreteOutputModel(B, prior, self.ignore_outliers)

    def sample(self, observations_per_state: List[np.ndarray]) -> None:
        r"""
        Sample a new set of distribution parameters given a sample of observations from the given state.
        The internal parameters are updated.

        Parameters
        ----------
        observations_per_state :  [ numpy.array with shape (N_k,) ] of length :attr:`n_hidden_states`
            observations[k] are all observations associated with hidden state k

        Examples
        --------
        Initialize output model

        >>> B = np.array([[0.5, 0.5], [0.1, 0.9]])
        >>> output_model = DiscreteOutputModel(B)

        Sample given observation

        >>> obs = [np.asarray([0, 0, 0, 1, 1, 1]),
        ...        np.asarray([1, 1, 1, 1, 1, 1])]
        >>> output_model.sample(obs)
        """
        # todo why ignore observation states w/o counts in a hidden state parameter sample?
        _bindings.discrete.sample(observations_per_state, self.output_probabilities, self.prior)


class GaussianOutputModel(OutputModel):
    r""" HMM output probability model using one-dimensional Gaussians.

    Parameters
    ----------
    n_states : int
        number of hidden states
    means : array_like, optional, default=None
        means of the output Gaussians, length must match number of hidden states
    sigmas : array_like, optional, default=None
        sigmas of the output Gaussians, length must match number of hidden states
    ignore_outliers : bool, optional, default=True
        whether to ignore outliers which could cause numerical instabilities
    """

    def __init__(self, n_states: int, means=None, sigmas=None,
                 ignore_outliers: bool = True):
        if means is None:
            means = np.zeros((n_states,))
        else:
            means = np.asarray(means)
        if sigmas is None:
            sigmas = np.zeros((n_states,))
        else:
            sigmas = np.asarray(sigmas)
        means, sigmas = means.squeeze(), sigmas.squeeze()
        if means.ndim != 1 or sigmas.ndim != 1:
            raise ValueError("Means and sigmas must be one-dimensional.")
        if means.shape[0] != n_states or sigmas.shape[0] != n_states:
            raise ValueError(f"The number of means and sigmas provided ({means.shape[0]} and {sigmas.shape[0]}, "
                             f"respectively) must match the number of output states.")
        self._means = means
        self._sigmas = sigmas

        super(GaussianOutputModel, self).__init__(n_hidden_states=n_states, n_observable_states=-1,
                                                  ignore_outliers=ignore_outliers)

    @property
    def means(self):
        r""" Mean values of Gaussian output densities. """
        return self._means

    @property
    def sigmas(self):
        r""" Standard deviations of Gaussian output densities. """
        return self._sigmas

    def to_state_probability_trajectory(self, observations: np.ndarray) -> np.ndarray:
        state_probabilities = _bindings.gaussian.to_output_probability_trajectory(observations, self.means, self.sigmas)
        if self.ignore_outliers:
            self._handle_outliers(state_probabilities)
        return state_probabilities

    def generate_observation_trajectory(self, hidden_state_trajectory: np.ndarray) -> np.ndarray:
        """ Generate synthetic observation data from a given state sequence.

        Parameters
        ----------
        hidden_state_trajectory : numpy.array with shape (T,) of int type
            s_t[t] is the hidden state sampled at time t

        Returns
        -------
        o_t : numpy.array with shape (T,) of type dtype
            o_t[t] is the observation associated with state s_t[t]

        Examples
        --------

        Generate an observation model and synthetic state trajectory.

        >>> nobs = 1000
        >>> output_model = GaussianOutputModel(n_states=3, means=[-1, 0, +1], sigmas=[0.5, 1, 2])
        >>> s_t = np.random.randint(0, output_model.n_hidden_states, size=[nobs])

        Generate a synthetic trajectory

        >>> o_t = output_model.generate_observation_trajectory(s_t)

        """
        return _bindings.gaussian.generate_observation_trajectory(hidden_state_trajectory, self.means, self.sigmas)

    def submodel(self, states: Optional[np.ndarray] = None, obs: Optional[np.ndarray] = None):
        if states is None:
            states = np.arange(self.means.shape[0])
        return GaussianOutputModel(len(states), means=self.means[states], sigmas=self.sigmas[states],
                                   ignore_outliers=self.ignore_outliers)

    def fit(self, observations: List[np.ndarray], weights: List[np.ndarray]):
        """
        Fits the output model given the observations and weights

        Parameters
        ----------
        observations : [ ndarray(T_k,) ] with K elements
            A list of K observation trajectories, each having length T_k and d dimensions
        weights : [ ndarray(T_k,n_states) ] with K elements
            A list of K weight matrices, each having length T_k
            weights[k][t,n] is the weight assignment from observations[k][t] to state index n

        Examples
        --------

        Generate an observation model and samples from each state.

        >>> ntrajectories = 3
        >>> nobs = 1000
        >>> output_model = GaussianOutputModel(n_states=3, means=np.array([-1, 0, +1]), sigmas=np.array([0.5, 1, 2]))
        >>> observations = [ np.random.randn(nobs) for _ in range(ntrajectories) ] # random observations
        >>> weights = [ np.random.dirichlet([2, 3, 4], size=nobs) for _ in range(ntrajectories) ] # random weights

        Update the observation model parameters my a maximum-likelihood fit. Fit returns self.

        >>> output_model = output_model.fit(observations, weights)

        """
        if self.means.dtype == np.float32:
            means, sigmas = _bindings.gaussian.fit32(self.n_hidden_states, observations, weights)
        else:
            means, sigmas = _bindings.gaussian.fit64(self.n_hidden_states, observations, weights)

        self._means = means
        self._sigmas = sigmas
        if np.any(self._sigmas < np.finfo(self._sigmas.dtype).eps):
            raise RuntimeError('at least one sigma is too small to continue.')
        return self

    def sample(self, observations_per_state: List[np.ndarray]) -> None:
        """
        Sample a new set of distribution parameters given a sample of observations from the given state.

        The internal parameters are updated accordingly.

        Parameters
        ----------
        observations_per_state :  [ numpy.array with shape (N_k,) ] with `n_states` elements
            observations[k] is a set of observations sampled from state `k`

        Examples
        --------

        Generate synthetic observations.

        >>> n_states = 3
        >>> nobs = 1000
        >>> output_model = GaussianOutputModel(n_states=n_states, means=np.array([-1, 0, 1]),
        ...                                    sigmas=np.array([0.5, 1, 2]))
        >>> observations = [output_model.generate_observation_trajectory(np.array([state_index]*nobs))
        ...                 for state_index in range(n_states)]

        Update output parameters by sampling.

        >>> output_model.sample(observations)

        """
        for state_index in range(self.n_hidden_states):
            # Update state emission distribution parameters.

            observations_in_state = observations_per_state[state_index]
            # Determine number of samples in this state.
            nsamples_in_state = len(observations_in_state)

            # Skip update if no observations.
            if nsamples_in_state == 0:
                import warnings
                warnings.warn('Warning: State %d has no observations.' % state_index)
            if nsamples_in_state > 0:  # Sample new mu.
                self.means[state_index] = np.random.randn() * self.sigmas[state_index] / np.sqrt(
                    nsamples_in_state) + np.mean(observations_in_state)
            if nsamples_in_state > 1:  # Sample new sigma
                # This scheme uses the improper Jeffreys prior on sigma^2, P(mu, sigma^2) \propto 1/sigma
                chisquared = np.random.chisquare(nsamples_in_state - 1)
                sigmahat2 = np.mean((observations_in_state - self.means[state_index]) ** 2)
                self.sigmas[state_index] = np.sqrt(sigmahat2) / np.sqrt(chisquared / nsamples_in_state)
