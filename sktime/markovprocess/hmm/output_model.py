import abc
from typing import Optional, List

import numpy as np

import sktime.markovprocess.hmm._hmm_bindings.output_models as _bindings

from sktime.base import Model


class OutputModel(Model, metaclass=abc.ABCMeta):

    def __init__(self, n_hidden_states: int, n_observable_states: int, ignore_outliers: bool = True):
        self._n_hidden_states = n_hidden_states
        self._n_observable_states = n_observable_states
        self._ignore_outliers = ignore_outliers

    @property
    def ignore_outliers(self) -> bool:
        return self._ignore_outliers

    @ignore_outliers.setter
    def ignore_outliers(self, value: bool):
        self._ignore_outliers = value

    @property
    def n_hidden_states(self) -> int:
        return self._n_hidden_states

    @property
    def n_observable_states(self) -> int:
        return self._n_observable_states

    @abc.abstractmethod
    def to_state_probability_trajectory(self, observations: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def generate_observation_trajectory(self, hidden_state_trajectory: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def sample(self, observations: List[np.ndarray]) -> None:
        pass

    @abc.abstractmethod
    def fit(self, observations: List[np.ndarray], weights: List[np.ndarray]):
        r"""
        Fits the output model given the observations and weights.

        Parameters
        ----------
        observations : list of ndarray
            A list of K observation trajectories
        weights : list of ndarray
            A list of K weight matrices, each having length T_k of their corresponding observation trajectory.
            Evaluating weights[k][t,n] should yield the weight assignment from observations[k][t] to state index n.

        Returns
        -------
        A reference to this output model instance.
        """
        pass

    @staticmethod
    def _handle_outliers(state_probability_trajectory: np.ndarray) -> None:
        r"""
        Takes a state probability trajectory, shape (T, n_hidden_states), and sets all probabilities which sum up to
        0 to uniform (or rather constant one, the trajectory is re-normalized later to make it row-stochastic again).
        This operation is preformed in-place.

        Parameters
        ----------
        state_probability_trajectory : (T, n_hidden) ndarray
            state probability trajectory: for each time step, the probability to be in a certain hidden state
        """
        _bindings.handle_outliers(state_probability_trajectory)


class DiscreteOutputModel(OutputModel):

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
        return self._prior

    @property
    def output_probabilities(self):
        return self._output_probabilities

    def to_state_probability_trajectory(self, observations: np.ndarray) -> np.ndarray:
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
        # normalize
        self._output_probabilities /= np.sum(self._output_probabilities, axis=1)[:, None]
        return self

    def sample(self, observations_per_state: List[np.ndarray]) -> None:
        r"""

        Parameters
        ----------
        observations_per_state

        Returns
        -------

        """
        # todo why ignore observation states w/o counts in a hidden state parameter sample?
        _bindings.discrete.sample(observations_per_state, self.output_probabilities, self.prior)


class GaussianOutputModel(OutputModel):

    def __init__(self, n_states: int, means: Optional[np.ndarray] = None, sigmas: Optional[np.ndarray] = None,
                 ignore_outliers: bool = True):
        if means is None:
            means = np.zeros((n_states,))
        if sigmas is None:
            sigmas = np.zeros((n_states,))
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
        return self._means

    @property
    def sigmas(self):
        return self._sigmas

    def to_state_probability_trajectory(self, observations: np.ndarray) -> np.ndarray:
        state_probabilities = _bindings.gaussian.to_output_probability_trajectory(observations, self.means, self.sigmas)
        if self.ignore_outliers:
            self._handle_outliers(state_probabilities)
        return state_probabilities

    def generate_observation_trajectory(self, hidden_state_trajectory: np.ndarray) -> np.ndarray:
        """
        Generate synthetic observation data from a given state sequence.

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
        >>> s_t = np.random.randint(0, output_model.n_states, size=[nobs])

        Generate a synthetic trajectory

        >>> o_t = output_model.generate_observation_trajectory(s_t)

        """
        return _bindings.gaussian.generate_observation_trajectory(hidden_state_trajectory, self.means, self.sigmas)

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

        Update the observation model parameters my a maximum-likelihood fit.

        >>> output_model.fit(observations, weights)

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

    def sample(self, observations: List[np.ndarray]) -> None:
        """
        Sample a new set of distribution parameters given a sample of observations from the given state.

        Both the internal parameters and the attached HMM model are updated.

        Parameters
        ----------
        observations :  [ numpy.array with shape (N_k,) ] with `n_states` elements
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

            observations_in_state = observations[state_index]
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
