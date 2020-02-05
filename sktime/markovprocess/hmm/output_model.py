import abc
from typing import Optional

import numpy as np

import sktime.markovprocess.hmm._hmm_bindings.output_models as _bindings


class OutputModel(metaclass=abc.ABCMeta):

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
        ...

    @abc.abstractmethod
    def generate_observation_trajectory(self, hidden_state_trajectory: np.ndarray) -> np.ndarray:
        ...

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
                 ignore_outliers:bool = False):
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



