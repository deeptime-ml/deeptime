from typing import Union, Optional

import numpy as np

from sktime.base import Model
from sktime.markovprocess import MarkovStateModel
from sktime.markovprocess.hmm.output_model import OutputModel, DiscreteOutputModel


class HiddenMarkovStateModel(Model):

    def __init__(self, transition_model: Union[np.ndarray, MarkovStateModel],
                 output_model: Union[np.ndarray, OutputModel],
                 initial_distribution: Optional[np.ndarray] = None):
        r"""
        Constructs a new hidden markov state model from a (m, m) hidden transition matrix (macro states), an
        observation probability matrix that maps from hidden to observable states (micro states), i.e., a (m, n)-matrix,
        and an initial distribution over the hidden states.

        Parameters
        ----------
        transition_model : (m,m) ndarray or MarkovStateModel
            Transition matrix for hidden (macro) states
        output_model : (m,n) ndarray or OutputModel
            observation probability matrix from hidden to observable (micro) states or OutputModel instance which yields
            the mapping from hidden to observable state.
        initial_distribution : (m,) ndarray, optional, default=None
            Initial distribution of the hidden (macro) states. Default is uniform.
        """
        if isinstance(transition_model, np.ndarray):
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

    @property
    def transition_model(self) -> MarkovStateModel:
        return self._transition_model

    @property
    def output_model(self) -> OutputModel:
        return self._output_model

    @property
    def initial_distribution(self) -> np.ndarray:
        return self._initial_distribution

    @property
    def n_hidden_states(self):
        return self.output_model.n_hidden_states
