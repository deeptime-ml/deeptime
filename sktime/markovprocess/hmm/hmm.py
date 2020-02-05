from typing import Union

import numpy as np

from sktime.base import Model
from sktime.markovprocess import MarkovStateModel
from sktime.markovprocess.hmm.output_model import OutputModel


class HiddenMarkovStateModel(Model):

    def __init__(self, transition_model: Union[np.ndarray, MarkovStateModel],
                 observation_probabilities: Union[np.ndarray, OutputModel],
                 initial_distribution: np.ndarray):
        r"""
        Constructs a new hidden markov state model from a (m, m) hidden transition matrix (macro states), an
        observation probability matrix that maps from hidden to observable states (micro states), i.e., a (m, n)-matrix,
        and an initial distribution over the hidden states.

        Parameters
        ----------
        transition_model : (m,m) ndarray or MarkovStateModel
            Transition matrix for hidden (macro) states
        initial_distribution : (m,) ndarray
            Initial distribution of the hidden (macro) states
        observation_probabilities : (m,n) ndarray or OutputModel
            observation probability matrix from hidden to observable (micro) states or OutputModel instance which yields
            the mapping from hidden to observable state.
        """