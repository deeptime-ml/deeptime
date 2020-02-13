from typing import Union, Optional, List

import numpy as np

from sktime.base import Model
from sktime.markovprocess import MarkovStateModel, TransitionCountModel
from sktime.markovprocess.hmm.output_model import OutputModel, DiscreteOutputModel
from sktime.util import ensure_dtraj_list
import sktime.markovprocess.hmm._hmm_bindings as _bindings


class HiddenMarkovStateModel(Model):

    def __init__(self, transition_model: Union[np.ndarray, MarkovStateModel],
                 output_model: Union[np.ndarray, OutputModel],
                 initial_distribution: Optional[np.ndarray] = None,
                 likelihoods: Optional[np.ndarray] = None,
                 state_probabilities: Optional[List[np.ndarray]] = None,
                 initial_count : Optional[np.ndarray] = None,
                 hidden_state_trajectories : Optional[List[np.ndarray]] = None,
                 count_model: Optional[TransitionCountModel] = None):
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
        likelihoods : (k,) ndarray, optional, default=None
            Likelihood progression of the HMM as it was trained for k iterations with Baum-Welch.
        state_probabilities : list of ndarray, optional, default=None
            List of state probabilities for each trajectory that the model was trained on (gammas).
        initial_count : ndarray, optional, default=None
            Initial counts of the hidden (macro) states, computed from the gamma output of the Baum-Welch algorithm
        count_model : TransitionCountModel, optional, default=None
            Transition count model containing count matrix, lagtime, and physical time description
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
        self._likelihoods = likelihoods
        self._state_probabilities = state_probabilities
        self._initial_count = initial_count
        self._hidden_state_trajectories = hidden_state_trajectories
        self._count_model = count_model

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

    @property
    def likelihoods(self) -> Optional[np.ndarray]:
        return self._likelihoods

    @property
    def likelihood(self) -> Optional[float]:
        if self.likelihoods is not None:
            return self.likelihoods[-1]
        return None

    @property
    def gammas(self) -> Optional[List[np.ndarray]]:
        return self._state_probabilities

    @property
    def count_model(self) -> Optional[TransitionCountModel]:
        return self._count_model

    @property
    def transition_counts(self) -> Optional[np.ndarray]:
        return self.count_model.count_matrix if self.count_model is not None else None

    @property
    def initial_count(self) -> Optional[np.ndarray]:
        return self._initial_count

    @property
    def hidden_state_trajectories(self) -> Optional[List[np.ndarray]]:
        return self._hidden_state_trajectories

    @property
    def output_probabilities(self):
        if isinstance(self.output_model, DiscreteOutputModel):
            return self.output_model.output_probabilities
        # todo can this be generalized?
        raise ValueError("Output probabilities are only available for HMMs with discrete output model.")

    @property
    def stationary_distribution_obs(self):
        if isinstance(self.output_model, DiscreteOutputModel):
            return np.dot(self.transition_model.stationary_distribution, self.output_probabilities)
        raise RuntimeError("only available for discrete output model")

    @property
    def lifetimes(self):
        return -self.transition_model.lagtime / np.log(np.diag(self.transition_model.transition_matrix))

    def compute_viterbi_paths(self, observations: List[np.ndarray]):
        """Computes the Viterbi paths using the current HMM model"""
        observations = ensure_dtraj_list(observations)
        A = self.transition_model.transition_matrix
        pi = self.initial_distribution
        state_probabilities = [
            self.output_model.to_state_probability_trajectory(obs) for obs in observations
        ]
        paths = [viterbi(A, obs, pi) for obs in state_probabilities]
        return paths


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
    return _bindings.util.viterbi(transition_matrix=transition_matrix,
                                  state_probability_trajectory=state_probability_trajectory,
                                  initial_distribution=initial_distribution)
