import numpy as np
import pytest
from deeptime.markov.msm.tram import TRAMModel
from deeptime.markov import TransitionCountEstimator, TransitionCountModel
from deeptime.markov.msm import MarkovStateModelCollection


def test_init_tram_model():
    n_markov_states = 3
    n_therm_states = 2
    transition_counts = np.zeros((n_therm_states, n_markov_states, n_markov_states))

    # make stochastic transition matrix
    transition_matrices = np.random.rand(n_therm_states, n_markov_states, n_markov_states)
    transition_matrices /= np.sum(transition_matrices, axis=-1, keepdims=True)

    biased_conf_energies = np.random.rand(n_therm_states, n_markov_states)
    count_models = [TransitionCountModel(counts) for counts in transition_counts]

    # construct model.
    model = TRAMModel(count_models, transition_matrices, biased_conf_energies=biased_conf_energies)
    MEMM = model.markov_state_model_collection
    assert isinstance(MEMM, MarkovStateModelCollection)
    assert (MEMM.transition_matrix == transition_matrices[0]).all()
    assert MEMM.n_connected_msms == n_therm_states
    MEMM.select(1)
    assert (MEMM.transition_matrix == transition_matrices[1]).all()
