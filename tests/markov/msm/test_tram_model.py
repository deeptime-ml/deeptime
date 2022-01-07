import numpy as np
import pytest
from deeptime.markov.msm.tram import TRAMModel
from deeptime.markov import TransitionCountEstimator, TransitionCountModel
from deeptime.markov.msm import MarkovStateModelCollection


def random_model(n_therm_states, n_markov_states, transition_matrices=None):
    transition_counts = np.zeros((n_therm_states, n_markov_states, n_markov_states))

    if transition_matrices is None:
        # make stochastic transition matrix
        transition_matrices = np.random.rand(n_therm_states, n_markov_states, n_markov_states)
        transition_matrices /= np.sum(transition_matrices, axis=-1, keepdims=True)

    biased_conf_energies = np.random.rand(n_therm_states, n_markov_states)
    lagrangians = np.random.rand(n_therm_states, n_markov_states)
    modified_state_counts_log = np.log(np.random.rand(n_therm_states, n_markov_states))
    count_models = [TransitionCountModel(counts) for counts in transition_counts]
    # construct model.
    return TRAMModel(count_models, transition_matrices, biased_conf_energies=biased_conf_energies,
                     lagrangian_mult_log=lagrangians,
                     modified_state_counts_log=modified_state_counts_log)

def test_init_tram_model():
    n_therm_states = 2
    n_markov_states = 3

    transition_matrices = np.random.rand(n_therm_states, n_markov_states, n_markov_states)
    transition_matrices /= np.sum(transition_matrices, axis=-1, keepdims=True)

    model = random_model(n_therm_states, n_markov_states, transition_matrices=transition_matrices)

    MEMM = model.markov_state_model_collection
    assert isinstance(MEMM, MarkovStateModelCollection)
    assert (MEMM.transition_matrix == transition_matrices[0]).all()
    assert MEMM.n_connected_msms == n_therm_states
    MEMM.select(1)
    assert (MEMM.transition_matrix == transition_matrices[1]).all()

    assert model.n_markov_states == n_markov_states
    assert model.n_therm_states == n_therm_states


def test_compute_pmf():
    model = random_model(5, 5)
    n_samples = 100

    trajs = np.random.uniform(0, 1, (5, n_samples))
    dtrajs = np.digitize(trajs, np.linspace(0, 1, 5, endpoint=False)) - 1
    binned_trajs = np.digitize(trajs, np.linspace(0, 1, 10, endpoint=False)) - 1
    bias_matrices = np.random.uniform(0, 1, (5, n_samples, 5))

    pmf = model.compute_PMF(dtrajs, bias_matrices, binned_trajs)

    assert len(pmf) == 10
    assert (pmf >= 0).all()

def test_compute_observable():
    model = random_model(5, 5)
    n_samples = 100

    dtrajs = np.random.uniform(0, 5, (5, n_samples))
    bias_matrices = np.random.uniform(0, 1, (5, n_samples, 5))
    obs = np.ones_like(dtrajs)

    res1 = model.compute_observable(dtrajs, bias_matrices, observable_values=obs)
    assert res1 > 0

    # observable should change linearly with observed values
    obs *= -2
    res2 = model.compute_observable(dtrajs, bias_matrices, observable_values=obs)
    assert res2 == res1 * -2
