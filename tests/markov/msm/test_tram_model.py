import numpy as np
from deeptime.markov import TransitionCountEstimator, TransitionCountModel
from deeptime.markov.msm import MarkovStateModelCollection, TRAM, TRAMModel


def make_random_model(n_therm_states, n_markov_states, transition_matrices=None):
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

    model = make_random_model(n_therm_states, n_markov_states, transition_matrices=transition_matrices)

    memm = model.msm_collection
    np.testing.assert_(isinstance(memm, MarkovStateModelCollection))
    np.testing.assert_equal(memm.transition_matrix, transition_matrices[0])
    np.testing.assert_equal(memm.n_connected_msms, n_therm_states)
    memm.select(1)
    np.testing.assert_equal(memm.transition_matrix, transition_matrices[1])

    np.testing.assert_equal(model.n_markov_states, n_markov_states)
    np.testing.assert_equal(model.n_therm_states, n_therm_states)


def test_compute_pmf():
    model = make_random_model(5, 5)
    n_samples = 100

    trajs = np.random.uniform(0, 1, (5, n_samples))
    dtrajs = np.digitize(trajs, np.linspace(0, 1, 5, endpoint=False)) - 1
    binned_trajs = np.digitize(trajs, np.linspace(0, 1, 10, endpoint=False)) - 1
    bias_matrices = np.random.uniform(0, 1, (5, n_samples, 5))

    pmf = model.compute_PMF(dtrajs, bias_matrices, binned_trajs)

    np.testing.assert_equal(len(pmf), 10)
    np.testing.assert_(np.min(pmf >= 0))


def test_compute_observable():
    model = make_random_model(5, 5)
    n_samples = 100

    dtrajs = np.random.uniform(0, 5, (5, n_samples))
    bias_matrices = np.random.uniform(0, 1, (5, n_samples, 5))
    obs = np.ones_like(dtrajs)

    res1 = model.compute_observable(obs, dtrajs, bias_matrices)
    np.testing.assert_(res1 > 0)

    # observable should change linearly with observed values
    obs *= -2
    res2 = model.compute_observable(obs, dtrajs, bias_matrices)
    np.testing.assert_equal(res2, res1 * -2)
