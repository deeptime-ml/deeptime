import numpy as np
import pytest
from deeptime.markov.msm.tram import TRAM
from deeptime.markov import TransitionCountEstimator


def get_connected_set_from_dtrajs_input(dtrajs, tram):
    tram.n_markov_states = np.max(np.concatenate(dtrajs)) + 1
    tram.n_therm_states = len(dtrajs)

    dtrajs = [np.asarray(traj) for traj in dtrajs]
    ttrajs = np.asarray([[i] * len(traj) for i, traj in enumerate(dtrajs)])
    bias_matrices = [np.ones((len(dtrajs[i]), len(dtrajs))) for i in range(len(dtrajs))]

    return tram._find_largest_connected_set(ttrajs, dtrajs, bias_matrices)


@pytest.mark.parametrize(
    "test_input,expected",
    [([[1, 2, 3, 2, 1], [4, 5, 6, 5, 4]], [1, 2, 3]),
     ([[1, 2, 3], [3, 4, 5], [5, 3, 2]], [2, 3, 4, 5]),
     ([[1, 2, 3, 2]], [2, 3]),
     ([[1, 2, 3, 2], [3, 2]], [2, 3]),
     ([[1, 2, 3, 2], [3, 2, 1]], [1, 2, 3]),
     ([[1, 2, 3, 2], [3, 4, 3, 4]], [2, 3, 4]),
     ([[1, 2, 1, 3, 2, 7, 7, 7, 6], [3, 4, 3, 3, 4, 5, 6, 6, 5, 4]], [1, 2, 3, 4, 5, 6, 7])],
)
def test_connected_set_summed_count_matrix(test_input, expected):
    tram = TRAM(lagtime=1, count_mode='sliding', connectivity='summed_count_matrix')
    cset = get_connected_set_from_dtrajs_input(test_input, tram)
    assert np.array_equal(cset.state_symbols, np.asarray(expected))


@pytest.mark.parametrize(
    "test_input,expected",
    [([[1, 2, 3, 2, 1], [4, 5, 6, 5, 4]], [1, 2, 3]),
     ([[1, 2, 3, 2, 1], [3, 4, 5, 4, 4]], [1, 2, 3]),
     ([[1, 2, 3, 2, 1], [4, 3, 4, 5, 4]], [1, 2, 3, 4, 5]),
     ([[1, 2, 3], [3, 4, 5], [5, 3, 2]], [3]),
     ([[1, 2, 3, 2], [3, 1, 2]], [2, 3]),
     ([[1, 2, 1, 3, 2, 7, 7, 6], [3, 4, 3, 3, 4, 5, 6, 5, 4]], [1, 2, 3, 4, 5, 6]),
     ([[1, 2, 3, 2, 1], [3, 5, 6, 5, 3], [3, 5, 6, 5, 3]], [1, 2, 3, 5, 6])]
)
def test_connected_set_post_hoc_RE(test_input, expected):
    tram = TRAM(lagtime=1, count_mode='sliding', connectivity='post_hoc_RE')
    cset = get_connected_set_from_dtrajs_input(test_input, tram)
    assert np.array_equal(cset.state_symbols, np.asarray(expected))


@pytest.mark.parametrize(
    "test_input,expected",
    [([[1, 2, 3, 2, 1], [4, 5, 6, 5, 4]], [1, 2, 3]),
     ([[1, 2, 3, 2, 1], [4, 3, 4, 5, 4]], [1, 2, 3]),
     ([[1, 2, 3, 2], [3, 1, 2]], [2, 3]),
     ([[1, 2, 1, 3, 2, 7, 7, 6], [3, 4, 3, 3, 4, 5, 6, 5, 4]], [3, 4, 5, 6]),
     ([[1, 2, 3, 2, 1], [3, 5, 6, 5, 3], [3, 5, 6, 5, 3]], [1, 2, 3])]
)
def test_connected_set_post_hoc_RE_no_connectivity(test_input, expected):
    tram = TRAM(lagtime=1, count_mode='sliding', connectivity='post_hoc_RE')
    tram.connectivity_factor = 0.0
    cset = get_connected_set_from_dtrajs_input(test_input, tram)
    assert np.array_equal(cset.state_symbols, np.asarray(expected))


@pytest.mark.parametrize(
    "test_input,expected",
    [([[1, 2, 3, 2, 1], [4, 5, 6, 5, 4]], [1, 2, 3]),
     ([[1, 2, 3, 2, 1], [3, 4, 5, 4, 4]], [1, 2, 3]),
     ([[1, 2, 3, 2, 1], [4, 3, 4, 5, 4]], [1, 2, 3, 4, 5]),
     ([[1, 2, 3], [3, 4, 5], [5, 3, 2]], [3]),
     # ([[0, 1, 2, 3]], []),
     ([[1, 2, 3, 2], [3, 1, 2]], [2, 3]),
     ([[1, 2, 1, 3, 2, 7, 7, 6], [3, 4, 3, 3, 4, 5, 6, 5, 4]], [3, 4, 5, 6]),
     ([[1, 2, 3, 2, 1], [3, 5, 6, 5, 3], [3, 5, 6, 5, 3]], [1, 2, 3, 5, 6])]
)
def test_connected_set_BAR_variance(test_input, expected):
    tram = TRAM(lagtime=1, count_mode='sliding', connectivity='BAR_variance', connectivity_factor=1.0)
    cset = get_connected_set_from_dtrajs_input(test_input, tram)
    assert np.array_equal(cset.state_symbols, np.asarray(expected))


@pytest.mark.parametrize(
    "test_input,expected",
    [([[1, 2, 3, 2, 1], [4, 5, 6, 5, 4]], [1, 2, 3]),
     ([[1, 2, 3, 2, 1], [4, 3, 4, 5, 4]], [1, 2, 3]),
     ([[1, 2, 3, 2], [3, 1, 2]], [2, 3]),
     ([[1, 2, 1, 3, 2, 7, 7, 6], [3, 4, 3, 3, 4, 5, 6, 5, 4]], [3, 4, 5, 6]),
     ([[1, 2, 3, 2, 1], [3, 5, 6, 5, 3], [3, 5, 6, 5, 3]], [1, 2, 3])]
)
def test_connected_set_BAR_variance_no_connectivity(test_input, expected):
    tram = TRAM(lagtime=1, count_mode='sliding', connectivity='BAR_variance')
    tram.connectivity_factor = 0.0
    cset = get_connected_set_from_dtrajs_input(test_input, tram)
    assert np.array_equal(cset.state_symbols, np.asarray(expected))


def test_restrict_to_connected_set():
    tram = TRAM()
    input = np.asarray([[0, 1, 2, 3, 4, 5, 1], [2, 4, 2, 1, 3, 1, 4]])
    tram.n_therm_states = len(input)
    counts_model = TransitionCountEstimator(1, 'sliding').fit_fetch(input)
    tram.largest_connected_set = counts_model.submodel([1, 2, 3])
    output = tram._restrict_to_connected_set(input)
    assert np.array_equal(output, [[-1, 1, 2, 3, -1, -1, 1], [2, -1, 2, 1, 3, 1, -1]])


def test_make_count_models():
    tram = TRAM()
    input = [np.asarray([-1, 1, 2, 3, -1, -1, 1]), np.asarray([2, -5, 0, 1, 3, 1, 4]), np.asarray([-1, -1, -1, -1])]
    tram.n_therm_states = len(input)
    tram.n_markov_states = np.max(np.concatenate(input)) + 1
    state_counts, transition_counts = tram._make_count_models(input)
    assert len(tram.count_models) == tram.n_therm_states
    assert state_counts.shape == (tram.n_therm_states, tram.n_markov_states)
    assert transition_counts.shape == (tram.n_therm_states, tram.n_markov_states, tram.n_markov_states)
    assert np.array_equal(tram.count_models[0].state_symbols, [0,1,2,3])
    assert np.array_equal(tram.count_models[1].state_symbols, [0,1,2,3,4])
    assert np.array_equal(tram.count_models[2].state_symbols, [0,1,2,3,4])
    assert np.all(state_counts[2] == 0)