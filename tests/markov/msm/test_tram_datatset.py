import numpy as np
import pytest
from deeptime.markov.msm.tram import TRAMDataset
from deeptime.markov import TransitionCountEstimator, TransitionCountModel


def make_random_input_data(n_therm_states, n_markov_states, n_samples=10, make_ttrajs = True):
    dtrajs = [np.random.randint(0, n_markov_states, size=n_samples) for _ in range(n_therm_states)]
    bias_matrices = [np.random.rand(n_samples, n_therm_states) for _ in range(n_therm_states)]

    if make_ttrajs:
        ttrajs = [np.random.randint(0, n_therm_states, size=n_samples) for _ in range(n_therm_states)]
        return dtrajs, bias_matrices, ttrajs

    return dtrajs, bias_matrices

def get_connected_set_from_dtrajs_input(dtrajs, connectivity, has_ttrajs=True, connectivity_factor=1):
    n_markov_states = np.max(np.concatenate(dtrajs)) + 1
    n_therm_states = len(dtrajs)

    dtrajs = [np.asarray(traj) for traj in dtrajs]
    if has_ttrajs:
        ttrajs =[np.asarray([i] * len(traj)) for i, traj in enumerate(dtrajs)]
    else:
        ttrajs = None

    bias_matrices = [np.ones((len(dtrajs[i]), len(dtrajs))) for i in range(len(dtrajs))]
    tramdata = TRAMDataset(dtrajs=dtrajs, ttrajs=ttrajs, bias_matrices=bias_matrices, n_markov_states=n_markov_states,
                           n_therm_states=n_therm_states, lagtime=1, count_mode='sliding')

    return tramdata._find_largest_connected_set(connectivity=connectivity, connectivity_factor=connectivity_factor)


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
@pytest.mark.parametrize("has_ttrajs", [True, False])
def test_connected_set_summed_count_matrix(test_input, has_ttrajs, expected):
    cset = get_connected_set_from_dtrajs_input(test_input, connectivity='summed_count_matrix', has_ttrajs=has_ttrajs)
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
@pytest.mark.parametrize("has_ttrajs", [True, False])
def test_connected_set_post_hoc_RE(test_input, has_ttrajs, expected):
    cset = get_connected_set_from_dtrajs_input(test_input, connectivity='post_hoc_RE', has_ttrajs=has_ttrajs)
    assert np.array_equal(cset.state_symbols, np.asarray(expected))


@pytest.mark.parametrize(
    "test_input,expected",
    [([[1, 2, 3, 2, 1], [4, 5, 6, 5, 4]], [1, 2, 3]),
     ([[1, 2, 3, 2, 1], [4, 3, 4, 5, 4]], [1, 2, 3]),
     ([[1, 2, 3, 2], [3, 1, 2]], [2, 3]),
     ([[1, 2, 1, 3, 2, 7, 7, 6], [3, 4, 3, 3, 4, 5, 6, 5, 4]], [3, 4, 5, 6]),
     ([[1, 2, 3, 2, 1], [3, 5, 6, 5, 3], [3, 5, 6, 5, 3]], [1, 2, 3])]
)
@pytest.mark.parametrize("has_ttrajs", [True, False])
def test_connected_set_post_hoc_RE_no_connectivity(test_input, has_ttrajs, expected):
    cset = get_connected_set_from_dtrajs_input(test_input, connectivity='post_hoc_RE', has_ttrajs=has_ttrajs,
                                               connectivity_factor=0)
    assert np.array_equal(cset.state_symbols, np.asarray(expected))


@pytest.mark.parametrize(
    "test_input,expected",
    [([[1, 2, 3, 2, 1], [4, 5, 6, 5, 4]], [1, 2, 3]),
     ([[1, 2, 3, 2, 1], [3, 4, 5, 4, 4]], [1, 2, 3]),
     ([[1, 2, 3, 2, 1], [4, 3, 4, 5, 4]], [1, 2, 3, 4, 5]),
     ([[1, 2, 3], [3, 4, 5], [5, 3, 2]], [3]),
     ([[1, 2, 3, 2], [3, 1, 2]], [2, 3]),
     ([[1, 2, 1, 3, 2, 7, 7, 6], [3, 4, 3, 3, 4, 5, 6, 5, 4]], [3, 4, 5, 6]),
     ([[1, 2, 3, 2, 1], [3, 5, 6, 5, 3], [3, 5, 6, 5, 3]], [1, 2, 3, 5, 6])]
)
@pytest.mark.parametrize("has_ttrajs", [True, False])
def test_connected_set_BAR_variance(test_input, has_ttrajs, expected):
    cset = get_connected_set_from_dtrajs_input(test_input, connectivity='BAR_variance', has_ttrajs=has_ttrajs)
    assert np.array_equal(cset.state_symbols, np.asarray(expected))


@pytest.mark.parametrize(
    "test_input,expected",
    [([[1, 2, 3, 2, 1], [4, 5, 6, 5, 4]], [1, 2, 3]),
     ([[1, 2, 3, 2, 1], [4, 3, 4, 5, 4]], [1, 2, 3]),
     ([[1, 2, 3, 2], [3, 1, 2]], [2, 3]),
     ([[1, 2, 1, 3, 2, 7, 7, 6], [3, 4, 3, 3, 4, 5, 6, 5, 4]], [3, 4, 5, 6]),
     ([[1, 2, 3, 2, 1], [3, 5, 6, 5, 3], [3, 5, 6, 5, 3]], [1, 2, 3])]
)
@pytest.mark.parametrize("has_ttrajs", [True, False])
def test_connected_set_BAR_variance_no_connectivity(test_input, has_ttrajs, expected):
    cset = get_connected_set_from_dtrajs_input(test_input, connectivity='BAR_variance', has_ttrajs=has_ttrajs,
                                               connectivity_factor=0)
    assert np.array_equal(cset.state_symbols, np.asarray(expected))


@pytest.mark.parametrize(
    "test_input,expected",
    [([[0, 1, 2, 3, 4, 5, 1], [2, 4, 2, 1, 3, 1, 4]], [[-1, 1, 2, 3, -1, -1, 1], [2, -1, 2, 1, 3, 1, -1]])]
)
def test_restrict_to_connected_set(test_input, expected):
    input = [np.asarray(i) for i in test_input]
    _, bias_matrices = make_random_input_data(2, 7, n_samples=len(input[0]), make_ttrajs=False)
    tramdata = TRAMDataset(dtrajs=test_input, bias_matrices=bias_matrices)
    counts_model = TransitionCountEstimator(1, 'sliding').fit_fetch(input)
    cset = counts_model.submodel([1, 2, 3])
    tramdata.restrict_to_connected_set(cset)
    assert np.array_equal(tramdata.dtrajs, expected)


@pytest.mark.parametrize(
    "lagtime", [1, 3]
)
def test_make_count_models(lagtime):
    tram = TRAM(lagtime=lagtime)
    traj_fragments = [np.asarray([1, 1, 2, 3, 1, 1, 1]), np.asarray([2, 0, 0, 1, 3, 1, 4]), np.asarray([2, 2, 2, 2])]
    tram.n_therm_states = len(traj_fragments)
    tram.n_markov_states = np.max(np.concatenate(traj_fragments)) + 1
    state_counts, transition_counts = tram._construct_count_models([[fragments] for fragments in traj_fragments])
    assert len(tram.count_models) == tram.n_therm_states
    assert state_counts.shape == (tram.n_therm_states, tram.n_markov_states)
    assert transition_counts.shape == (tram.n_therm_states, tram.n_markov_states, tram.n_markov_states)
    assert np.array_equal(tram.count_models[0].state_symbols, [0, 1, 2, 3])
    assert np.array_equal(tram.count_models[1].state_symbols, [0, 1, 2, 3, 4])
    assert np.array_equal(tram.count_models[2].state_symbols, [0, 1, 2])
    for k in range(tram.n_therm_states):
        np.testing.assert_equal(transition_counts[k].sum(), len(traj_fragments[k]) - lagtime)
        np.testing.assert_equal(state_counts[k].sum(), len(traj_fragments[k]))


@pytest.mark.parametrize(
    "input", [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
               [1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
               [1, 0, 1, 1, 1, 1, 1, 1, 1, 1]]]
)
def test_transposed_count_matrices_bug(input):
    tram = TRAM(connectivity='summed_count_matrix')
    input = np.asarray(input)
    tram.n_therm_states = len(input)
    tram.n_markov_states = np.max(np.concatenate(input)) + 1
    state_counts, transition_counts = tram._construct_count_models([[frag] for frag in input])
    assert np.array_equal(state_counts, [[10, 0], [9, 1], [4, 6], [3, 7], [1, 9]])
    assert np.array_equal(transition_counts,
                          [[[9, 0], [0, 0]], [[7, 1], [1, 0]], [[2, 2], [1, 4]], [[1, 1], [2, 5]], [[0, 1], [1, 7]]])


@pytest.mark.parametrize(
    "test_input,expected",
    [([[0, 0, 0, 1, 0, 0, 0], [1, 0, 1, 1, 1, 1]], [[(0, 0, 3), (0, 3, 7), (1, 0, 2)], [(1, 2, 6)]]),
     ([[0, 0, 0, 1, 1, 1, 0], [1, 0, 0, 1, 1, 1]], [[(0, 0, 3), (1, 0, 3)], [(0, 3, 6), (1, 3, 6)]]),
     ([[0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0]], [[(0, 0, 3), (0, 3, 7), (1, 0, 3), (1, 3, 6)], []])]
)
def test_trajectory_fragments_mapping(test_input, expected):
    tram = TRAM()
    tram.n_therm_states = np.max(np.concatenate(test_input)) + 1
    mapping = tram._find_trajectory_fragment_mapping([np.asarray(inp) for inp in test_input])
    assert mapping == expected


@pytest.mark.parametrize(
    "dtrajs, ttrajs, expected",
    [([[1, -1, 3, -1, 5, 6, 7], [8, 9, 10, 11, 12, 13, -1]],
      [[0, 0, 0, 1, 0, 0, 1], [0, 0, 1, 1, 0, 1, 1]],
      [[[1, 3], [5, 6], [8, 9]], [[10, 11], [12, 13]]])]
)
def test_get_trajectory_fragments(dtrajs, ttrajs, expected):
    tram = TRAM()
    tram.n_therm_states = 2
    mapping = tram._find_trajectory_fragments([np.asarray(dtraj) for dtraj in dtrajs],
                                              [np.asarray(ttraj) for ttraj in ttrajs])
    for k in range(tram.n_therm_states):
        assert len(mapping[k]) == len(expected[k])
        assert np.all([np.array_equal(mapping[k][i], expected[k][i]) for i in range(len(mapping[k]))])


@pytest.mark.parametrize(
    "dtrajs, expected",
    [([[1, 2, -1, -1, -1, 6, 7], [8, 9, 10, 11, 12, 13, 14]], [[1, 2, 6, 7], [8, 9, 10, 11, 12, 13, 14]]),
     ([[1, 2, 3, 4], [5, -1, -1, 8], [-1, -1, -1]], [[1, 2, 3, 4], [5, 8], []])]
)
@pytest.mark.parametrize(
    "ttrajs", [None, []]
)
def test_get_trajectory_fragments_no_ttrajs(dtrajs, ttrajs, expected):
    tram = TRAM()
    tram.n_therm_states = 2
    mapping = tram._find_trajectory_fragments([np.asarray(dtraj) for dtraj in dtrajs],
                                              ttrajs)
    for k in range(tram.n_therm_states):
        assert len(mapping[k][0]) == len(expected[k])
        assert np.all([np.array_equal(mapping[k][0][i], expected[k][i]) for i in range(len(mapping[k]))])
