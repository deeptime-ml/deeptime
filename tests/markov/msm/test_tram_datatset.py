import numpy as np
import pytest
from deeptime.markov.msm.tram import TRAMDataset
from deeptime.markov import TransitionCountEstimator, TransitionCountModel
from deeptime.markov.msm.tram._tram_bindings import tram as tram_bindings


def make_matching_bias_matrix(dtrajs, n_therm_states = None):
    if n_therm_states is None:
        n_therm_states = len(dtrajs)
    return [np.random.rand(len(traj), n_therm_states) for traj in dtrajs]


def make_random_input_data(n_therm_states, n_markov_states, n_samples=10, make_ttrajs=True):
    dtrajs = [np.random.randint(0, n_markov_states, size=n_samples) for _ in range(n_therm_states)]
    bias_matrices = make_matching_bias_matrix(dtrajs, n_therm_states)

    if make_ttrajs:
        ttrajs = [np.random.randint(0, n_therm_states, size=n_samples) for _ in range(n_therm_states)]
        return dtrajs, bias_matrices, ttrajs

    return dtrajs, bias_matrices


def get_connected_set_from_dtrajs_input(dtrajs, connectivity, has_ttrajs=True, connectivity_factor=1):
    dtrajs = [np.asarray(traj) for traj in dtrajs]
    if has_ttrajs:
        ttrajs = [np.asarray([i] * len(traj)) for i, traj in enumerate(dtrajs)]
    else:
        ttrajs = None

    bias_matrices = [np.ones((len(traj), len(dtrajs))) for traj in dtrajs]
    tramdata = TRAMDataset(dtrajs=dtrajs, ttrajs=ttrajs, bias_matrices=bias_matrices, lagtime=1, count_mode='sliding')

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
    np.testing.assert_equal(cset.state_symbols, np.asarray(expected))


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
    np.testing.assert_equal(cset.state_symbols, np.asarray(expected))


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
    np.testing.assert_equal(cset.state_symbols, np.asarray(expected))


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
    np.testing.assert_equal(cset.state_symbols, np.asarray(expected))


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
    np.testing.assert_equal(cset.state_symbols, np.asarray(expected))


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
    np.testing.assert_equal(tramdata.dtrajs, expected)


@pytest.mark.parametrize(
    "lagtime", [1, 3]
)
def test_make_count_models(lagtime):
    dtrajs = [np.asarray([1,1,2,3,1,1,1,2,0,0,1,3,1,4,2,2,2,2])]
    ttrajs = [np.asarray([0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2])]
    bias_matrices = make_matching_bias_matrix(dtrajs, 3)

    dataset = TRAMDataset(dtrajs=dtrajs, ttrajs=ttrajs, bias_matrices = bias_matrices, lagtime=lagtime)
    dataset._compute_counts()

    np.testing.assert_equal(len(dataset.count_models), dataset.n_therm_states)
    np.testing.assert_equal(dataset.state_counts.shape, (dataset.n_therm_states, dataset.n_markov_states))
    np.testing.assert_equal(dataset.transition_counts.shape, (dataset.n_therm_states, dataset.n_markov_states, dataset.n_markov_states))
    np.testing.assert_equal(dataset.count_models[0].state_symbols, [0, 1, 2, 3])
    np.testing.assert_equal(dataset.count_models[1].state_symbols, [0, 1, 2, 3, 4])
    np.testing.assert_equal(dataset.count_models[2].state_symbols, [0, 1, 2])
    for k in range(dataset.n_therm_states):
        np.testing.assert_equal(dataset.transition_counts[k].sum(), len(dataset._find_trajectory_fragments()[k][0]) - lagtime)
        np.testing.assert_equal(dataset.state_counts[k].sum(), len(dataset._find_trajectory_fragments()[k][0]))


@pytest.mark.parametrize(
    "input", [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
               [1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
               [1, 0, 1, 1, 1, 1, 1, 1, 1, 1]]]
)
def test_transposed_count_matrices_bug(input):
    dtrajs = [np.asarray(traj) for traj in input]
    bias_matrices = make_matching_bias_matrix(dtrajs)
    dataset = TRAMDataset(dtrajs=dtrajs, bias_matrices=bias_matrices)
    dataset.restrict_to_largest_connected_set(connectivity='summed_count_matrix')
    dataset._compute_counts()
    np.testing.assert_equal(dataset.state_counts, [[10, 0], [9, 1], [4, 6], [3, 7], [1, 9]])
    np.testing.assert_equal(dataset.transition_counts,
                          [[[9, 0], [0, 0]], [[7, 1], [1, 0]], [[2, 2], [1, 4]], [[1, 1], [2, 5]], [[0, 1], [1, 7]]])


@pytest.mark.parametrize(
    "test_input,expected",
    [([[0, 0, 0, 1, 0, 0, 0], [1, 0, 1, 1, 1, 1]], [[(0, 0, 3), (0, 3, 7), (1, 0, 2)], [(1, 2, 6)]]),
     ([[0, 0, 0, 1, 1, 1, 0], [1, 0, 0, 1, 1, 1]], [[(0, 0, 3), (1, 0, 3)], [(0, 3, 6), (1, 3, 6)]]),
     ([[0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0]], [[(0, 0, 3), (0, 3, 7), (1, 0, 3), (1, 3, 6)], []])]
)
def test_trajectory_fragments_mapping(test_input, expected):
    n_therm_states = np.max(np.concatenate(test_input)) + 1
    mapping = tram_bindings.find_trajectory_fragment_indices([np.asarray(inp) for inp in test_input], n_therm_states)
    np.testing.assert_equal(mapping, expected)


def test_trajectory_fragments_mapping_no_ttrajs():
    dtrajs, bias_matrices = make_random_input_data(5, 8, make_ttrajs=False)
    dataset = TRAMDataset(dtrajs=dtrajs, bias_matrices=bias_matrices)

    np.testing.assert_equal([[traj] for traj in dtrajs], dataset._find_trajectory_fragments())


@pytest.mark.parametrize(
    "dtrajs, ttrajs, expected",
    [([[1, -1, 3, -1, 5, 6, 7], [8, 9, 10, 11, 12, 13, -1]],
      [[0, 0, 0, 1, 0, 0, 1], [0, 0, 1, 1, 0, 1, 1]],
      [[[1, 3], [5, 6], [8, 9]], [[10, 11], [12, 13]]])]
)
def test_get_trajectory_fragments(dtrajs, ttrajs, expected):
    dtrajs=[np.asarray(d) for d in dtrajs]
    ttrajs=[np.asarray(t) for t in ttrajs]
    bias_matrices = make_matching_bias_matrix(dtrajs)
    dataset = TRAMDataset(dtrajs=dtrajs, ttrajs=ttrajs, bias_matrices=bias_matrices)

    mapping = dataset._find_trajectory_fragments()
    for k in range(dataset.n_therm_states):
        np.testing.assert_equal(len(mapping[k]), len(expected[k]))
        [np.testing.assert_equal(mapping[k][i], expected[k][i]) for i in range(len(mapping[k]))]


def test_unknown_connectivity():
    dtrajs, bias_matrices, ttrajs = make_random_input_data(2, 2)
    dataset = TRAMDataset(dtrajs, bias_matrices, ttrajs)
    with np.testing.assert_raises(ValueError):
        dataset.restrict_to_largest_connected_set(connectivity='this_is_some_unknown_connectivity')
