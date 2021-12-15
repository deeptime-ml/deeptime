import numpy as np
import pytest
from deeptime.markov.msm.tram import TRAM
from deeptime.markov import TransitionCountEstimator, TransitionCountModel
from deeptime.markov.msm import MarkovStateModelCollection


class tram_estimator_mock():
    def __init__(self, n_therm_states, n_markov_states):
        self.therm_state_energies = lambda: np.zeros(n_therm_states)
        self.biased_conf_energies = lambda: np.zeros((n_therm_states, n_markov_states))
        self.markov_state_energies = lambda: np.zeros(n_markov_states)
        transition_matrices = np.random.rand(n_therm_states, n_markov_states, n_markov_states)
        self.transition_matrices = lambda: transition_matrices / np.sum(transition_matrices, axis=-1, keepdims=True)


def get_connected_set_from_dtrajs_input(dtrajs, tram, has_ttrajs=True):
    tram.n_markov_states = np.max(np.concatenate(dtrajs)) + 1
    tram.n_therm_states = len(dtrajs)

    dtrajs = [np.asarray(traj) for traj in dtrajs]
    if has_ttrajs:
        ttrajs = np.asarray([[i] * len(traj) for i, traj in enumerate(dtrajs)])
    else:
        ttrajs = None

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
@pytest.mark.parametrize("has_ttrajs", [True, False])
def test_connected_set_summed_count_matrix(test_input, has_ttrajs, expected):
    tram = TRAM(lagtime=1, count_mode='sliding', connectivity='summed_count_matrix')
    cset = get_connected_set_from_dtrajs_input(test_input, tram, has_ttrajs)
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
    tram = TRAM(lagtime=1, count_mode='sliding', connectivity='post_hoc_RE')
    cset = get_connected_set_from_dtrajs_input(test_input, tram, has_ttrajs)
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
    tram = TRAM(lagtime=1, count_mode='sliding', connectivity='post_hoc_RE')
    tram.connectivity_factor = 0.0
    cset = get_connected_set_from_dtrajs_input(test_input, tram, has_ttrajs)
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
    tram = TRAM(lagtime=1, count_mode='sliding', connectivity='BAR_variance', connectivity_factor=1.0)
    cset = get_connected_set_from_dtrajs_input(test_input, tram, has_ttrajs)
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
    tram = TRAM(lagtime=1, count_mode='sliding', connectivity='BAR_variance')
    tram.connectivity_factor = 0.0
    cset = get_connected_set_from_dtrajs_input(test_input, tram, has_ttrajs)
    assert np.array_equal(cset.state_symbols, np.asarray(expected))


@pytest.mark.parametrize(
    "test_input,expected",
    [([[0, 1, 2, 3, 4, 5, 1], [2, 4, 2, 1, 3, 1, 4]], [[-1, 1, 2, 3, -1, -1, 1], [2, -1, 2, 1, 3, 1, -1]])]
)
def test_restrict_to_connected_set(test_input, expected):
    tram = TRAM()
    input = np.asarray(test_input)
    tram.n_therm_states = len(input)
    counts_model = TransitionCountEstimator(1, 'sliding').fit_fetch(input)
    tram._largest_connected_set = counts_model.submodel([1, 2, 3])
    output = tram._restrict_to_connected_set(input)
    assert np.array_equal(output, expected)


@pytest.mark.parametrize(
    "lagtime", [1, 3]
)
def test_make_count_models(lagtime):
    tram = TRAM(lagtime=lagtime)
    input = [np.asarray([1, 1, 2, 3, 1, 1, 1]), np.asarray([2, 0, 0, 1, 3, 1, 4]), np.asarray([2, 2, 2, 2])]
    tram.n_therm_states = len(input)
    tram.n_markov_states = np.max(np.concatenate(input)) + 1
    state_counts, transition_counts = tram._make_count_models(input)
    assert len(tram.count_models) == tram.n_therm_states
    assert state_counts.shape == (tram.n_therm_states, tram.n_markov_states)
    assert transition_counts.shape == (tram.n_therm_states, tram.n_markov_states, tram.n_markov_states)
    assert np.array_equal(tram.count_models[0].state_symbols, [0, 1, 2, 3])
    assert np.array_equal(tram.count_models[1].state_symbols, [0, 1, 2, 3, 4])
    assert np.array_equal(tram.count_models[2].state_symbols, [0, 1, 2])
    for k in range(tram.n_therm_states):
        assert transition_counts[k].sum() == len(input[k]) - lagtime
        assert state_counts[k].sum() == len(input[k])


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
    state_counts, transition_counts = tram._make_count_models(input)
    assert np.array_equal(state_counts, [[10, 0], [9, 1], [4, 6], [3, 7], [1, 9]])
    assert np.array_equal(transition_counts,
                          [[[9, 0], [0, 0]], [[7, 1], [1, 0]], [[2, 2], [1, 4]], [[1, 1], [2, 5]], [[0, 1], [1, 7]]])


def test_to_markov_model():
    tram = TRAM()
    tram.n_markov_states = 3
    tram.n_therm_states = 2
    tram._tram_estimator = tram_estimator_mock(tram.n_therm_states, tram.n_markov_states)
    transition_counts = np.zeros((2, 3, 3))
    tram.count_models = [TransitionCountModel(counts) for counts in transition_counts]
    tram._to_markov_model()
    model = tram.fetch_model()
    assert isinstance(model, MarkovStateModelCollection)
    assert (model.transition_matrix == tram._transition_matrices[0]).all()
    assert model.n_connected_msms == tram.n_therm_states
    model.select(1)
    assert (model.transition_matrix == tram._transition_matrices[1]).all()


@pytest.mark.parametrize(
    "test_input,expected",
    [([[0, 0, 0, 1, 0, 0, 0], [1, 0, 1, 1, 1, 1]], [[(0, 0, 3), (0, 3, 7), (1, 0, 2)], [(1, 2, 6)]]),
     ([[0, 0, 0, 1, 1, 1, 0], [1, 0, 0, 1, 1, 1]], [[(0, 0, 3), (1, 0, 3)], [(0, 3, 6), (1, 3, 6)]]),
     ([[0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0]], [[(0, 0, 3), (0, 3, 7), (1, 0, 3), (1, 3, 6)], []])]
)
def test_trajectory_fragments_mapping(test_input, expected):
    tram = TRAM()
    tram.n_therm_states = np.max(np.concatenate(test_input)) + 1
    mapping = tram._get_trajectory_fragment_mapping([np.asarray(inp) for inp in test_input])
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
    mapping = tram._get_trajectory_fragments([np.asarray(dtraj) for dtraj in dtrajs],
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
    mapping = tram._get_trajectory_fragments([np.asarray(dtraj) for dtraj in dtrajs],
                                             ttrajs)
    for k in range(tram.n_therm_states):
        assert len(mapping[k]) == len(expected[k])
        assert np.all([np.array_equal(mapping[k][i], expected[k][i]) for i in range(len(mapping[k]))])


def test_unpack_input():
    tram = TRAM()
    arr = np.zeros(10)
    try:
        dtrajs, biases, ttrajs = tram._unpack_input((arr, arr, arr))
        assert np.array_equal(dtrajs, arr) and np.array_equal(biases, arr) and np.array_equal(ttrajs, arr)
        dtrajs, biases, ttrajs = tram._unpack_input((arr, arr))
        assert np.array_equal(dtrajs, arr) and np.array_equal(biases, arr) and len(ttrajs) == 0
    except IndexError:
        pytest.fail("IndexError while unpacking input!")


def test_tram_fit_fetch():
    trajs = np.asarray([[0, 1, 1, 1, 1, 2, 2, 1, 0, 0], [1, 2, 3, 2, 2, 1, 0, 1, 2, 2], [2, 1, 2, 3, 2, 3, 3, 4, 3, 3],
                        [3, 2, 2, 3, 4, 4, 3, 4, 3, 2], [3, 2, 3, 3, 4, 4, 3, 4, 4, 3]])
    trajs = trajs / 5 * 3 - 1.5

    dtrajs = np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
                         [1, 0, 0, 1, 1, 1, 1, 1, 1, 0], [1, 0, 1, 1, 1, 1, 1, 1, 1, 1]])

    bias_centers = [-1, -0.5, 0.0, 0.5, 1]

    def harmonic(x0, x):
        return 0.1 * (x - x0) ** 2

    # construct bias matric using harmonic potentials
    bias_matrices = np.zeros((len(bias_centers), 10, len(bias_centers)))
    for i, traj in enumerate(trajs):
        for j, bias_center in enumerate(bias_centers):
            bias = lambda x, x0=bias_center: harmonic(x0, x)
            bias_matrices[i, :, j] = bias(traj)

    tram = TRAM(maxiter=100, connectivity='summed_count_matrix', save_convergence_info=True)
    tram.fit((dtrajs, bias_matrices))
    assert np.allclose(tram.therm_state_energies, [0.15673362, 0.077853, 0.04456354, 0.05706922, 0.11557514])
    assert np.allclose(tram.markov_state_energies, [1.0550639, 0.42797176])

    model = tram.fetch_model()
    assert np.allclose(model.stationary_distribution, [1.])
    model.select(1)
    assert np.allclose(tram.fetch_model().stationary_distribution, [0.3678024695571382, 0.6321975304428619])
    model.select(2)
    assert np.allclose(tram.fetch_model().transition_matrix, [[0.53558684, 0.46441316], [0.2403782,  0.7596218]])
