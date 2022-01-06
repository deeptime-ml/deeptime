import numpy as np
import pytest
from deeptime.markov.msm.tram import TRAM, unpack_input_tuple
from deeptime.markov.msm.tram._tram_bindings import tram as bindings
from deeptime.markov import TransitionCountEstimator, TransitionCountModel
from deeptime.markov.msm import MarkovStateModelCollection


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
        assert transition_counts[k].sum() == len(traj_fragments[k]) - lagtime
        assert state_counts[k].sum() == len(traj_fragments[k])


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


def test_unpack_input():
    arr = np.zeros(10)
    try:
        dtrajs, biases, ttrajs = unpack_input_tuple((arr, arr, arr))
        assert np.array_equal(dtrajs, arr) and np.array_equal(biases, arr) and np.array_equal(ttrajs, arr)
        dtrajs, biases, ttrajs = unpack_input_tuple((arr, arr))
        assert np.array_equal(dtrajs, arr) and np.array_equal(biases, arr) and len(ttrajs) == 0
    except IndexError:
        pytest.fail("IndexError while unpacking input!")


def test_unpack_input():
    arr = np.zeros(10)
    with pytest.raises(ValueError) as exinfo:
        unpack_input_tuple((arr, arr, arr, arr))
        assert 'Unexpected number of arguments' in exinfo.value


@pytest.mark.parametrize(
    "ttrajs", [np.asarray([[0, 0, 0], [1, 1, 1]]),
              [np.asarray(traj) for traj in [[0, 0, 0], [1, 1, 1]]],
               None]
)
@pytest.mark.parametrize(
    "dtrajs", [np.asarray([[0, 1, 0], [0, 1, 2]]),
               [np.asarray(traj) for traj in [[0, 1, 0], [0, 1, 2]]]]
)
@pytest.mark.parametrize(
    "bias_matrix_as_ndarray", [True, False]
)
def test_tram_different_input_data_types(dtrajs, ttrajs, bias_matrix_as_ndarray):
    bias_matrices = [np.random.rand(len(traj), 2) for traj in dtrajs]
    if bias_matrix_as_ndarray:
        bias_matrices = np.asarray(bias_matrices)

    tram = TRAM(maxiter=100)
    if ttrajs is None:
        tram.fit((dtrajs, bias_matrices))
    else:
        tram.fit((dtrajs, bias_matrices, ttrajs))


def test_lagtime_too_long():
    dtrajs = np.asarray([[0, 1, 0], [0, 1, 2, 1], [2, 3]])
    bias_matrices = [np.random.rand(len(traj), 3) for traj in dtrajs]
    tram = TRAM(maxiter=100, lagtime=2)
    tram.fit((dtrajs, bias_matrices))


def test_tram_fit():
    trajs = np.asarray([[0, 1, 1, 1, 1, 2, 2, 1, 0, 0], [1, 2, 3, 2, 2, 1, 0, 1, 2, 2], [2, 1, 2, 3, 2, 3, 3, 4, 3, 3],
                        [3, 2, 2, 3, 4, 4, 3, 4, 3, 2], [3, 2, 3, 3, 4, 4, 3, 4, 4, 3]])
    trajs = trajs / 5 * 3 - 1.5

    dtrajs = [np.asarray(i, dtype=np.int) for i in
              [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
               [1, 0, 0, 1, 1, 1, 1, 1, 1, 0], [1, 0, 1, 1, 1, 1, 1, 1, 1, 1]]]
    ttrajs = [np.ones((len(dtrajs[i])), dtype=np.int) * i for i in range(len(dtrajs))]

    bias_centers = [-1, -0.5, 0.0, 0.5, 1]

    def harmonic(x0, x):
        return 0.1 * (x - x0) ** 2

    # construct bias matric using harmonic potentials
    bias_matrices = [np.zeros((len(traj), len(bias_centers)), dtype=np.float64) for traj in dtrajs]
    for i, traj in enumerate(trajs):
        for j, bias_center in enumerate(bias_centers):
            bias = lambda x, x0=bias_center: harmonic(x0, x)
            bias_matrices[i][:, j] = bias(traj)

    therm_energies_1 = TRAM(maxiter=100).fit_fetch((dtrajs, bias_matrices)).therm_state_energies
    therm_energies_2 = TRAM(maxiter=100).fit_fetch((dtrajs, bias_matrices, ttrajs)).therm_state_energies
    assert (therm_energies_1 == therm_energies_2).all()

    # changing one ttrajs element should result in a change of the output
    ttrajs[0][2] = 1
    therm_energies_3 = TRAM(maxiter=100).fit_fetch((dtrajs, bias_matrices, ttrajs)).therm_state_energies
    assert (therm_energies_3 != therm_energies_1).any()

def test_tram_initialize_from_model():
    from .test_tram_model import random_model
    model = random_model(5, 3, transition_matrices=None)

    tram = TRAM(model=model)
    assert tram.n_markov_states == model.n_markov_states
    assert tram.n_therm_states == model.n_therm_states


def test_tram_integration():
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

    tram = TRAM(maxiter=100, connectivity='summed_count_matrix')
    assert tram.log_likelihood is None

    model = tram.fit_fetch((dtrajs, bias_matrices))

    # energies are identical. so are count matrices. and transition matrices
    assert np.allclose(model.therm_state_energies, [0.15673362, 0.077853, 0.04456354, 0.05706922, 0.11557514])
    assert np.allclose(model.markov_state_energies, [1.0550639, 0.42797176])

    MEMM = model.markov_state_model_collection
    assert np.allclose(MEMM.stationary_distribution, [1.])
    MEMM.select(1)
    assert np.allclose(MEMM.stationary_distribution, [0.3678024695571382, 0.6321975304428619])
    assert np.allclose(MEMM.transition_matrix,
                       [[0.7777777777777777, 0.22222222222222224], [0.12928535495314722, 0.8707146450468528]])
    MEMM.select(2)
    assert np.allclose(MEMM.transition_matrix, [[0.53558684, 0.46441316], [0.2403782, 0.7596218]])

    weights = model.compute_sample_weights(dtrajs, bias_matrices)
    assert np.allclose(np.sum(weights), 1)
    assert tram.log_likelihood < 0


def test_unknown_connectivity():
    with np.testing.assert_raises(ValueError):
        TRAM(connectivity='this_is_some_unknown_connectivity')


def to_numpy_arrays(dtrajs, bias_matrices, ttrajs):
    dtrajs = [np.asarray(traj) for traj in dtrajs]

    if ttrajs is not None:
        ttrajs = [np.asarray(traj) for traj in ttrajs]

    if not isinstance(bias_matrices, np.ndarray):
        bias_matrices = [np.asarray(M) for M in bias_matrices]

    return dtrajs, bias_matrices, ttrajs

@pytest.mark.parametrize(
    "dtrajs, bias_matrices, ttrajs",
    [
        ([[0, 0, 0], [0, 0, 0]], np.zeros((2, 3, 3)), None),
        ([[0, 0, 0], [0, 0]], np.zeros((2, 3, 2)), None),
        ([[0, 0, 0], [0, 0, 0]], np.zeros((2, 2, 2)), None),
        ([[0, 0, 0], [0, 0, 0]], np.zeros((3, 2, 2)), None),
        ([[0, 0, 0], [0, 0, 0]], np.zeros((1, 2, 2)), None),
        ([[0, 0, 0], [0, 0]], [[[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0, 0]]], None),
        ([[0, 0, 0], [0, 'x', 0]], np.zeros((2, 3, 3)), None),
        ([[0, 0, 0], [0, 0, 0]], np.zeros((2, 3, 2)), [[0, 0, 0], [0, 0, 0]]),
        ([[0, 0, 0], [0, 0, 0]], np.zeros((2, 3, 2)), [[0, 0, 0], [0, 1, 2]]),
        ([[0, 0, 0], [0, 0, 0]], np.zeros((2, 3, 2)), [[0, 0], [1, 1, 1]]),
        ([[0, 0, 0], [0, 0, 0]], np.zeros((2, 3, 2)), [[0, 0, 'x'], [1, 1, 1]]),
        ([[0, 0, 0], [0, 0, 0]], np.zeros((2, 3, 2)), [[0, 0, 0], [0, 1]])
    ]
)
def test_invalid_input(dtrajs, bias_matrices, ttrajs):
    dtrajs, bias_matrices, ttrajs = to_numpy_arrays(dtrajs, bias_matrices, ttrajs)
    tram = TRAM()

    with np.testing.assert_raises(ValueError):
        tram._validate_input(ttrajs, dtrajs, bias_matrices)


@pytest.mark.parametrize(
    "dtrajs, bias_matrices, ttrajs",
    [
        ([[0, 0, 0], [0, 0, 0]], np.zeros((2, 3, 3)), None),
        ([[0, 0, 0], [0, 0]], np.zeros((2, 3, 2)), None),
        ([[0, 0, 0], [0, 0, 0]], np.zeros((2, 2, 2)), None),
        ([[0, 0, 0], [0, 0, 0]], np.zeros((3, 2, 2)), None),
        ([[0, 0, 0], [0, 0, 0]], np.zeros((1, 2, 2)), None),
        ([[0, 0, 0], [0, 0]], [[[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0, 0]]], None),
        ([[0, 0, 0], [0, 'x', 0]], np.zeros((2, 3, 3)), None),
        ([[0, 0, 0], [0, 0, 0]], np.zeros((2, 3, 2)), [[0, 0, 0], [0, 1, 2]]),
        ([[0, 0, 0], [0, 0, 0]], np.zeros((2, 3, 2)), [[0, 0], [1, 1, 1]]),
        ([[0, 0, 0], [0, 0, 0]], np.zeros((2, 3, 2)), [[0, 0, 'x'], [1, 1, 1]]),
        ([[0, 0, 0], [0, 0, 0]], np.zeros((2, 3, 2)), [[0, 0, 0], [0, 1]]),
        ([[0, 1, 0], [0, 0, 0]], np.zeros((2, 3, 2)), [[0, 0, 0], [0, 0, 0]]),
        ([[0, 0, 0], [0, 1, 0]], np.zeros((2, 3, 2)), [[0, 2, 0], [0, 0, 0]]),
    ]
)
def test_invalid_input_initialized_from_model(dtrajs, bias_matrices, ttrajs):
    tram = TRAM()
    tram.n_therm_states = 2
    tram.n_markov_states = 1

    dtrajs, bias_matrices, ttrajs = to_numpy_arrays(dtrajs, bias_matrices, ttrajs)

    with np.testing.assert_raises(ValueError):
        tram._validate_input(ttrajs, dtrajs, bias_matrices)

@pytest.mark.parametrize(
    "dtrajs, bias_matrices, ttrajs",
    [
        ([[0, 0, 1], [0, 1, 0]], np.zeros((2, 3, 2)), [[0, 0, 0], [0, 0, 0]])
    ]
)
def test_valid_input_initialized_from_model(dtrajs, bias_matrices, ttrajs):
    tram = TRAM()
    tram.n_therm_states = 2
    tram.n_markov_states = 2

    dtrajs, bias_matrices, ttrajs = to_numpy_arrays(dtrajs, bias_matrices, ttrajs)

    tram._validate_input(ttrajs, dtrajs, bias_matrices)
