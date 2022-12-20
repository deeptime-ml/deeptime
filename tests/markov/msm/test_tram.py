import numpy as np
import pytest
import warnings
from tqdm import tqdm
from sklearn.exceptions import ConvergenceWarning

from deeptime.markov.msm import TRAM, TRAMDataset
from tests.testing_utilities import ProgressMock

from .test_tram_model import make_random_model


def make_random_input_data(n_therm_states, n_markov_states, n_samples=10, make_ttrajs=True):
    dtrajs = [np.random.randint(0, n_markov_states, size=n_samples) for _ in range(n_therm_states)]
    bias_matrices = [np.random.rand(n_samples, n_therm_states) for _ in range(n_therm_states)]

    if make_ttrajs:
        ttrajs = [np.random.randint(0, n_therm_states, size=n_samples) for _ in range(n_therm_states)]
        return dtrajs, bias_matrices, ttrajs

    return dtrajs, bias_matrices


def test_unpack_input():
    arr = np.zeros(10)
    with pytest.raises(ValueError) as exinfo:
        TRAM().fit((arr, arr, arr, arr))
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
@pytest.mark.parametrize(
    "init_strategy", ["MBAR", None]
)
def test_tram_different_input_data_types(dtrajs, ttrajs, bias_matrix_as_ndarray, init_strategy):
    bias_matrices = [np.random.rand(len(traj), 2) for traj in dtrajs]
    if bias_matrix_as_ndarray:
        bias_matrices = np.asarray(bias_matrices)

    tram = TRAM(maxiter=100, init_strategy=init_strategy)
    if ttrajs is None:
        tram.fit((dtrajs, bias_matrices))
    else:
        tram.fit((dtrajs, bias_matrices, ttrajs))


def test_lagtime_too_long():
    dtrajs = np.asarray([[0, 1, 0], [0, 1, 2, 1], [2, 3]], dtype=object)
    bias_matrices = [np.random.rand(len(traj), 3) for traj in dtrajs]
    tram = TRAM(maxiter=100, lagtime=2)
    with np.testing.assert_raises(ValueError):
        tram.fit((dtrajs, bias_matrices))


def test_fit_empty_markov_state():
    dtrajs = [np.asarray(arr) for arr in [[0, 1, 0], [0, 1, 0, 3], [3, 3]]]
    bias_matrices = [np.random.rand(len(traj), 3) for traj in dtrajs]
    TRAM().fit((dtrajs, bias_matrices))


def test_tram_fit():
    dtrajs, bias_matrices = make_random_input_data(5, 10, make_ttrajs=False)

    ttrajs = [np.ones((len(dtrajs[i])), dtype=int) * i for i in range(len(dtrajs))]

    therm_energies_1 = TRAM(maxiter=100).fit_fetch((dtrajs, bias_matrices)).therm_state_energies
    therm_energies_2 = TRAM(maxiter=100).fit_fetch((dtrajs, bias_matrices, ttrajs)).therm_state_energies
    np.testing.assert_equal(therm_energies_1, therm_energies_2)

    # changing one ttrajs element should result in a change of the output
    ttrajs[0][2] += 1
    therm_energies_3 = TRAM(maxiter=100).fit_fetch((dtrajs, bias_matrices, ttrajs)).therm_state_energies

    with np.testing.assert_raises(AssertionError):
        np.testing.assert_equal(therm_energies_3, therm_energies_1)


def test_tram_continue_estimation():
    from .test_tram_model import make_random_model
    model = make_random_model(5, 8, transition_matrices=None)
    dtrajs, bias_matrices = make_random_input_data(5, 8, make_ttrajs=False)

    weights_1 = model.compute_sample_weights_log(dtrajs, bias_matrices)

    tram = TRAM()
    model = tram.fit_fetch((dtrajs, bias_matrices), model=model)
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(weights_1, model.compute_sample_weights_log(dtrajs, bias_matrices))


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
            def bias(x, x0=bias_center):
                return harmonic(x0, x)

            bias_matrices[i, :, j] = bias(traj)

    tram = TRAM(maxiter=100)
    model = tram.fit_fetch((dtrajs, bias_matrices))

    # energies are identical. so are count matrices. and transition matrices
    np.testing.assert_almost_equal(model.therm_state_energies,
                                   [0.15673362, 0.077853, 0.04456354, 0.05706922, 0.11557514])
    np.testing.assert_almost_equal(model.markov_state_energies, [1.0550639, 0.42797176])

    memm = model.msm_collection
    np.testing.assert_almost_equal(memm.stationary_distribution, [1.])
    memm.select(1)
    np.testing.assert_almost_equal(memm.stationary_distribution, [0.3678024695571382, 0.6321975304428619])
    np.testing.assert_almost_equal(memm.transition_matrix,
                                   [[0.7777777777777777, 0.22222222222222224],
                                    [0.12928535495314722, 0.8707146450468528]])
    memm.select(2)
    np.testing.assert_almost_equal(memm.transition_matrix, [[0.53558684, 0.46441316], [0.2403782, 0.7596218]])

    weights = model.compute_sample_weights_log(dtrajs, bias_matrices)
    np.testing.assert_almost_equal(np.sum(np.exp(weights)), 1)


def to_numpy_arrays(dtrajs, bias_matrices, ttrajs):
    dtrajs = [np.asarray(traj) for traj in dtrajs]

    if ttrajs is not None:
        ttrajs = [np.asarray(traj) for traj in ttrajs]

    if not isinstance(bias_matrices, np.ndarray):
        bias_matrices = [np.asarray(M, dtype=object) for M in bias_matrices]

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
        tram.fit((ttrajs, dtrajs, bias_matrices))


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
def test_invalid_input_with_model(dtrajs, bias_matrices, ttrajs):
    tram = TRAM()
    model = make_random_model(2, 1)

    dtrajs, bias_matrices, ttrajs = to_numpy_arrays(dtrajs, bias_matrices, ttrajs)

    with np.testing.assert_raises(ValueError):
        if ttrajs is None:
            tram.fit((dtrajs, bias_matrices), model)
        else:
            tram.fit((dtrajs, bias_matrices, ttrajs), model)


@pytest.mark.parametrize(
    "dtrajs, bias_matrices, ttrajs",
    [
        ([[0, 0, 1], [0, 1, 0]], np.zeros((2, 3, 2)), [[0, 0, 0], [1, 1, 1]])
    ]
)
def test_valid_input_with_model(dtrajs, bias_matrices, ttrajs):
    tram = TRAM()
    model = make_random_model(2, 2)
    dtrajs, bias_matrices, ttrajs = to_numpy_arrays(dtrajs, bias_matrices, ttrajs)
    tram.fit((dtrajs, bias_matrices, ttrajs), model)


@pytest.mark.parametrize(
    "track_log_likelihoods", [True, False]
)
def test_callback_called(track_log_likelihoods):
    tram = TRAM(track_log_likelihoods=track_log_likelihoods, callback_interval=2, maxiter=10)
    tram_input = make_random_input_data(5, 5)
    tram.fit(tram_input)
    np.testing.assert_equal(len(tram.log_likelihoods), 5)
    np.testing.assert_equal(len(tram.energy_increments), 5)
    np.testing.assert_(np.min(tram.energy_increments) > 0)
    if track_log_likelihoods:
        np.testing.assert_((np.asarray(tram.log_likelihoods) > -np.inf).all())
    else:
        np.testing.assert_((np.asarray(tram.log_likelihoods) == -np.inf).all())


def test_progress_bar_update_called():
    progress = ProgressMock()

    class ProgressFactory:
        def __new__(cls, *args, **kwargs): return progress

    tram = TRAM(callback_interval=2, maxiter=10, progress=ProgressFactory, init_strategy=None)
    tram.fit(make_random_input_data(5, 5))

    # update() should be called 5 times
    np.testing.assert_equal(progress.n_update_calls, 5)
    # description should be set once initially and on each update call
    np.testing.assert_equal(progress.n_description_updates, progress.n_update_calls + 1)
    np.testing.assert_equal(progress.n, 10)
    # and close() one time
    np.testing.assert_equal(progress.n_close_calls, 1)


def test_progress_bar_update_called_with_mbar():
    progress = ProgressMock()

    class ProgressFactory:
        def __new__(cls, *args, **kwargs): return progress

    tram = TRAM(callback_interval=2, maxiter=10, progress=ProgressFactory, init_maxiter=10, init_maxerr=1e-15)
    tram.fit(make_random_input_data(5, 5))

    # update() should be called 10 times
    np.testing.assert_equal(progress.n_update_calls, 10)
    # description should be set once initially for both MBAR and TRAM (=2) plus once on each update call
    np.testing.assert_equal(progress.n_description_updates, progress.n_update_calls + 2)
    np.testing.assert_equal(progress.n, 20)
    # and close() one time
    np.testing.assert_equal(progress.n_close_calls, 2)


def test_tqdm_progress_bar():
    tram = TRAM(callback_interval=2, maxiter=10, progress=tqdm)
    tram.fit(make_random_input_data(5, 5))


@pytest.mark.parametrize(
    "init_strategy", ["MBAR", None]
)
def test_fit_with_dataset(init_strategy):
    dataset = TRAMDataset(dtrajs=[np.asarray([0, 1, 2])], bias_matrices=[np.asarray([[1.], [2.], [3.]])])
    tram = TRAM(init_strategy=init_strategy)
    tram.fit(dataset)


@pytest.mark.parametrize(
    "init_strategy", ["MBAR", None]
)
def test_fit_with_dataset(init_strategy):
    input_data = make_random_input_data(20, 2)
    tram = TRAM(init_strategy=init_strategy)
    tram.fit(input_data)


def test_mbar_initalization():
    (dtrajs, bias_matrices) = make_random_input_data(5, 5, make_ttrajs=False)
    tram = TRAM(callback_interval=2, maxiter=0, progress=tqdm, init_maxiter=100)
    ll1 = tram.fit_fetch((dtrajs, bias_matrices)).compute_log_likelihood(dtrajs, bias_matrices)

    tram = TRAM(callback_interval=2, maxiter=0, progress=tqdm, init_maxiter=0)
    ll2 = tram.fit_fetch((dtrajs, bias_matrices)).compute_log_likelihood(dtrajs, bias_matrices)

    np.testing.assert_(ll1 > ll2)


def test_unknown_init_strategy():
    tram = TRAM(init_strategy="this_is_not_an_init_strategy")
    with np.testing.assert_raises(ValueError):
        tram.fit(make_random_input_data(3, 3))


def test_mbar_initialization_zero_iterations():
    tram1 = TRAM(init_strategy="MBAR", init_maxiter=0, maxiter=3)
    tram2 = TRAM(init_strategy=None, init_maxiter=0, maxiter=3)

    input_data = make_random_input_data(5, 5)
    model1 = tram1.fit_fetch(input_data)
    model2 = tram2.fit_fetch(input_data)
    np.testing.assert_equal(model1.biased_conf_energies, model2.biased_conf_energies)


def test_converged_before_callback_called_does_not_produce_warning():
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.filterwarnings("error")
        np.random.seed(1) 
        input_data = make_random_input_data(5, 5)

        tram = TRAM(callback_interval=50, maxerr=0.1, maxiter=50)
        try:
            tram.fit(input_data)
        except ConvergenceWarning:
            assert False


def test_fit_with_dataset_uses_correct_lagtime():
    tram = TRAM()
    dtrajs, biases = make_random_input_data(5, 5, n_samples=5, make_ttrajs=False)
    with np.testing.assert_raises(ValueError):
        dataset = TRAMDataset(dtrajs, biases, lagtime=100)


def test_fit_without_dataset_uses_correct_lagtime():
    tram = TRAM(lagtime=100)
    input = make_random_input_data(5, 5, n_samples=5, make_ttrajs=False)
    with np.testing.assert_raises(ValueError):
        tram.fit(input)

