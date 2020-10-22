import pytest
import numpy as np
from numpy.testing import assert_raises, assert_array_almost_equal, assert_equal

from deeptime.decomposition import DMD


@pytest.fixture
def toy_data():
    A = np.linalg.qr(np.random.normal(size=(25, 25)))[0]
    x0 = np.random.uniform(-1, 1, size=(A.shape[0],))

    xs = np.empty((500, A.shape[0]))
    xs[0] = x0
    for t in range(1, len(xs)):
        xs[t] = A @ xs[t - 1]

    x = xs[:-1]
    y = xs[1:]
    return A, x, y


@pytest.mark.parametrize("mode", list(DMD.available_modes) + ["....."])
def test_mode(mode):
    if mode in DMD.available_modes:
        assert_equal(DMD(mode).mode, mode)
    else:
        with assert_raises(ValueError):
            DMD(mode)


@pytest.mark.parametrize("rank", [None, 13, 15, 24])
@pytest.mark.parametrize("mode", DMD.available_modes)
@pytest.mark.parametrize("driver", ['scipy', 'numpy'])
def test_rank_cutoff(toy_data, rank, mode, driver):
    A, x, y = toy_data
    model = DMD(mode=mode, driver=driver, rank=rank).fit((x, y)).fetch_model()
    if rank is None:
        expected_rank = A.shape[0]
    else:
        expected_rank = rank
    assert_equal(model.eigenvalues.shape[0], expected_rank)
    assert_equal(model.modes.shape[0], expected_rank)
    assert_equal(model.modes.shape[1], A.shape[0])


@pytest.mark.parametrize("mode", DMD.available_modes + ('bogus',))
@pytest.mark.parametrize("driver", ['scipy', 'numpy', 'bogus'])
def test_decomposition(toy_data, mode, driver):
    A, x, y = toy_data
    if driver == 'bogus' or mode == 'bogus':
        with assert_raises(ValueError):
            DMD(mode, driver=driver)
    else:
        model = DMD(mode, driver=driver).fit((x, y)).fetch_model()

        Atilde = model.modes.T @ np.diag(model.eigenvalues) @ model.modes.conj()
        assert_array_almost_equal(A, Atilde)


@pytest.mark.parametrize("mode", DMD.available_modes)
def test_propagate(toy_data, mode):
    A, x, y = toy_data
    model = DMD(mode).fit((x, y)).fetch_model()
    ytilde = model.transform(x[:50])
    assert_array_almost_equal(ytilde, y[:50])
