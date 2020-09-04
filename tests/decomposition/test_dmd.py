import pytest
import numpy as np

from sktime.decomposition import DMD


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
        np.testing.assert_equal(DMD(mode).mode, mode)
    else:
        with np.testing.assert_raises(ValueError):
            DMD(mode)


@pytest.mark.parametrize("mode", DMD.available_modes)
def test_decomposition(toy_data, mode):
    A, x, y = toy_data
    model = DMD(mode).fit((x, y)).fetch_model()

    Atilde = model.modes @ np.diag(model.eigenvalues) @ model.modes.conj().T
    np.testing.assert_array_almost_equal(A, Atilde)


@pytest.mark.parametrize("mode", DMD.available_modes)
def test_propagate(toy_data, mode):
    A, x, y = toy_data
    model = DMD(mode).fit((x, y)).fetch_model()
    ytilde = model.transform(x[:50])
    np.testing.assert_array_almost_equal(ytilde, y[:50])
