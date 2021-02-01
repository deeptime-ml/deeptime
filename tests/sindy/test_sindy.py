import numpy as np
import pytest
from numpy.testing import assert_equal, assert_
from scipy.integrate import odeint
from sklearn.preprocessing import PolynomialFeatures

import deeptime as dt
from deeptime.sindy import SINDy, STLSQ


@pytest.fixture
def data_lorenz():
    def lorenz(z, t):
        return [
            10 * (z[1] - z[0]),
            z[0] * (28 - z[2]) - z[1],
            z[0] * z[1] - 8 / 3 * z[2],
        ]

    t = np.linspace(0, 5, 500)
    x0 = [8, 27, -7]
    x = odeint(lorenz, x0, t)
    x_dot = np.array([lorenz(xi, 1) for xi in x])
    return np.array(x), np.array(x_dot), t


@pytest.fixture
def data_inhomogeneous_ode():
    def ode(z, t):
        return [2 + (1 - z[0]) * z[1], 1 + z[0]]

    t = np.linspace(0, 5, 500)
    x0 = [0.5, 0.5]
    x = odeint(ode, x0, t)
    x_dot = np.array([ode(xi, 1) for xi in x])
    return np.array(x), x_dot, t


def test_learned_coefficients(data_lorenz):
    x, x_dot, _ = data_lorenz
    model = SINDy().fit(x, y=x_dot).fetch_model()

    true_coef = np.zeros((3, 10))
    true_coef[0, 1] = -10
    true_coef[0, 2] = 10
    true_coef[1, 1] = 28
    true_coef[1, 2] = -1
    true_coef[1, 6] = -1
    true_coef[2, 3] = -8 / 3
    true_coef[2, 5] = 1

    model_coef = model.coefficients

    np.testing.assert_almost_equal(true_coef, model_coef, decimal=3)


def test_score(data_lorenz):
    x, x_dot, t = data_lorenz
    model = SINDy().fit(x, y=x_dot).fetch_model()

    assert model.score(x, y=x_dot) > 0.9

    # score method should approximate derivative itself
    assert model.score(x, t=t) > 0.9


def test_equations(data_lorenz):
    x, x_dot, _ = data_lorenz
    model = SINDy().fit(x, y=x_dot).fetch_model()
    new_coef = np.zeros_like(model.coef_)

    new_coef[0, 0] = 1
    new_coef[1, 1] = -1
    new_coef[1, 2] = 2
    new_coef[2, 4] = 5
    model.coef_ = new_coef

    expected_equations = ["1.000 1", "-1.000 x0 + 2.000 x1", "5.000 x0^2"]

    assert_equal(model.equations(precision=3), expected_equations)


def test_predict(data_lorenz):
    x, _, _ = data_lorenz
    # Learn the identity map
    model = SINDy().fit(x, y=x).fetch_model()

    np.testing.assert_almost_equal(x, model.predict(x), decimal=4)


def test_simulate(data_lorenz):
    x, x_dot, t = data_lorenz
    model = SINDy().fit(x, y=x_dot).fetch_model()

    x_sim = model.simulate(x[0], t)

    assert x_sim.shape == x.shape


# Ensure the model can learn an inhomogeneous ODE
@pytest.mark.parametrize("library", [PolynomialFeatures(include_bias=False), dt.basis.Monomials(2, 2)])
def test_inhomogeneous_ode(data_inhomogeneous_ode, library):
    x, x_dot, _ = data_inhomogeneous_ode

    # Prevent constant term from being included in library,
    # but allow optimizer to include bias term.
    est = SINDy(
        library=library,
        optimizer=STLSQ(fit_intercept=True),
    )
    model = est.fit(x, y=x_dot).fetch_model()
    model.transform(x)

    assert model.score(x, y=x_dot) > 0.9


@pytest.mark.parametrize("library", [PolynomialFeatures(degree=2), dt.basis.Monomials(2, 3)])
@pytest.mark.parametrize("lhs", [None, ['dx0', 'dx1', 'dx2']])
@pytest.mark.parametrize("precision", [1, 2, 3, 4, 5])
def test_print(capsys, data_lorenz, library, lhs, precision):
    x, x_dot, _ = data_lorenz
    model = SINDy(library=library).fit(x, y=x_dot).fetch_model()

    true_coef = np.zeros((3, 10))
    true_coef[0, 1] = -10
    true_coef[0, 2] = 10
    true_coef[1, 1] = 28
    true_coef[1, 2] = -1
    true_coef[1, 6] = -1
    true_coef[2, 3] = -8 / 3
    true_coef[2, 5] = 1

    model_coef = model.coefficients

    lhs_ref = ["x0", "x1", "x2"] if lhs is None else lhs

    np.testing.assert_almost_equal(true_coef, model_coef, decimal=3)
    capsys.readouterr()  # ignore, this resets the buffers
    model.print(lhs=lhs, precision=precision)
    captured = capsys.readouterr()
    printed = captured.out
    lines = [line for line in printed.split('\n') if len(line) > 0]
    assert_equal(len(lines), 3)
    for i in range(3):
        assert_(f"{lhs_ref[i]}'" in lines[i])
    assert_(f"{model_coef[0, 1]:.{precision}f} x0" in lines[0])
    assert_(f"{model_coef[0, 2]:.{precision}f} x1" in lines[0])
    assert_(f"{model_coef[1, 1]:.{precision}f} x0" in lines[1])
    assert_(f"{model_coef[1, 2]:.{precision}f} x1" in lines[1])
    assert_(f"{model_coef[1, 6]:.{precision}f} x0 x2" in lines[1])
    assert_(f"{model_coef[2, 3]:.{precision}f} x2" in lines[2])
    assert_(f"{model_coef[2, 5]:.{precision}f} x0 x1" in lines[2])
