import numpy as np
import pytest
from scipy.integrate import odeint
from sklearn.preprocessing import PolynomialFeatures


from sktime.sindy import SINDy, SINDyModel, STLSQ


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
    return x, x_dot, t


@pytest.fixture
def data_inhomogeneous_ode():
    def ode(z, t):
        return [2 + (1 - z[0]) * z[1], 1 + z[0]]

    t = np.linspace(0, 5, 500)
    x0 = [0.5, 0.5]
    x = odeint(ode, x0, t)
    x_dot = np.array([ode(xi, 1) for xi in x])
    return x, x_dot, t


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

    model_coef = model.coef_

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

    expected_equations = ["1.0 1", "-1.0 x0 + 2.0 x1", "5.0 x0^2"]

    assert expected_equations == model.equations()


def test_predict(data_lorenz):
    x, _, _ = data_lorenz
    # Learn the identity map
    model = SINDy().fit(x, y=x).fetch_model()

    np.testing.assert_almost_equal(x, model.predict(x), decimal=4)


# Ensure the model can learn an inhomogeneous ODE
def test_inhomogeneous_ode(data_inhomogeneous_ode):
    x, x_dot, _ = data_inhomogeneous_ode

    # Prevent constant term from being included in library,
    # but allow optimizer to include bias term.
    est = SINDy(
        library=PolynomialFeatures(include_bias=False),
        optimizer=STLSQ(fit_intercept=True),
    )
    model = est.fit(x, y=x_dot).fetch_model()

    assert model.score(x, y=x_dot) > 0.9
