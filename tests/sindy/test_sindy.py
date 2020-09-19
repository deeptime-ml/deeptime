import numpy as np
import pytest
from scipy.integrate import odeint


from sktime.pysindy import SINDy, SINDyModel, STLSQ


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

    return x


def test_learned_coefficients(data_lorenz):
    x = data_lorenz
    x_dot
    
    est = SINDy()
    est.fit((x, None))

    model = est.fetch_model()
    # TODO