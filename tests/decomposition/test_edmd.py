import numpy as np
import pytest
from numpy.testing import assert_almost_equal

import deeptime as dt


@pytest.mark.parametrize("factor", [.1, .5, 10.])
def test_constant_data(factor):
    data = np.ones((1000, 5))
    data_t = np.copy(data) * factor
    est = dt.decomposition.EDMD(dt.basis.Identity())
    model = est.fit((data, data_t)).fetch_model()
    assert_almost_equal(model.eigenvalues[0], factor)
    assert_almost_equal(model.eigenvalues[1:], 0.)

    Y = model.transform(data)[:, 0]  # should be constant in 1st component, rest is noise
    assert_almost_equal(Y, Y[0])

    Y = model.transform(data, forward=False)[:, 0]  # should be constant in 1st component, rest is noise
    assert_almost_equal(Y, Y[0])

    Y = model.transform(data, propagate=True)[:, 0]  # should be constant in 1st component, rest is noise
    assert_almost_equal(Y, Y[0])

    Y = model.transform(data, propagate=True, forward=False)[:, 0]  # should be constant in 1st component, rest is noise
    assert_almost_equal(Y, Y[0])

    assert_almost_equal(model.forward(data), data_t)
    assert_almost_equal(model.forward(data_t), data_t * factor)
