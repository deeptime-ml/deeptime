import numpy as np
from numpy.testing import assert_almost_equal

import deeptime as dt


def test_linearly_evolved_data():
    data = np.random.uniform(-1, 1, size=(1000, 4))
    eigs = np.arange(4).astype(np.float64)
    eig_l = np.linalg.qr(np.random.normal(size=(4, 4)))[0]
    Kt = eig_l @ np.diag(eigs) @ eig_l.T
    est = dt.decomposition.EDMD(dt.basis.Identity())
    model = est.fit((data, data @ Kt)).fetch_model()

    assert_almost_equal(model.forward(data), data @ Kt)
    assert_almost_equal(model.forward(data @ Kt), data @ Kt @ Kt)

    np.testing.assert_array_almost_equal(Kt, model.operator)


def test_polynomially_evolved_data():
    data = np.random.uniform(0, 1, size=(50, 5)).astype(np.float64)
    data_t = 5. * data + 3 + data**2
    basis = dt.basis.Monomials(5, 5)
    psi_y = basis(data_t)

    est = dt.decomposition.EDMD(basis)
    model = est.fit((data, data_t)).fetch_model()
    assert_almost_equal(model.forward(data), psi_y, decimal=4)
