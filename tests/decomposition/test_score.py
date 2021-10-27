import pytest
import numpy as np
from numpy.testing import assert_almost_equal

import deeptime as dt
from deeptime.decomposition import vamp_score_data


def nonlinearity(x):
    return np.exp(-x * x)


class ChiRnd:

    def __init__(self, n=15, scale=1, bias_var=5, out_dim=150):
        self.n_basis = n
        self.out_dim = out_dim
        self.W = np.random.normal(scale=scale, size=(2, self.n_basis))
        self.b = np.random.uniform(-bias_var, bias_var, size=(self.n_basis,))

        self.W2 = np.random.normal(scale=scale, size=(self.n_basis, out_dim))
        self.b2 = np.random.uniform(-bias_var, bias_var, size=(out_dim,))

    def __call__(self, x):
        return nonlinearity(x @ self.W + self.b) @ self.W2 + self.b2


def test_vamp_estimator_consistency(fixed_seed):
    chi = ChiRnd()
    _, traj = dt.data.sqrt_model(5000, seed=13)
    vamp = dt.decomposition.VAMP(1, observable_transform=chi, epsilon=1e-6).fit(traj).fetch_model()
    s2_ref = vamp.score(2)
    s2 = vamp_score_data(traj[:-1], traj[1:], vamp)
    assert_almost_equal(s2, s2_ref)

    s2_double = vamp_score_data(traj[:-1], traj[1:], transformation=vamp, epsilon=1e-6)
    assert_almost_equal(s2_ref, s2_double)

    s2_triple = vamp_score_data(traj[:-1], traj[1:], transformation=vamp.transform)
    assert_almost_equal(s2_ref, s2_triple)


def test_kcca_estimator_consistency(fixed_seed):
    _, traj = dt.data.sqrt_model(50, seed=13)

    f = dt.data.quadruple_well()
    X = np.random.uniform(-2, 2, size=(500, 2))
    Y = f(X)

    kcca = dt.decomposition.KernelCCA(dt.kernels.GaussianKernel(1.), 6, epsilon=1e-6) \
        .fit((X, Y), lagtime=1).fetch_model()
    vamp_w_kcca = vamp_score_data(X, Y, transformation=kcca, r=2, epsilon=1e-16)
    vamp_w_kcca_transform = vamp_score_data(X, Y, transformation=lambda x: kcca.transform(x).real, r=2, epsilon=1e-16)
    vamp = dt.decomposition.VAMP(1, observable_transform=kcca).fit((X, Y)).fetch_model()
    assert_almost_equal(vamp_w_kcca, vamp.score(2))
    assert_almost_equal(vamp_w_kcca, vamp_w_kcca_transform)
