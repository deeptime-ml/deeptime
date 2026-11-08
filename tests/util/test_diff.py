import numpy as np
import pytest
from numpy.testing import assert_, assert_array_almost_equal

import deeptime.util.diff as diff


@pytest.mark.parametrize('sparse', [False, True])
def test_tv_derivative(sparse):
    # seeded: with unseeded noise the max error has a small but real chance of exceeding the threshold below
    rng = np.random.default_rng(42)
    noise_variance = .08 * .08
    x0 = np.linspace(0, 2.0 * np.pi, 400)
    testf = np.sin(x0) + rng.normal(0.0, np.sqrt(noise_variance), x0.shape)
    true_deriv = np.cos(x0)
    df = diff.tv_derivative(x0, testf, alpha=0.01, tol=1e-5, fd_window_radius=5, sparse=sparse)
    max_diff = np.max(np.abs(df - true_deriv))
    assert_(max_diff < 0.5)

    df2 = diff.tv_derivative(x0, testf, u0=df, alpha=0.01, tol=1e-5, fd_window_radius=5)
    assert_array_almost_equal(df, df2, decimal=1)


@pytest.mark.parametrize('sparse', [False, True])
def test_tv_derivative_convergence_criterion(sparse):
    # regression: the relative-change stopping criterion used norm(s[0]) (only the first
    # element of the update) instead of norm(s) (the whole update vector). With the buggy
    # criterion a tolerance-limited run stops almost immediately, far from the converged
    # solution. With a sensible tolerance the early-stopped result must be close to the
    # fully converged one.
    rng = np.random.default_rng(42)
    x0 = np.linspace(0, 2.0 * np.pi, 200)
    testf = np.sin(x0) + rng.normal(0.0, 0.05, x0.shape)
    converged = diff.tv_derivative(x0, testf, alpha=0.01, tol=None, maxit=500, sparse=sparse)
    early = diff.tv_derivative(x0, testf, alpha=0.01, tol=1e-4, maxit=500, sparse=sparse)
    assert_(np.max(np.abs(early - converged)) < 0.02)
