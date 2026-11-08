import numpy as np
import pytest
from numpy.testing import assert_, assert_array_almost_equal, assert_equal

import deeptime.util.diff as diff


@pytest.mark.parametrize('sparse', [False, True])
def test_tv_derivative(sparse):
    noise_variance = .08 * .08
    x0 = np.linspace(0, 2.0 * np.pi, 400)
    testf = np.sin(x0) + np.random.normal(0.0, np.sqrt(noise_variance), x0.shape)
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


@pytest.mark.parametrize('k, window_radius', [(1, 1), (1, 2), (2, 3), (3, 4)])
def test_finite_difference_operator_midpoints(k, window_radius):
    xs = np.cumsum(np.random.default_rng(7).uniform(0.01, 0.2, size=30))
    op = diff.finite_difference_operator_midpoints(xs, k=k, window_radius=window_radius)
    width = 2 * window_radius + 1
    expected = np.zeros((len(xs) - 1, len(xs)))
    for row in range(len(xs) - 1):
        start = min(max(row - window_radius, 0), len(xs) - width)
        window = slice(start, start + width)
        expected[row, window] = diff.finite_difference_coefficients(.5 * (xs[row] + xs[row + 1]), list(xs[window]),
                                                                    k=k)
    assert_equal(op.nnz, (len(xs) - 1) * width)
    assert_array_almost_equal(op.toarray(), expected)


def test_finite_difference_errors():
    with pytest.raises(ValueError):
        diff.finite_difference_coefficients(0., [0., 1.], k=2)
    with pytest.raises(ValueError):
        diff.finite_difference_coefficients(2., [0., 1., 1.5], k=1)
    with pytest.raises(ValueError):
        diff.finite_difference_operator_midpoints(np.linspace(0., 1., 4), window_radius=2)
    with pytest.raises(ValueError):
        diff.finite_difference_operator_midpoints(np.zeros(1), window_radius=0)
