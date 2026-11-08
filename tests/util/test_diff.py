import math

import numpy as np
import pytest
from numpy.testing import assert_, assert_array_almost_equal, assert_allclose, assert_array_equal

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


def _nonuniform_grid(n, seed=7):
    return np.cumsum(np.random.default_rng(seed).uniform(0.05, 0.2, size=n))


@pytest.mark.parametrize('n', [2, 3, 5, 8])
@pytest.mark.parametrize('k', [0, 1, 2, 3])
def test_finite_difference_coefficients_exact_on_polynomials(n, k):
    if k >= n:
        pytest.skip('needs more nodes than derivative order')
    xs = _nonuniform_grid(n)
    x_bar = np.random.default_rng(n * 10 + k).uniform(xs.min(), xs.max())
    w = diff.finite_difference_coefficients(x_bar, xs, k=k)
    assert w.shape == (n,)
    # n nodes determine the degree-(n-1) interpolant, so all lower degrees must differentiate exactly
    for p in range(n):
        expected = math.perm(p, k) * x_bar ** (p - k) if p >= k else 0.
        assert_allclose(w @ xs ** p, expected, rtol=1e-9, atol=1e-9)


def test_finite_difference_coefficients_known_weights():
    # textbook four-point stencil for f' at the midpoint of a uniform grid with spacing h
    h = .25
    xs = h * np.arange(4)
    w = diff.finite_difference_coefficients(1.5 * h, xs, k=1)
    assert_allclose(w, np.array([1., -27., 27., -1.]) / (24. * h))


def test_finite_difference_coefficients_batched_matches_single():
    rng = np.random.default_rng(3)
    xs = np.sort(rng.uniform(-2., 2., size=(6, 5)), axis=0)
    x_bar = rng.uniform(xs.min(axis=0), xs.max(axis=0))
    batched = diff.finite_difference_coefficients(x_bar, xs, k=2)
    for col in range(xs.shape[1]):
        assert_allclose(batched[:, col], diff.finite_difference_coefficients(x_bar[col], xs[:, col], k=2))


@pytest.mark.parametrize('k, window_radius', [(1, 1), (1, 2), (2, 2), (2, 3), (3, 4)])
def test_finite_difference_operator_midpoints_exact_on_polynomials(k, window_radius):
    xs = _nonuniform_grid(40)
    op = diff.finite_difference_operator_midpoints(xs, k=k, window_radius=window_radius)
    midpoints = .5 * (xs[:-1] + xs[1:])
    for p in range(2 * window_radius + 1):
        expected = math.perm(p, k) * midpoints ** (p - k) if p >= k else np.zeros_like(midpoints)
        scale = max(np.abs(expected).max(), 1.)
        assert_allclose(op @ xs ** p, expected, rtol=1e-8, atol=1e-8 * scale)


@pytest.mark.parametrize('window_radius', [1, 2, 3])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_finite_difference_operator_midpoints_structure(window_radius, dtype):
    n = 12
    width = 2 * window_radius + 1
    xs = np.linspace(0., 1., n).astype(dtype)
    op = diff.finite_difference_operator_midpoints(xs, k=0, window_radius=window_radius)
    assert op.format == 'csc'
    assert op.shape == (n - 1, n)
    assert op.dtype == dtype
    assert op.nnz == (n - 1) * width
    op = op.tocsr()
    op.sort_indices()
    for row in range(n - 1):
        start = min(max(row - window_radius, 0), n - width)
        assert_array_equal(op.indices[op.indptr[row]:op.indptr[row + 1]], np.arange(start, start + width))


def test_finite_difference_errors():
    with pytest.raises(ValueError, match='larger than k'):
        diff.finite_difference_coefficients(0., [0., 1.], k=2)
    with pytest.raises(ValueError, match='min\\(xs\\) <= xbar'):
        diff.finite_difference_coefficients(2., [0., 1., 1.5], k=1)
    with pytest.raises(ValueError, match='at least length 2\\*radius\\+1'):
        diff.finite_difference_operator_midpoints(np.linspace(0., 1., 4), window_radius=2)
    with pytest.raises(ValueError, match='at least two nodes'):
        diff.finite_difference_operator_midpoints(np.zeros(1), window_radius=0)
