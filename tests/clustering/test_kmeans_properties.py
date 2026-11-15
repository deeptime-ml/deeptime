"""Hypothesis properties of the k-means bindings.

CI runs each property for about a second. To fuzz for longer:  pytest tests/clustering -m fuzz --fuzz-seconds 600
Failures are shrunk and saved to .hypothesis/, the next run replays them first.
"""
import numpy as np
from hypothesis import given, target, strategies as st
from hypothesis.extra import numpy as hnp
from numpy.testing import assert_array_equal

import deeptime as dt
from tests.testing_utilities import fuzz

impl = dt.clustering.metrics['euclidean']


@st.composite
def frames_and_centers(draw, max_frames=64, max_dim=6, max_k=12, min_magnitude=0.):
    """min_magnitude > 0: nonzero values at least that large, so that squared differences do not underflow."""
    dtype = np.dtype(draw(st.sampled_from([np.float32, np.float64])))
    n = draw(st.integers(1, max_frames))
    dim = draw(st.integers(1, max_dim))
    width = 8 * dtype.itemsize
    if min_magnitude > 0:
        magnitudes = st.floats(min_magnitude, 1e3, width=width)
        floats = st.one_of(st.just(0.), magnitudes, magnitudes.map(lambda x: -x))
    else:
        floats = st.floats(-1e3, 1e3, width=width, allow_nan=False, allow_infinity=False)
    # small integers give exact ties, floats everything else
    elements = st.one_of(st.integers(-8, 8).map(float), floats)
    X = draw(hnp.arrays(dtype, (n, dim), elements=elements))
    if draw(st.booleans()):
        k = draw(st.integers(1, min(n, max_k)))
        C = X[draw(st.lists(st.integers(0, n - 1), min_size=k, max_size=k))]
    else:
        C = draw(hnp.arrays(dtype, (draw(st.integers(1, max_k)), dim), elements=elements))
    return np.ascontiguousarray(X), np.ascontiguousarray(C)


def squared_distances(X, C):
    X, C = X.astype(np.float64), C.astype(np.float64)
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)


def assert_nearest(X, C, labels):
    """Every frame is assigned to a closest center, up to the rounding error of a float distance."""
    d2 = squared_distances(X, C)
    eps = np.finfo(X.dtype).eps
    scale = (X.astype(np.float64) ** 2).sum(1) + (C.astype(np.float64) ** 2).sum(1).max()
    allowed = 4 * (X.shape[1] + 2) * eps * scale + 1e3 * np.finfo(X.dtype).tiny
    excess = d2[np.arange(len(X)), labels] - d2.min(1)
    assert (excess <= allowed).all(), f"frame {np.argmax(excess - allowed)} not assigned to a closest center"


def target_near_ties(X, C):
    if len(C) > 1:
        d2 = np.sort(squared_distances(X, C), axis=1)
        target(-float(np.min((d2[:, 1] - d2[:, 0]) / (d2[:, 1] + 1e-300))), label="relative gap to 2nd center")


@fuzz
@given(frames_and_centers(), st.sampled_from([1, 2, 3]))
def test_assign_is_nearest(data, n_threads):
    X, C = data
    target_near_ties(X, C)
    labels = impl.assign(X, C, n_threads)
    assert labels.shape == (len(X),) and labels.min() >= 0 and labels.max() < len(C)
    assert_nearest(X, C, labels)


@fuzz
@given(frames_and_centers(), st.sampled_from([1, 2, 3]))
def test_step_moves_centers_to_means(data, n_threads):
    X, C = data
    target_near_ties(X, C)
    new_centers, labels = impl.kmeans.cluster(X, C, n_threads)
    assert new_centers.shape == C.shape and new_centers.dtype == X.dtype
    assert_nearest(X, C, labels)
    allowed = 4 * len(X) * np.finfo(X.dtype).eps * np.abs(X).max(initial=0) + 1e3 * np.finfo(X.dtype).tiny
    for j in range(len(C)):
        members = X[labels == j]
        if len(members) == 0:
            assert_array_equal(new_centers[j], C[j], err_msg=f"center {j} without frames moved")
        else:
            np.testing.assert_allclose(new_centers[j], members.astype(np.float64).mean(0), rtol=0, atol=allowed,
                                       err_msg=f"center {j} is not the mean of its frames")


@fuzz
@given(frames_and_centers(), st.integers(1, 6))
def test_loop_equals_steps(data, max_iter):
    X, C = data
    # tolerance -1 never converges, so the loop runs max_iter steps
    centers, code, iterations, inertias = impl.kmeans.cluster_loop(X, C.copy(), 1, max_iter, -1., None)
    assert (code, iterations, len(inertias)) == (1, max_iter, max_iter)
    step = C
    for _ in range(max_iter):
        step, _ = impl.kmeans.cluster(X, step, 1)
    assert_array_equal(centers, step)


@fuzz
@given(frames_and_centers(), st.integers(1, 6), st.sampled_from([1, 2, 3]))
def test_inertias(data, max_iter, n_threads):
    """Lloyd never increases the cost; the reported inertia is at least the cost of the returned centers."""
    X, C = data
    centers, _, _, inertias = impl.kmeans.cluster_loop(X, C.copy(), n_threads, max_iter, -1., None)
    inertias = inertias.astype(np.float64)
    cost = squared_distances(X, centers).min(1).sum()
    slack = 8 * (len(X) + X.shape[1]) * np.finfo(X.dtype).eps * max(inertias.max(), cost) + 1e3 * np.finfo(X.dtype).tiny
    assert (np.diff(inertias) <= slack).all(), f"inertia increased: {inertias}"
    assert inertias[-1] >= cost - slack, f"inertia {inertias[-1]} below the cost {cost} of the returned centers"


@fuzz
@given(frames_and_centers(), st.integers(0, 2 ** 31 - 1))
def test_kmeans_plusplus_picks_frames(data, seed):
    X, C = data
    k = min(len(C), len(X))
    centers = impl.kmeans.init_centers_kmpp(X, k, seed, 1, None)
    assert centers.shape == (k, X.shape[1])
    assert (centers[:, None, :] == X[None, :, :]).all(-1).any(1).all(), "a center is not a frame"
    assert_array_equal(centers, impl.kmeans.init_centers_kmpp(X, k, seed, 1, None), err_msg="not reproducible")


@fuzz
@given(frames_and_centers(min_magnitude=2. ** -10), st.integers(-20, 20), st.integers(1, 30),
       st.sampled_from([0., 1e-5, -1.]), st.integers(0, 2 ** 31 - 1))
def test_power_of_two_scaling_is_exact(data, exponent, max_iter, tolerance, seed):
    """Scaling by 2**m is exact in floating point and commutes with every operation of k-means, bit by bit."""
    X, C = data
    s = X.dtype.type(2.) ** exponent
    k = min(len(C), len(X))
    assert_array_equal(impl.kmeans.init_centers_kmpp(X * s, k, seed, 1, None),
                       impl.kmeans.init_centers_kmpp(X, k, seed, 1, None) * s, err_msg="k-means++")
    centers, code, iterations, inertias = impl.kmeans.cluster_loop(X, C.copy(), 1, max_iter, tolerance, None)
    scaled = impl.kmeans.cluster_loop(X * s, C * s, 1, max_iter, tolerance, None)
    assert (scaled[1], scaled[2]) == (code, iterations)
    assert_array_equal(scaled[0], centers * s, err_msg="centers")
    assert_array_equal(scaled[3], inertias * s * s, err_msg="inertias")


@fuzz
@given(frames_and_centers(), st.sampled_from([1, 2, 3]))
def test_converged_centers_are_a_fixed_point(data, n_threads):
    """Converged centers are the means of the frames closest to them, so one more step does not move them."""
    X, C = data
    centers, code, _, _ = impl.kmeans.cluster_loop(X, C.copy(), n_threads, 300, 0., None)
    if code != 0:
        return
    target_near_ties(X, centers)
    stepped, _ = impl.kmeans.cluster(X, centers, n_threads)
    allowed = 4 * len(X) * np.finfo(X.dtype).eps * np.abs(X).max(initial=0) + 1e3 * np.finfo(X.dtype).tiny
    np.testing.assert_allclose(stepped, centers, rtol=0, atol=allowed, err_msg="converged, but a step moves centers")
