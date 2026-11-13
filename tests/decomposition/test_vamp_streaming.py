import numpy as np
import pytest

from deeptime.decomposition import TICA, VAMP


@pytest.fixture
def batches():
    rng = np.random.default_rng(42)
    return [(rng.normal(size=(24, 3)) + shift, rng.normal(size=(24, 3)) - shift)
            for shift in (0., 2., -1.)]


@pytest.mark.parametrize('estimator', [VAMP, TICA])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('accessor', ['fetch_model', 'model', 'transform'])
def test_partial_fit_survives_model_access(batches, estimator, weighted, accessor):
    reference = estimator(lagtime=2, scaling=None)
    streaming = estimator(lagtime=2, scaling=None)
    snapshots = []
    for i, (x, y) in enumerate(batches):
        weights = np.linspace(1., 2. + i, len(x)) if weighted else None
        reference.partial_fit((x, y), weights=weights)
        streaming.partial_fit((x, y), weights=weights)
        if accessor == 'fetch_model':
            model = streaming.fetch_model()
        elif accessor == 'model':
            model = streaming.model
        else:
            streaming.transform(x)
            model = streaming.fetch_model()
        assert streaming.fetch_model() is model
        snapshots.append((model, model.mean_0.copy(), model.cov_00.copy()))

    expected = reference.fetch_model()
    actual = streaming.fetch_model()
    for name in ('mean_0', 'mean_t', 'cov_00', 'cov_0t', 'cov_tt', 'singular_values'):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), atol=1e-12)
    np.testing.assert_allclose(actual.transform(batches[0][0]), expected.transform(batches[0][0]), atol=1e-12)
    assert actual.lagtime == expected.lagtime == 2
    for model, mean, covariance in snapshots:
        np.testing.assert_array_equal(model.mean_0, mean)
        np.testing.assert_array_equal(model.cov_00, covariance)


@pytest.mark.parametrize('estimator', [VAMP, TICA])
@pytest.mark.parametrize('from_covariances', [False, True])
def test_fit_resets_partial_fit_after_model_access(batches, estimator, from_covariances):
    streaming = estimator(lagtime=2, scaling=None)
    streaming.partial_fit(batches[0]).fetch_model()
    if from_covariances:
        cov = estimator.covariance_estimator(lagtime=2).partial_fit(batches[1]).fetch_model()
        streaming.fit_from_covariances(cov)
    else:
        streaming.fit(batches[1])
    expected = estimator(lagtime=2, scaling=None).fit_fetch(batches[1])
    actual = streaming.fetch_model()
    np.testing.assert_allclose(actual.mean_0, expected.mean_0)
    np.testing.assert_allclose(actual.cov_00, expected.cov_00)

    # An explicit fit ends the old stream; a subsequent partial_fit starts a new one.
    actual = streaming.partial_fit(batches[2]).fetch_model()
    expected = estimator(lagtime=2, scaling=None).partial_fit(batches[2]).fetch_model()
    np.testing.assert_allclose(actual.mean_0, expected.mean_0)
    np.testing.assert_allclose(actual.cov_00, expected.cov_00)


@pytest.mark.parametrize('estimator_cls', [VAMP, TICA])
def test_new_lagtime_after_model_access(estimator_cls):
    rng = np.random.default_rng(871)
    a = rng.normal(size=(30, 3))
    b = rng.normal(size=(40, 3)) + 5
    est = estimator_cls(lagtime=1, scaling=None).partial_fit((a[:-1], a[1:]))
    est.fetch_model()
    est.set_params(lagtime=2)
    actual = est.partial_fit((b[:-2], b[2:])).fetch_model()
    expected = estimator_cls(lagtime=2, scaling=None).partial_fit((b[:-2], b[2:])).fetch_model()
    assert actual.lagtime == est.lagtime == expected.lagtime == 2
    # The former lag-1 trajectory cannot be combined with lag-2 statistics.
    for attr in ('mean_0', 'mean_t', 'cov_00', 'cov_0t', 'cov_tt'):
        np.testing.assert_allclose(getattr(actual, attr), getattr(expected, attr), rtol=1e-12, atol=1e-12)
