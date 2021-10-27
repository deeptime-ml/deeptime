"""
Created on 02.02.2015

@author: marscher, clonker
"""

import unittest

import numpy as np
import pytest

from deeptime.covariance import Covariance
from deeptime.data import ellipsoids
from deeptime.decomposition import TICA, VAMP
from deeptime.markov.msm import MarkovStateModel
from deeptime.numeric import ZeroRankError


def test_fit_reset():
    lag = 100
    np.random.seed(0)
    data = np.random.randn(23000, 3)

    estimator = TICA(dim=1, lagtime=lag)
    model1 = estimator.fit_from_timeseries(data).fetch_model()
    # ------- run again with new chunksize -------
    covars = TICA.covariance_estimator(lagtime=lag).fit(data)
    estimator.fit_from_covariances(covars)
    model2 = estimator.fetch_model()

    assert model1 != model2
    np.testing.assert_array_almost_equal(model1.mean_0, model2.mean_0)
    np.testing.assert_array_almost_equal(model1.cov_00, model2.cov_00)
    np.testing.assert_array_almost_equal(model1.cov_0t, model2.cov_0t)


def test_constant_features():
    z = np.zeros((100, 10))
    o = np.ones((100, 10))
    z_lagged = (z[:-10], z[10:])
    o_lagged = (o[:-10], o[10:])
    tica_obj = TICA()
    cov_estimator = TICA.covariance_estimator(lagtime=1)
    cov_estimator.partial_fit(z_lagged)
    with np.testing.assert_raises(ZeroRankError):
        model = tica_obj.fit(cov_estimator.fetch_model())
        _ = model.timescales(lagtime=1)
        tica_obj.transform(z)
    cov_estimator.partial_fit(o_lagged)
    try:
        model = tica_obj.fit(cov_estimator).fetch_model()
        _ = model.timescales(lagtime=1)
        tica_obj.transform(z)
    except ZeroRankError:
        pytest.fail('ZeroRankError was raised unexpectedly.')


@pytest.mark.parametrize('estimator', [TICA, VAMP], ids=lambda clazz: clazz.__name__)
def test_multiple_fetch(estimator):
    data = np.random.normal(size=(10000, 5))
    est = estimator(lagtime=1)
    m1 = est.fit(data).model
    m2 = est.model
    m3 = est.partial_fit((data, data)).model
    np.testing.assert_(m1 is m2)
    np.testing.assert_(m1 is not m3)
    np.testing.assert_(m2 is not m3)


def test_vamp_consistency():
    trajectory = ellipsoids(seed=13).observations(10000, n_dim=50)
    cov_estimator = VAMP.covariance_estimator(lagtime=1)
    cov_estimator.compute_ctt = False
    cov_estimator.reversible = True
    cov_estimator.fit(trajectory)
    koopman1 = VAMP(dim=2).fit(cov_estimator).fetch_model()
    koopman2 = TICA(dim=2, scaling=None, lagtime=1).fit(trajectory).fetch_model()
    np.testing.assert_array_almost_equal(koopman1.singular_values, koopman2.singular_values, decimal=1)
    np.testing.assert_array_almost_equal(
        np.abs(koopman1.singular_vectors_left),
        np.abs(koopman2.singular_vectors_left),
        decimal=2)
    np.testing.assert_array_almost_equal(
        np.abs(koopman1.singular_vectors_right),
        np.abs(koopman2.singular_vectors_right),
        decimal=2)
    np.testing.assert_array_almost_equal(koopman1.timescales(), koopman2.timescales(), decimal=2)


def test_dim_parameter():
    np.testing.assert_equal(TICA(dim=3).dim, 3)
    np.testing.assert_equal(TICA(var_cutoff=.5).var_cutoff, 0.5)
    with np.testing.assert_raises(ValueError):
        TICA(dim=-1)  # negative int
    with np.testing.assert_raises(ValueError):
        TICA(var_cutoff=5.5)  # float > 1
    with np.testing.assert_raises(ValueError):
        TICA(var_cutoff=-0.1)  # negative float


@pytest.mark.parametrize("scaling_param", [(None, True), ('kinetic_map', True), ('km', True),
                                           ('commute_map', True), ('bogus', False)],
                         ids=lambda x: f"scaling-{x[0]}-valid-{x[1]}")
def test_scaling_parameter(scaling_param):
    scaling, valid_scaling = scaling_param
    if valid_scaling:
        # set via ctor
        estimator = TICA(scaling=scaling)
        np.testing.assert_equal(estimator.scaling, scaling)

        # set via property
        estimator = TICA()
        estimator.scaling = scaling
        np.testing.assert_equal(estimator.scaling, scaling)
    else:
        with np.testing.assert_raises(ValueError):
            TICA(scaling=scaling)
        with np.testing.assert_raises(ValueError):
            TICA().scaling = scaling


def test_fit_from_cov():
    data = np.random.normal(size=(500, 3))
    # fitting with C0t and symmetric covariances, should pass
    TICA().fit(Covariance(1, compute_c0t=True, reversible=True).fit(data))

    with np.testing.assert_raises(ValueError):
        TICA().fit(Covariance(1, compute_c0t=True, reversible=False).fit(data))

    with np.testing.assert_raises(ValueError):
        TICA().fit(Covariance(1, compute_c0t=False, reversible=True).fit(data))


def generate_hmm_test_data():
    state = np.random.RandomState(123)

    # generate HMM with two Gaussians
    P = np.array([[0.99, 0.01],
                  [0.01, 0.99]])
    T = 40000
    means = [np.array([-1, 1]), np.array([1, -1])]
    widths = [np.array([0.3, 2]), np.array([0.3, 2])]
    # continuous trajectory
    X = np.zeros((T, 2))
    # hidden trajectory
    dtraj = MarkovStateModel(P).simulate(n_steps=T)
    means = np.array(means)
    widths = np.array(widths)

    normal_vals = state.normal(size=(len(X), 2))

    X[:, 0] = means[dtraj][:, 0] + widths[dtraj][:, 0] * normal_vals[:, 0]
    X[:, 1] = means[dtraj][:, 1] + widths[dtraj][:, 1] * normal_vals[:, 1]

    # Set the lag time:
    lag = 10
    # Compute mean free data:
    mref = (np.sum(X[:-lag, :], axis=0) +
            np.sum(X[lag:, :], axis=0)) / float(2 * (T - lag))
    mref_nr = np.sum(X[:-lag, :], axis=0) / float(T - lag)
    X_mf = X - mref[None, :]
    X_mf_nr = X - mref_nr[None, :]
    # Compute correlation matrices:
    cov_ref = (np.dot(X_mf[:-lag, :].T, X_mf[:-lag, :]) + np.dot(X_mf[lag:, :].T, X_mf[lag:, :])) / float(2 * (T - lag))
    cov_ref_nr = np.dot(X_mf_nr[:-lag, :].T, X_mf_nr[:-lag, :]) / float(T - lag)
    cov_tau_ref = (np.dot(X_mf[:-lag, :].T, X_mf[lag:, :]) + np.dot(X_mf[lag:, :].T, X_mf[:-lag, :])) / float(
        2 * (T - lag))
    cov_tau_ref_nr = np.dot(X_mf_nr[:-lag, :].T, X_mf_nr[lag:, :]) / float(T - lag)

    return dict(lagtime=lag, cov_ref_00=cov_ref, cov_ref_00_nr=cov_ref_nr, cov_ref_0t=cov_tau_ref,
                cov_ref_0t_nr=cov_tau_ref_nr, data=X)


class TestTICAExtensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_data = generate_hmm_test_data()
        cls.lagtime = test_data['lagtime']
        cls.cov_ref_00 = test_data['cov_ref_00']
        cls.cov_ref_00_nr = test_data['cov_ref_00_nr']
        cls.cov_ref_0t = test_data['cov_ref_0t']
        cls.cov_ref_0t_nr = test_data['cov_ref_0t_nr']
        cls.data = test_data['data']

        # perform unscaled TICA
        cls.model_unscaled = TICA(dim=1, lagtime=cls.lagtime, scaling=None).fit_from_timeseries(cls.data).fetch_model()
        cls.transformed_data_unscaled = cls.model_unscaled.transform(cls.data, propagate=False)

    def test_variances(self):
        vars_unscaled = np.var(self.transformed_data_unscaled, axis=0)
        assert np.max(np.abs(vars_unscaled - 1.0)) < 0.01

    def test_kinetic_map(self):
        tica = TICA(scaling='km', dim=None, lagtime=self.lagtime).fit(self.data).fetch_model()
        O = tica.transform(self.data, propagate=False)
        vars = np.var(O, axis=0)
        refs = tica.singular_values ** 2
        assert np.max(np.abs(vars - refs)) < 0.01

    def test_commute_map(self):
        # todo: this is just a sanity check for now, something more meaningful should be tested
        TICA(scaling='commute_map', dim=None, lagtime=self.lagtime).fit(self.data).fetch_model()

    def test_cumvar(self):
        assert len(self.model_unscaled.cumulative_kinetic_variance) == 2
        assert np.allclose(self.model_unscaled.cumulative_kinetic_variance[-1], 1.0)

    def test_cov(self):
        np.testing.assert_allclose(self.model_unscaled.cov_00, self.cov_ref_00)
        np.testing.assert_allclose(self.model_unscaled.cov_0t, self.cov_ref_0t)

    def test_dimension(self):
        assert self.model_unscaled.output_dimension == 1
        # Test other variants
        model = TICA(var_cutoff=1.0, lagtime=self.lagtime).fit(self.data).fetch_model()
        assert model.output_dimension == 2
        model = TICA(var_cutoff=.9, lagtime=self.lagtime).fit(self.data).fetch_model()
        assert model.output_dimension == 1

        invalid_dims = [0, 0.0, 1.1, -1]
        for invalid_dim in invalid_dims:
            with self.assertRaises(ValueError):
                TICA(dim=invalid_dim)
