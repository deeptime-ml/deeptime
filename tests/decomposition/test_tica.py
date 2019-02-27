# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


"""
Created on 02.02.2015

@author: marscher
"""

import unittest

import numpy as np
from sktime.data.util import timeshifted_split
from sktime.decomposition.tica import TICA
from sktime.numeric.eigen import ZeroRankError


class TestTICA(unittest.TestCase):

    def test_fit_reset(self):
        chunk = 40
        lag = 100
        np.random.seed(0)
        data = np.random.randn(23000, 3)

        est = TICA(lagtime=lag, dim=1)
        for X, Y in timeshifted_split(data, lagtime=lag, chunksize=chunk):
            est.partial_fit((X, Y))
        model1 = est.fetch_model().copy()
        # ------- run again with new chunksize -------
        est.fit((data[:-lag], data[lag:]))
        model2 = est.fetch_model().copy()

        assert model1 != model2
        np.testing.assert_array_almost_equal(model1.mean_0, model2.mean_0)
        np.testing.assert_array_almost_equal(model1.cov_00, model2.cov_00)
        np.testing.assert_array_almost_equal(model1.cov_0t, model2.cov_0t)

    def test_constant_features(self):
        z = np.zeros((100, 10))
        o = np.ones((100, 10))
        z_lagged = (z[:-10], z[10:])
        o_lagged = (o[:-10], o[10:])
        tica_obj = TICA(lagtime=10)
        model = tica_obj.partial_fit(z_lagged).fetch_model()
        with self.assertRaises(ZeroRankError):
            _ = model.timescales
        with self.assertRaises(ZeroRankError):
            tica_obj.transform(z)
        model = tica_obj.partial_fit(o_lagged).fetch_model()
        try:
            _ = model.timescales
            tica_obj.transform(z)
        except ZeroRankError:
            self.fail('ZeroRankError was raised unexpectedly.')


def generate_hmm_test_data():
    import msmtools.generation as msmgen

    state = np.random.RandomState(123)

    # generate HMM with two Gaussians
    P = np.array([[0.99, 0.01],
                  [0.01, 0.99]])
    T = 40000
    means = [np.array([-1, 1]), np.array([1, -1])]
    widths = [np.array([0.3, 2]), np.array([0.3, 2])]
    # continuous trajectory
    X = np.zeros((T, 2))
    X2 = np.zeros((T, 2))
    # hidden trajectory
    dtraj = msmgen.generate_traj(P, T)
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
        cls.timeshifted_data_pair = (cls.data[:-cls.lagtime], cls.data[cls.lagtime:])

        # perform unscaled TICA
        cls.model_unscaled = TICA(cls.lagtime, dim=1, scaling=None).fit(cls.timeshifted_data_pair).fetch_model()
        cls.transformed_data_unscaled = cls.model_unscaled.transform(cls.data)

        # non-reversible TICA
        cls.model_nonrev = TICA(cls.lagtime, dim=1, scaling=None, reversible=False).fit(cls.timeshifted_data_pair) \
            .fetch_model()
        cls.transformed_data_nonrev = cls.model_nonrev.transform(cls.data)

    def test_variances(self):
        vars_unscaled = np.var(self.transformed_data_unscaled, axis=0)
        vars_nonrev = np.var(self.transformed_data_nonrev, axis=0)
        assert np.max(np.abs(vars_unscaled - 1.0)) < 0.01
        assert np.max(np.abs(vars_nonrev - 1.0)) < 0.01

    def test_kinetic_map(self):
        tica = TICA(lagtime=self.lagtime, dim=None, scaling='kinetic_map').fit(self.timeshifted_data_pair).fetch_model()
        O = tica.transform(self.data)
        vars = np.var(O, axis=0)
        refs = tica.eigenvalues ** 2
        assert np.max(np.abs(vars - refs)) < 0.01

    def test_cumvar(self):
        assert len(self.model_unscaled.cumvar) == 2
        assert np.allclose(self.model_unscaled.cumvar[-1], 1.0)
        assert len(self.model_nonrev.cumvar) == 2
        assert np.allclose(self.model_nonrev.cumvar[-1], 1.0)

    def test_cov(self):
        np.testing.assert_allclose(self.model_unscaled.cov_00, self.cov_ref_00)
        np.testing.assert_allclose(self.model_nonrev.cov_00, self.cov_ref_00_nr)
        np.testing.assert_allclose(self.model_unscaled.cov_0t, self.cov_ref_0t)
        np.testing.assert_allclose(self.model_nonrev.cov_0t, self.cov_ref_0t_nr)

    def test_dimension(self):
        assert self.model_unscaled.output_dimension() == 1
        assert self.model_nonrev.output_dimension() == 1
        # Test other variants
        model = TICA(lagtime=self.lagtime, dim=1.0).fit(self.timeshifted_data_pair).fetch_model()
        assert model.output_dimension() == 2
        model = TICA(lagtime=self.lagtime, dim=.9).fit(self.timeshifted_data_pair).fetch_model()
        assert model.output_dimension() == 1

        invalid_dims = [0, 0.0, 1.1, -1]
        for invalid_dim in invalid_dims:
            with self.assertRaises(ValueError):
                TICA(lagtime=self.lagtime, dim=invalid_dim)


if __name__ == "__main__":
    unittest.main()
