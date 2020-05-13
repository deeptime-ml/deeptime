# This file is part of PyEMMA.
#
# Copyright (c) 2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
@author: paul
"""

import unittest

import numpy as np
import pytest

from sktime.data.util import timeshifted_split
from sktime.decomposition.vamp import VAMP
from sktime.markov._base import cvsplit_dtrajs
from tests.markov.msm.test_mlmsm import estimate_markov_model


def random_matrix(n, rank=None, eps=0.01):
    m = np.random.randn(n, n)
    u, s, v = np.linalg.svd(m)
    if rank is None:
        rank = n
    if rank > n:
        rank = n
    s = np.concatenate((np.maximum(s, eps)[0:rank], np.zeros(n - rank)))
    return u.dot(np.diag(s)).dot(v)


@pytest.mark.parametrize("with_statistics", [True, False], ids=["w/ statistics", "w/o statistics"])
def test_expectation_sanity(with_statistics):
    data = np.random.normal(size=(10000, 5))
    model = VAMP(lagtime=1).fit(data).fetch_model()
    observations = np.random.normal(size=(100, 5))
    if with_statistics:
        statistics = np.random.normal(size=(100, 5)).T
    else:
        statistics = None
    model.expectation(observations.T, statistics)


class TestVAMPEstimatorSelfConsistency(unittest.TestCase):
    def test_full_rank(self):
        self.do_test(20, 20, test_partial_fit=True)

    def test_low_rank(self):
        dim = 30
        rank = 15
        self.do_test(dim, rank, test_partial_fit=True)

    def do_test(self, dim, rank, test_partial_fit=False):
        # setup
        N_frames = [123, 456, 789]
        N_trajs = len(N_frames)
        A = random_matrix(dim, rank)
        trajs = []
        mean = np.random.randn(dim)
        for i in range(N_trajs):
            # set up data
            white = np.random.randn(N_frames[i], dim)
            brown = np.cumsum(white, axis=0)
            correlated = np.dot(brown, A)
            trajs.append(correlated + mean)

        # test
        tau = 50
        vamp = VAMP(lagtime=tau, scaling=None, right=True).fit(trajs).fetch_model()

        assert vamp.output_dimension <= rank

        atol = np.finfo(np.float32).eps * 10.0
        rtol = np.finfo(np.float32).resolution
        phi_trajs = [vamp.transform(X)[tau:, :] for X in trajs]
        phi = np.concatenate(phi_trajs)
        mean_right = phi.sum(axis=0) / phi.shape[0]
        cov_right = phi.T.dot(phi) / phi.shape[0]
        np.testing.assert_allclose(mean_right, 0.0, rtol=rtol, atol=atol)
        np.testing.assert_allclose(cov_right, np.eye(vamp.output_dimension), rtol=rtol, atol=atol)

        vamp.right = False
        # vamp = estimate_vamp(trajs, lag=tau, scaling=None, right=False)
        psi_trajs = [vamp.transform(X)[0:-tau, :] for X in trajs]
        psi = np.concatenate(psi_trajs)
        mean_left = psi.sum(axis=0) / psi.shape[0]
        cov_left = psi.T.dot(psi) / psi.shape[0]
        np.testing.assert_allclose(mean_left, 0.0, rtol=rtol, atol=atol)
        np.testing.assert_allclose(cov_left, np.eye(vamp.output_dimension), rtol=rtol, atol=atol)

        # compute correlation between left and right
        assert phi.shape[0] == psi.shape[0]
        C01_psi_phi = psi.T.dot(phi) / phi.shape[0]
        n = max(C01_psi_phi.shape)
        C01_psi_phi = C01_psi_phi[0:n, :][:, 0:n]
        np.testing.assert_allclose(C01_psi_phi, np.diag(vamp.singular_values[0:vamp.output_dimension]), rtol=rtol, atol=atol)

        if test_partial_fit:
            vamp2 = VAMP(lagtime=tau, scaling=None).fit(trajs).fetch_model()

            atol = 1e-14
            rtol = 1e-5

            np.testing.assert_allclose(vamp.singular_values, vamp2.singular_values)
            np.testing.assert_allclose(vamp.mean_0, vamp2.mean_0, atol=atol, rtol=rtol)
            np.testing.assert_allclose(vamp.mean_t, vamp2.mean_t, atol=atol, rtol=rtol)
            np.testing.assert_allclose(vamp.cov_00, vamp2.cov_00, atol=atol, rtol=rtol)
            np.testing.assert_allclose(vamp.cov_0t, vamp2.cov_0t, atol=atol, rtol=rtol)
            np.testing.assert_allclose(vamp.cov_tt, vamp2.cov_tt, atol=atol, rtol=rtol)
            np.testing.assert_allclose(vamp.epsilon, vamp2.epsilon, atol=atol, rtol=rtol)
            np.testing.assert_allclose(vamp.output_dimension, vamp2.output_dimension, atol=atol, rtol=rtol)
            np.testing.assert_equal(vamp.scaling, vamp2.scaling)
            assert_allclose_ignore_phase(vamp.singular_vectors_left, vamp2.singular_vectors_left, atol=atol)
            assert_allclose_ignore_phase(vamp.singular_vectors_right, vamp2.singular_vectors_right, atol=rtol)

            # vamp2.singular_values # trigger diagonalization
            for t, ref in zip(trajs, phi_trajs):
                assert_allclose_ignore_phase(vamp2.transform(t[tau:], right=True), ref, rtol=rtol, atol=atol)

            for t, ref in zip(trajs, psi_trajs):
                assert_allclose_ignore_phase(vamp2.transform(t[0:-tau], right=False), ref, rtol=rtol, atol=atol)


def generate(T, N_steps, s0=0):
    dtraj = np.zeros(N_steps, dtype=int)
    s = s0
    T_cdf = T.cumsum(axis=1)
    for t in range(N_steps):
        dtraj[t] = s
        s = np.searchsorted(T_cdf[s, :], np.random.rand())
    return dtraj


def assert_allclose_ignore_phase(A, B, atol=1e-14, rtol=1e-5):
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    assert A.shape == B.shape
    for i in range(B.shape[1]):
        assert (np.allclose(A[:, i], B[:, i], atol=atol, rtol=rtol)
                or np.allclose(A[:, i], -B[:, i], atol=atol, rtol=rtol))


class TestVAMPModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        N_steps = 10000
        N_traj = 20
        lag = 1
        T = np.linalg.matrix_power(np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]), lag)
        dtrajs = [generate(T, N_steps) for _ in range(N_traj)]
        p0 = np.zeros(3)
        p1 = np.zeros(3)
        trajs = []
        for dtraj in dtrajs:
            traj = np.zeros((N_steps, T.shape[0]))
            traj[np.arange(len(dtraj)), dtraj] = 1.0
            trajs.append(traj)
            p0 += traj[:-lag, :].sum(axis=0)
            p1 += traj[lag:, :].sum(axis=0)
        estimator = VAMP(lagtime=lag, scaling=None, dim=1.0)
        vamp = estimator.fit(trajs).fetch_model()
        msm = estimate_markov_model(dtrajs, lag=lag, reversible=False)
        cls.trajs = trajs
        cls.dtrajs = dtrajs
        cls.trajs_timeshifted = list(timeshifted_split(cls.trajs, lagtime=lag, chunksize=5000))
        cls.lag = lag
        cls.msm = msm
        cls.vamp = vamp
        cls.estimator = estimator
        cls.p0 = p0 / p0.sum()
        cls.p1 = p1 / p1.sum()
        cls.atol = np.finfo(np.float32).eps * 1000.0

    def test_K_is_T(self):
        m0 = self.vamp.mean_0
        mt = self.vamp.mean_t
        C0 = self.vamp.cov_00 + m0[:, np.newaxis] * m0[np.newaxis, :]
        C1 = self.vamp.cov_0t + m0[:, np.newaxis] * mt[np.newaxis, :]
        K = np.linalg.inv(C0).dot(C1)
        np.testing.assert_allclose(K, self.msm.transition_matrix, atol=1E-5)

        Tsym = np.diag(self.p0 ** 0.5).dot(self.msm.transition_matrix).dot(np.diag(self.p1 ** -0.5))
        np.testing.assert_allclose(np.linalg.svd(Tsym)[1][1:], self.vamp.singular_values[0:2], atol=1E-7)

    def test_singular_functions_against_MSM(self):
        Tsym = np.diag(self.p0 ** 0.5).dot(self.msm.transition_matrix).dot(np.diag(self.p1 ** -0.5))
        Up, S, Vhp = np.linalg.svd(Tsym)
        Vp = Vhp.T
        U = Up * (self.p0 ** -0.5)[:, np.newaxis]
        V = Vp * (self.p1 ** -0.5)[:, np.newaxis]
        assert_allclose_ignore_phase(U[:, 0], np.ones(3), atol=1E-5)
        assert_allclose_ignore_phase(V[:, 0], np.ones(3), atol=1E-5)
        U = U[:, 1:]
        V = V[:, 1:]
        self.vamp.right = True
        phi = self.vamp.transform(np.eye(3))
        self.vamp.right = False
        psi = self.vamp.transform(np.eye(3))
        assert_allclose_ignore_phase(U, psi, atol=1E-5)
        assert_allclose_ignore_phase(V, phi, atol=1E-5)

        cumsum_Tsym = np.cumsum(S[1:] ** 2)
        cumsum_Tsym /= cumsum_Tsym[-1]
        np.testing.assert_allclose(self.vamp.cumvar, cumsum_Tsym)

    def test_cumvar_variance_cutoff(self):
        for d in (0.2, 0.5, 0.8, 0.9, 1.0):
            self.vamp.dim = d
            special_cumvar = np.asarray([0] + self.vamp.cumvar.tolist())
            self.assertLessEqual(d, special_cumvar[self.vamp.output_dimension], )
            self.assertLessEqual(special_cumvar[self.vamp.output_dimension - 1], d)

    def test_self_score_with_MSM(self):
        T = self.msm.transition_matrix
        Tadj = np.diag(1. / self.p1).dot(T.T).dot(np.diag(self.p0))
        NFro = np.trace(T.dot(Tadj))
        s2 = self.vamp.score(score_method='VAMP2')
        np.testing.assert_allclose(s2, NFro)

        Tsym = np.diag(self.p0 ** 0.5).dot(T).dot(np.diag(self.p1 ** -0.5))
        Nnuc = np.linalg.norm(Tsym, ord='nuc')
        s1 = self.vamp.score(score_method='VAMP1')
        np.testing.assert_allclose(s1, Nnuc)

        sE = self.vamp.score(score_method='VAMPE')
        np.testing.assert_allclose(sE, NFro)  # see paper appendix H.2

    def test_score_vs_MSM(self):
        trajs_test, trajs_train = cvsplit_dtrajs(self.trajs, random_state=32)
        dtrajs_test, dtrajs_train = cvsplit_dtrajs(self.dtrajs, random_state=32)

        methods = ('VAMP1', 'VAMP2', 'VAMPE')

        for m in methods:
            msm_train = estimate_markov_model(dtrajs=dtrajs_train, lag=self.lag, reversible=False)
            score_msm = msm_train.score(dtrajs_test, score_method=m, score_k=None)

            vamp_train = VAMP(lagtime=self.lag, dim=1.0).fit(trajs_train).fetch_model()
            vamp_test = VAMP(lagtime=self.lag, dim=1.0).fit(trajs_test).fetch_model()
            score_vamp = vamp_train.score(test_model=vamp_test, score_method=m)

            self.assertAlmostEqual(score_msm, score_vamp, places=2 if m == 'VAMPE' else 3, msg=m)

    def test_kinetic_map(self):
        lag = 10
        vamp = VAMP(lagtime=lag, scaling='km', right=False).fit(self.trajs).fetch_model()
        transformed = [vamp.transform(X)[:-lag] for X in self.trajs]
        std = np.std(np.concatenate(transformed), axis=0)
        np.testing.assert_allclose(std, vamp.singular_values[:vamp.output_dimension], atol=1e-4, rtol=1e-4)


class TestVAMPWithEdgeCaseData(unittest.TestCase):
    def test_1D_data(self):
        x = np.random.randn(10, 1)
        vamp = VAMP(lagtime=1, right=True).fit([x]).fetch_model()
        # Doing VAMP with 1-D data is just centering and normalizing the data.
        assert_allclose_ignore_phase(vamp.transform(x), (x - np.mean(x[1:, 0])) / np.std(x[1:, 0]))

    def test_const_data(self):
        from sktime.numeric.eigen import ZeroRankError
        with self.assertRaises(ZeroRankError):
            print(VAMP(lagtime=1).fit([np.ones((10, 2))]).fetch_model().singular_values)
        with self.assertRaises(ZeroRankError):
            print(VAMP(lagtime=1).fit([np.ones((10, 1))]).fetch_model().singular_values)


if __name__ == "__main__":
    unittest.main()
