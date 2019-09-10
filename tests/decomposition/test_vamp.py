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

from sktime.data.util import timeshifted_split
from sktime.decomposition.vamp import VAMP, VAMPModel, vamp_cktest
from sktime.markovprocess.transition_counting import cvsplit_dtrajs
from tests.markovprocess.test_msm import estimate_markov_model


def estimate_vamp(data, lag, partial_fit=False, return_estimator=False, **kwargs) -> VAMPModel:
    estimator = VAMP(lagtime=lag, **kwargs)
    estimator.fit(data)

    m = estimator.fetch_model()
    if return_estimator:
        return estimator, m
    return m


def random_matrix(n, rank=None, eps=0.01):
    m = np.random.randn(n, n)
    u, s, v = np.linalg.svd(m)
    if rank is None:
        rank = n
    if rank > n:
        rank = n
    s = np.concatenate((np.maximum(s, eps)[0:rank], np.zeros(n - rank)))
    return u.dot(np.diag(s)).dot(v)


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
        vamp = estimate_vamp(trajs, lag=tau, scaling=None)
        vamp.right = True

        assert vamp.dimension() <= rank

        atol = np.finfo(np.float32).eps * 10.0
        rtol = np.finfo(np.float32).resolution
        phi_trajs = [vamp.transform(X)[tau:, :] for X in trajs]
        phi = np.concatenate(phi_trajs)
        mean_right = phi.sum(axis=0) / phi.shape[0]
        cov_right = phi.T.dot(phi) / phi.shape[0]
        np.testing.assert_allclose(mean_right, 0.0, rtol=rtol, atol=atol)
        np.testing.assert_allclose(cov_right, np.eye(vamp.dimension()), rtol=rtol, atol=atol)

        vamp.right = False
        psi_trajs = [vamp.transform(X)[0:-tau, :] for X in trajs]
        psi = np.concatenate(psi_trajs)
        mean_left = psi.sum(axis=0) / psi.shape[0]
        cov_left = psi.T.dot(psi) / psi.shape[0]
        np.testing.assert_allclose(mean_left, 0.0, rtol=rtol, atol=atol)
        np.testing.assert_allclose(cov_left, np.eye(vamp.dimension()), rtol=rtol, atol=atol)

        # compute correlation between left and right
        assert phi.shape[0] == psi.shape[0]
        C01_psi_phi = psi.T.dot(phi) / phi.shape[0]
        n = max(C01_psi_phi.shape)
        C01_psi_phi = C01_psi_phi[0:n, :][:, 0:n]
        np.testing.assert_allclose(C01_psi_phi, np.diag(vamp.singular_values[0:vamp.dimension()]), rtol=rtol, atol=atol)

        if test_partial_fit:
            vamp2 = estimate_vamp(data=trajs, lag=tau, scaling=None, partial_fit=True)

            model_params = vamp.get_params()
            model_params2 = vamp2.get_params()

            atol = 1e-14
            rtol = 1e-5

            for n in model_params.keys():
                if model_params[n] is not None and model_params2[n] is not None:
                    if n not in ('U', 'V'):
                        np.testing.assert_allclose(model_params[n], model_params2[n], rtol=rtol, atol=atol,
                                                   err_msg='failed for model param %s' % n)
                    else:
                        assert_allclose_ignore_phase(model_params[n], model_params2[n], atol=atol)

            # vamp2.singular_values # trigger diagonalization
            vamp2.right = True
            for t, ref in zip(trajs, phi_trajs):
                assert_allclose_ignore_phase(vamp2.transform(t[tau:]), ref, rtol=rtol, atol=atol)

            vamp2.right = False
            for t, ref in zip(trajs, psi_trajs):
                assert_allclose_ignore_phase(vamp2.transform(t[0:-tau]), ref, rtol=rtol, atol=atol)


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
        estimator, vamp = estimate_vamp(trajs, lag=lag, scaling=None, dim=1.0, return_estimator=True)
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
        np.testing.assert_allclose(K, self.msm.P, atol=1E-5)

        Tsym = np.diag(self.p0 ** 0.5).dot(self.msm.P).dot(np.diag(self.p1 ** -0.5))
        np.testing.assert_allclose(np.linalg.svd(Tsym)[1][1:], self.vamp.singular_values[0:2], atol=1E-7)

    def test_singular_functions_against_MSM(self):
        Tsym = np.diag(self.p0 ** 0.5).dot(self.msm.P).dot(np.diag(self.p1 ** -0.5))
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
        references_sf = [U.T.dot(np.diag(self.p0)).dot(np.linalg.matrix_power(self.msm.P, k * self.lag)).dot(V).T for k
                         in
                         range(10 - 1)]
        cktest = vamp_cktest(test_estimator=self.estimator, model=self.vamp, n_observables=2, mlags=10, data=self.trajs).fetch_model()
        pred_sf = cktest.predictions
        esti_sf = cktest.estimates
        for e, p, r in zip(esti_sf[1:], pred_sf[1:], references_sf[1:]):
            np.testing.assert_allclose(np.diag(p), np.diag(r), atol=1E-6)
            np.testing.assert_allclose(np.abs(p), np.abs(r), atol=1E-6)

        cumsum_Tsym = np.cumsum(S[1:] ** 2)
        cumsum_Tsym /= cumsum_Tsym[-1]
        np.testing.assert_allclose(self.vamp.cumvar, cumsum_Tsym)

    def test_cumvar_variance_cutoff(self):
        for d in (0.2, 0.5, 0.8, 0.9, 1.0):
            self.vamp.dim = d
            special_cumvar = np.asarray([0] + self.vamp.cumvar.tolist())
            self.assertLessEqual(d, special_cumvar[self.vamp.dimension()], )
            self.assertLessEqual(special_cumvar[self.vamp.dimension() - 1], d)

    def test_CK_expectation_against_MSM(self):
        obs = np.eye(3)  # observe every state
        cktest = vamp_cktest(test_estimator=self.estimator, model=self.vamp, observables=obs, statistics=None, mlags=4,
                             data=self.trajs).fetch_model()
        pred = cktest.predictions[1:]
        est = cktest.estimates[1:]

        for i, (est_, pred_) in enumerate(zip(est, pred)):
            msm = estimate_markov_model(dtrajs=self.dtrajs, lag=self.lag * (i + 1), reversible=False)
            msm_esti = self.p0.T.dot(msm.P).dot(obs)
            msm_pred = self.p0.T.dot(np.linalg.matrix_power(self.msm.P, (i + 1))).dot(obs)
            np.testing.assert_allclose(pred_, msm_pred, atol=self.atol)
            np.testing.assert_allclose(est_, msm_esti, atol=self.atol)
            np.testing.assert_allclose(est_, pred_, atol=0.006)

    def test_CK_covariances_of_singular_functions(self):
        cktest = vamp_cktest(test_estimator=self.estimator, model=self.vamp, n_observables=2, mlags=4,
                             data=self.trajs).fetch_model()

        pred = cktest.predictions[1:]
        est = cktest.estimates[1:]
        error = np.max(np.abs(np.array(pred) - np.array(est))) / max(np.max(pred), np.max(est))
        assert error < 0.05

    def test_CK_covariances_against_MSM(self):
        obs = np.eye(3)  # observe every state
        sta = np.eye(3)  # restrict p0 to every state
        cktest = vamp_cktest(test_estimator=self.estimator, model=self.vamp, observables=obs, statistics=sta,
                             mlags=4, data=self.trajs).fetch_model()
        pred = cktest.predictions[1:]
        est = cktest.estimates[1:]

        for i, (est_, pred_) in enumerate(zip(est, pred)):
            msm = estimate_markov_model(dtrajs=self.dtrajs, lag=self.lag * (i + 1), reversible=False)
            msm_esti = (self.p0 * sta).T.dot(msm.P).dot(obs).T
            msm_pred = (self.p0 * sta).T.dot(np.linalg.matrix_power(self.msm.P, (i + 1))).dot(obs).T
            np.testing.assert_allclose(np.diag(pred_), np.diag(msm_pred), atol=self.atol)
            np.testing.assert_allclose(np.diag(est_), np.diag(msm_esti), atol=self.atol)
            np.testing.assert_allclose(np.diag(est_), np.diag(pred_), atol=0.006)

    def test_self_score_with_MSM(self):
        T = self.msm.P
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

            vamp_train = estimate_vamp(data=trajs_train, lag=self.lag, dim=1.0)
            vamp_test = estimate_vamp(data=trajs_test, lag=self.lag, dim=1.0)
            score_vamp = vamp_train.score(test_model=vamp_test, score_method=m)

            self.assertAlmostEqual(score_msm, score_vamp, places=2 if m == 'VAMPE' else 3, msg=m)

    def test_kinetic_map(self):
        lag = 10
        vamp = estimate_vamp(self.trajs, lag=lag, scaling='km', right=False)
        transformed = [vamp.transform(X)[:-lag] for X in self.trajs]
        std = np.std(np.concatenate(transformed), axis=0)
        np.testing.assert_allclose(std, vamp.singular_values[:vamp.dimension()], atol=1e-4, rtol=1e-4)


class TestVAMPWithEdgeCaseData(unittest.TestCase):
    def test_1D_data(self):
        x = np.random.randn(10, 1)
        vamp = estimate_vamp([x], 1, right=True)  # just test that this doesn't raise
        # Doing VAMP with 1-D data is just centering and normalizing the data.
        assert_allclose_ignore_phase(vamp.transform(x), (x - np.mean(x[1:, 0])) / np.std(x[1:, 0]))

    def test_const_data(self):
        from sktime.numeric.eigen import ZeroRankError
        with self.assertRaises(ZeroRankError):
            estimate_vamp([np.ones((10, 2))], 1)
        with self.assertRaises(ZeroRankError):
            estimate_vamp([np.ones(10)], 1)


if __name__ == "__main__":
    unittest.main()
