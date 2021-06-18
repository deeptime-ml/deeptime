"""
@author: paul, clonker
"""

import unittest

import numpy as np
import pytest

from deeptime.covariance import CovarianceModel
from deeptime.util.data import timeshifted_split, TimeLaggedDataset, TimeLaggedConcatDataset, TrajectoryDataset, \
    TrajectoriesDataset
from deeptime.data import ellipsoids
from deeptime.decomposition import TransferOperatorModel, CovarianceKoopmanModel, VAMP, cvsplit_trajs
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
@pytest.mark.parametrize("lag_multiple", [1, 2, 3], ids=lambda x: f"lag_multiple={x}")
def test_expectation_sanity(with_statistics, lag_multiple):
    data = np.random.normal(size=(10000, 5))
    vamp = VAMP(lagtime=1).fit_from_timeseries(data).fetch_model()
    input_dimension = 5
    n_observables = 10

    observations = np.random.normal(size=(input_dimension, n_observables))
    if with_statistics:
        n_statistics = 50
        statistics = np.random.normal(size=(input_dimension, n_statistics))
    else:
        statistics = None
    vamp.expectation(observations, statistics, lag_multiple=lag_multiple)


@pytest.mark.parametrize('components', [None, 0, [0, 1]], ids=lambda x: f"components={x}")
def test_propagate(components):
    K = np.diag(np.array([3., 2., 1.]))
    model = CovarianceKoopmanModel(np.eye(3), K, np.eye(3), CovarianceModel(), 3, 3)
    data = np.random.normal(size=(100, 3))
    fwd = model.propagate(data, components=components)
    if components is not None:
        components = np.atleast_1d(components)
        # update K
        K = K.copy()
        for i in range(len(K)):
            if i not in components:
                K[i, i] = 0.
    if components is None:
        np.testing.assert_array_almost_equal(fwd, data @ K)


@pytest.fixture(params=['trajectory', 'time-lagged-ds', 'concat-time-lagged-ds', 'traj-ds', 'trajs-ds'])
def full_rank_time_series(request):
    """ Yields a time series of which the propagator has full rank (7 in this case as data is mean-free). """
    random_state = np.random.RandomState(42)
    d = 8
    Q = np.linalg.qr(random_state.normal(size=(d, d)))[0]
    K = Q @ (np.diag(np.arange(1, d + 1)).astype(np.float64) / d) @ Q.T
    model = TransferOperatorModel(K)
    x = np.ones((1, d,)) * 100000
    traj = [x]
    for _ in range(1000):
        traj.append(model.forward(traj[-1]))
    traj = np.concatenate(traj)
    if request.param == 'trajectory':
        return traj, traj
    elif request.param == 'time-lagged-ds':
        return traj, TimeLaggedDataset(traj[:-1], traj[1:])
    elif request.param == 'concat-time-lagged-ds':
        return traj, TimeLaggedConcatDataset([TimeLaggedDataset(traj[:-1], traj[1:])])
    elif request.param == 'traj-ds':
        return traj, TrajectoryDataset(1, traj)
    elif request.param == 'trajs-ds':
        return traj, TrajectoriesDataset([TrajectoryDataset(1, traj)])
    else:
        raise ValueError(f"Unexpected request param {request.param}")


@pytest.mark.parametrize("dim", [0, 1, 2, 3, 4, 5, 6, 7], ids=lambda x: f"dim={x}")
def test_dim(full_rank_time_series, dim):
    traj, ds = full_rank_time_series
    if dim < 1:
        with np.testing.assert_raises(ValueError):
            VAMP(lagtime=1, dim=dim).fit(ds).fetch_model()
    else:
        est = VAMP(lagtime=1, dim=dim).fit(ds)
        projection = est.transform(traj)
        np.testing.assert_equal(projection.shape, (len(traj), dim))


@pytest.mark.parametrize("var_cutoff", [0., .5, 1., 1.1], ids=lambda x: f"var_cutoff={x}")
def test_var_cutoff(full_rank_time_series, var_cutoff):
    traj, ds = full_rank_time_series
    if 0 < var_cutoff <= 1:
        est = VAMP(lagtime=1, var_cutoff=var_cutoff).fit(ds)
        projection = est.transform(traj)
        np.testing.assert_equal(projection.shape[0], traj.shape[0])
        if var_cutoff == 1.:
            # data is internally mean-free
            np.testing.assert_equal(projection.shape[1], traj.shape[1] - 1)
        else:
            np.testing.assert_array_less(projection.shape[1], traj.shape[1])
    else:
        with np.testing.assert_raises(ValueError):
            VAMP(lagtime=1, var_cutoff=var_cutoff).fit(ds).fetch_model()


@pytest.mark.parametrize("dim", [None, 2, 3], ids=lambda x: f"dim={x}")
@pytest.mark.parametrize("var_cutoff", [None, .5, 1.], ids=lambda x: f"var_cutoff={x}")
@pytest.mark.parametrize("partial_fit", [False, True], ids=lambda x: f"partial_fit={x}")
def test_dim_and_var_cutoff(full_rank_time_series, dim, var_cutoff, partial_fit):
    traj, ds = full_rank_time_series
    # basically dim should be ignored here since var_cutoff takes precedence if it is None
    est = VAMP(lagtime=1, dim=dim, var_cutoff=var_cutoff)
    if partial_fit:
        for chunk in timeshifted_split(traj, lagtime=1, chunksize=15):
            est.partial_fit(chunk)
        est2 = VAMP(lagtime=1, dim=dim, var_cutoff=var_cutoff).fit(ds)
        np.testing.assert_array_almost_equal(est.fetch_model().operator,
                                             est2.fetch_model().operator, decimal=4)  # can fail on M$ with higher acc.
    else:
        est.fit(ds)
    projection = est.transform(traj)
    np.testing.assert_equal(projection.shape[0], traj.shape[0])
    if var_cutoff is not None:
        if var_cutoff == 1.:
            # data is internally mean-free
            np.testing.assert_equal(projection.shape[1], traj.shape[1] - 1)
        else:
            np.testing.assert_array_less(projection.shape[1], traj.shape[1])
    else:
        if dim is None:
            # data is internally mean-free
            np.testing.assert_equal(projection.shape[1], traj.shape[1] - 1)
        else:
            np.testing.assert_equal(projection.shape[1], dim)


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
        vamp = VAMP(scaling=None, lagtime=tau).fit(trajs).fetch_model()

        assert vamp.output_dimension <= rank

        atol = np.finfo(np.float32).eps * 10.0
        rtol = np.finfo(np.float32).resolution
        phi_trajs = [vamp.backward(X, propagate=False)[tau:, :] for X in trajs]
        phi = np.concatenate(phi_trajs)
        mean_right = phi.sum(axis=0) / phi.shape[0]
        cov_right = phi.T.dot(phi) / phi.shape[0]
        np.testing.assert_allclose(mean_right, 0.0, rtol=rtol, atol=atol)
        np.testing.assert_allclose(cov_right, np.eye(vamp.output_dimension), rtol=rtol, atol=atol)

        vamp.right = False
        # vamp = estimate_vamp(trajs, lag=tau, scaling=None, right=False)
        psi_trajs = [vamp.forward(X, propagate=False)[0:-tau, :] for X in trajs]
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
        np.testing.assert_allclose(C01_psi_phi, np.diag(vamp.singular_values[0:vamp.output_dimension]), rtol=rtol,
                                   atol=atol)

        if test_partial_fit:
            vamp2 = VAMP(lagtime=tau).fit(trajs).fetch_model()

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
                assert_allclose_ignore_phase(vamp2.backward(t[tau:], propagate=False), ref, rtol=rtol, atol=atol)

            for t, ref in zip(trajs, psi_trajs):
                assert_allclose_ignore_phase(vamp2.transform(t[0:-tau], propagate=False), ref, rtol=rtol, atol=atol)


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


def test_cktest():
    traj = ellipsoids().observations(n_steps=10000)
    estimator = VAMP(1, dim=1).fit(traj)
    validator = estimator.chapman_kolmogorov_validator(4)
    cktest = validator.fit(traj).fetch_model()
    np.testing.assert_almost_equal(cktest.predictions, cktest.estimates, decimal=1)


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
        estimator = VAMP(scaling=None, var_cutoff=1.0)
        cov = VAMP.covariance_estimator(lagtime=lag).fit(trajs).fetch_model()
        vamp = estimator.fit(cov).fetch_model()
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
        phi = self.vamp.backward(np.eye(3), propagate=False)
        psi = self.vamp.forward(np.eye(3), propagate=False)
        assert_allclose_ignore_phase(U, psi, atol=1E-5)
        assert_allclose_ignore_phase(V, phi, atol=1E-5)

        cumsum_Tsym = np.cumsum(S[1:] ** 2)
        cumsum_Tsym /= cumsum_Tsym[-1]
        np.testing.assert_allclose(self.vamp.cumulative_kinetic_variance, cumsum_Tsym)

    def test_cumvar_variance_cutoff(self):
        for d in (0.2, 0.5, 0.8, 0.9, 1.0):
            self.vamp.var_cutoff = d
            special_cumvar = np.asarray([0] + self.vamp.cumulative_kinetic_variance.tolist())
            self.assertLessEqual(d, special_cumvar[self.vamp.output_dimension], )
            self.assertLessEqual(special_cumvar[self.vamp.output_dimension - 1], d)

    def test_self_score_with_MSM(self):
        T = self.msm.transition_matrix
        Tadj = np.diag(1. / self.p1).dot(T.T).dot(np.diag(self.p0))
        NFro = np.trace(T.dot(Tadj))
        s2 = self.vamp.score(2)
        np.testing.assert_allclose(s2, NFro)

        Tsym = np.diag(self.p0 ** 0.5).dot(T).dot(np.diag(self.p1 ** -0.5))
        Nnuc = np.linalg.norm(Tsym, ord='nuc')
        s1 = self.vamp.score(1)
        np.testing.assert_allclose(s1, Nnuc)

        sE = self.vamp.score("E")
        np.testing.assert_allclose(sE, NFro)  # see paper appendix H.2

    def test_score_vs_MSM(self):
        trajs_test, trajs_train = cvsplit_trajs(self.trajs, random_state=32)
        dtrajs_test, dtrajs_train = cvsplit_trajs(self.dtrajs, random_state=32)

        methods = (1, 2, 'E')

        for m in methods:
            msm_train = estimate_markov_model(dtrajs=dtrajs_train, lag=self.lag, reversible=False)
            score_msm = msm_train.score(dtrajs_test, m, dim=None)
            vamp_train = VAMP(lagtime=self.lag, var_cutoff=1.0).fit_from_timeseries(trajs_train).fetch_model()
            vamp_test = VAMP(lagtime=self.lag, var_cutoff=1.0).fit_from_timeseries(trajs_test).fetch_model()
            score_vamp = vamp_train.score(test_model=vamp_test, r=m)

            self.assertAlmostEqual(score_msm, score_vamp, places=2 if m == 'E' else 3, msg=m)

    def test_kinetic_map(self):
        lag = 10
        vamp = VAMP(lagtime=lag, scaling='km').fit_from_timeseries(self.trajs).fetch_model()
        transformed = [vamp.transform(X)[:-lag] for X in self.trajs]
        std = np.std(np.concatenate(transformed), axis=0)
        np.testing.assert_allclose(std, vamp.singular_values[:vamp.output_dimension], atol=1e-4, rtol=1e-4)


class TestVAMPWithEdgeCaseData(unittest.TestCase):
    def test_1D_data(self):
        x = np.random.randn(10, 1)
        vamp = VAMP(lagtime=1).fit([x]).fetch_model()
        # Doing VAMP with 1-D data is just centering and normalizing the data.
        assert_allclose_ignore_phase(vamp.backward(x, propagate=False), (x - np.mean(x[1:, 0])) / np.std(x[1:, 0]))

    def test_const_data(self):
        from deeptime.numeric import ZeroRankError
        with self.assertRaises(ZeroRankError):
            print(VAMP(lagtime=1).fit(np.ones((10, 2))).fetch_model().singular_values)
        with self.assertRaises(ZeroRankError):
            print(VAMP(lagtime=1).fit(np.ones((10, 1))).fetch_model().singular_values)
