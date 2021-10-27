import unittest
import numpy as np
import pkg_resources

from deeptime.covariance import KoopmanWeightingModel
from deeptime.util.data import timeshifted_split
from deeptime.numeric import sort_eigs
import numpy.linalg as scl


def transform_C0(C, epsilon):
    d, V = scl.eigh(C)
    evmin = np.minimum(0, np.min(d))
    ep = np.maximum(-evmin, epsilon)
    d, V = sort_eigs(d, V)
    ind = np.where(np.abs(d) > ep)[0]
    d = d[ind]
    V = V[:, ind]
    V = scale_eigenvectors(V)
    R = np.dot(V, np.diag(d**(-0.5)))
    return R


def scale_eigenvectors(V):
    for j in range(V.shape[1]):
        jj = np.argmax(np.abs(V[:, j]))
        V[:, j] *= np.sign(V[jj, j])
    return V


class TestKoopmanTICA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Basis set definition:
        cls.nf = 10
        cls.chi = np.zeros((20, cls.nf), dtype=float)
        for n in range(cls.nf):
            cls.chi[2 * n:2 * (n + 1), n] = 1.0

        # Load simulations:
        f = np.load(pkg_resources.resource_filename(__name__, "data/test_data_koopman.npz"))
        trajs = [f[key] for key in f.keys()]
        cls.data = [cls.chi[traj, :] for traj in trajs]

        # Lag time:
        cls.tau = 10
        # Truncation for small eigenvalues:
        cls.epsilon = 1e-6

        # Compute the means:
        cls.mean_x = np.zeros(cls.nf)
        cls.mean_y = np.zeros(cls.nf)
        cls.frames = 0
        for traj in cls.data:
            cls.mean_x += np.sum(traj[:-cls.tau, :], axis=0)
            cls.mean_y += np.sum(traj[cls.tau:, :], axis=0)
            cls.frames += traj[:-cls.tau, :].shape[0]
        cls.mean_x *= (1.0 / cls.frames)
        cls.mean_y *= (1.0 / cls.frames)
        cls.mean_rev = 0.5 * (cls.mean_x + cls.mean_y)

        # Compute correlations:
        cls.C0 = np.zeros((cls.nf, cls.nf))
        cls.Ct = np.zeros((cls.nf, cls.nf))
        cls.C0_rev = np.zeros((cls.nf, cls.nf))
        cls.Ct_rev = np.zeros((cls.nf, cls.nf))
        for traj in cls.data:
            itraj = (traj - cls.mean_x[None, :]).copy()
            cls.C0 += np.dot(itraj[:-cls.tau, :].T, itraj[:-cls.tau, :])
            cls.Ct += np.dot(itraj[:-cls.tau, :].T, itraj[cls.tau:, :])
            itraj = (traj - cls.mean_rev[None, :]).copy()
            cls.C0_rev += np.dot(itraj[:-cls.tau, :].T, itraj[:-cls.tau, :]) \
                          + np.dot(itraj[cls.tau:, :].T, itraj[cls.tau:, :])
            cls.Ct_rev += np.dot(itraj[:-cls.tau, :].T, itraj[cls.tau:, :]) \
                          + np.dot(itraj[cls.tau:, :].T, itraj[:-cls.tau, :])
        cls.C0 *= (1.0 / cls.frames)
        cls.Ct *= (1.0 / cls.frames)
        cls.C0_rev *= (1.0 / (2 * cls.frames))
        cls.Ct_rev *= (1.0 / (2 * cls.frames))

        # Compute whitening transformation:
        cls.R = transform_C0(cls.C0, cls.epsilon)
        cls.Rrev = transform_C0(cls.C0_rev, cls.epsilon)

        # Perform non-reversible diagonalization
        cls.ln, cls.Rn = scl.eig(np.dot(cls.R.T, np.dot(cls.Ct, cls.R)))
        cls.ln, cls.Rn = sort_eigs(cls.ln, cls.Rn)
        cls.Rn = np.dot(cls.R, cls.Rn)
        cls.Rn = scale_eigenvectors(cls.Rn)
        cls.tsn = -cls.tau / np.log(np.abs(cls.ln))

        cls.ls, cls.Rs = scl.eig(np.dot(cls.Rrev.T, np.dot(cls.Ct_rev, cls.Rrev)))
        cls.ls, cls.Rs = sort_eigs(cls.ls, cls.Rs)
        cls.Rs = np.dot(cls.Rrev, cls.Rs)
        cls.Rs = scale_eigenvectors(cls.Rs)
        cls.tss = -cls.tau / np.log(np.abs(cls.ls))

        # Compute non-reversible Koopman matrix:
        cls.K = np.dot(cls.R.T, np.dot(cls.Ct, cls.R))
        cls.K = np.vstack((cls.K, np.dot((cls.mean_y - cls.mean_x), cls.R)))
        cls.K = np.hstack((cls.K, np.eye(cls.K.shape[0], 1, k=-cls.K.shape[0] + 1)))
        cls.N1 = cls.K.shape[0]

        # Compute u-vector:
        ln, Un = scl.eig(cls.K.T)
        ln, Un = sort_eigs(ln, Un)
        cls.u = np.real(Un[:, 0])
        v = np.eye(cls.N1, 1, k=-cls.N1 + 1)[:, 0]
        cls.u *= (1.0 / np.dot(cls.u, v))

        # Prepare weight object:
        u_mod = cls.u.copy()
        N = cls.R.shape[0]
        u_input = np.zeros(N + 1)
        u_input[0:N] = cls.R.dot(u_mod[0:-1])  # in input basis
        u_input[N] = u_mod[-1] - cls.mean_x.dot(cls.R.dot(u_mod[0:-1]))
        cls.weight_obj = KoopmanWeightingModel(u=u_input[:-1], u_const=u_input[-1], koopman_operator=cls.K,
                                               whitening_transformation=cls.R, covariances=None)

        # Compute weights over all data points:
        cls.wtraj = []
        for traj in cls.data:
            traj = np.dot((traj - cls.mean_x[None, :]), cls.R).copy()
            traj = np.hstack((traj, np.ones((traj.shape[0], 1))))
            cls.wtraj.append(np.dot(traj, cls.u))

        # Compute equilibrium mean:
        cls.mean_eq = np.zeros(cls.nf)
        q = 0
        for traj in cls.data:
            qwtraj = cls.wtraj[q]
            cls.mean_eq += np.sum((qwtraj[:-cls.tau, None] * traj[:-cls.tau, :]), axis=0) \
                           + np.sum((qwtraj[:-cls.tau, None] * traj[cls.tau:, :]), axis=0)
            q += 1
        cls.mean_eq *= (1.0 / (2 * cls.frames))

        # Compute reversible C0, Ct:
        cls.C0_eq = np.zeros((cls.N1, cls.N1))
        cls.Ct_eq = np.zeros((cls.N1, cls.N1))
        q = 0
        for traj in cls.data:
            qwtraj = cls.wtraj[q]
            traj = (traj - cls.mean_eq[None, :]).copy()
            cls.C0_eq += np.dot((qwtraj[:-cls.tau, None] * traj[:-cls.tau, :]).T, traj[:-cls.tau, :]) \
                         + np.dot((qwtraj[:-cls.tau, None] * traj[cls.tau:, :]).T, traj[cls.tau:, :])
            cls.Ct_eq += np.dot((qwtraj[:-cls.tau, None] * traj[:-cls.tau, :]).T, traj[cls.tau:, :]) \
                         + np.dot((qwtraj[:-cls.tau, None] * traj[cls.tau:, :]).T, traj[:-cls.tau, :])
            q += 1
        cls.C0_eq *= (1.0 / (2 * cls.frames))
        cls.Ct_eq *= (1.0 / (2 * cls.frames))

        # Solve re-weighted eigenvalue problem:
        S = transform_C0(cls.C0_eq, cls.epsilon)
        Ct_S = np.dot(S.T, np.dot(cls.Ct_eq, S))

        # Compute its eigenvalues:
        cls.lr, cls.Rr = scl.eigh(Ct_S)
        cls.lr, cls.Rr = sort_eigs(cls.lr, cls.Rr)
        cls.Rr = np.dot(S, cls.Rr)
        cls.Rr = scale_eigenvectors(cls.Rr)
        cls.tsr = -cls.tau / np.log(np.abs(cls.lr))

        def tica(data, lag, weights=None, **params):
            from deeptime.decomposition import TICA
            return TICA(var_cutoff=0.95, lagtime=lag, **params).fit_from_timeseries(data, weights=weights).fetch_model()

        # Set up the model:
        cls.koop_rev = tica(cls.data, lag=cls.tau, scaling=None)
        cls.koop_eq = tica(cls.data, lag=cls.tau, weights=cls.weight_obj, scaling=None)

    def test_mean_x(self):
        np.testing.assert_allclose(self.koop_rev.mean_0, self.mean_rev)
        np.testing.assert_allclose(self.koop_eq.mean_0, self.mean_eq)

    def test_C0(self):
        np.testing.assert_allclose(self.koop_rev.cov_00, self.C0_rev)
        np.testing.assert_allclose(self.koop_eq.cov_00, self.C0_eq)

    def test_Ct(self):
        np.testing.assert_allclose(self.koop_rev.cov_0t, self.Ct_rev)
        np.testing.assert_allclose(self.koop_eq.cov_0t, self.Ct_eq)

    def test_eigenvalues(self):
        np.testing.assert_allclose(self.koop_rev.singular_values, self.ls)
        np.testing.assert_allclose(self.koop_eq.singular_values, self.lr)
        np.testing.assert_allclose(self.koop_rev.timescales(self.tau), self.tss)
        np.testing.assert_allclose(self.koop_eq.timescales(self.tau), self.tsr)

    def test_eigenvectors(self):
        np.testing.assert_allclose(self.koop_rev.singular_vectors_left, self.Rs)
        np.testing.assert_allclose(self.koop_eq.singular_vectors_left, self.Rr)

    def test_transform(self):
        traj = self.data[0] - self.mean_rev[None, :]
        ev_traj_rev = np.dot(traj, self.Rs)[:, :2]
        out_traj_rev = self.koop_rev.transform(self.data[0], propagate=False)
        traj = self.data[0] - self.mean_eq[None, :]
        ev_traj_eq = np.dot(traj, self.Rr)[:, :2]
        out_traj_eq = self.koop_eq.transform(self.data[0], propagate=False)
        np.testing.assert_allclose(out_traj_rev, ev_traj_rev)
        np.testing.assert_allclose(out_traj_eq, ev_traj_eq)

    def test_koopman_estimator_partial_fit(self):
        from deeptime.covariance import KoopmanWeightingEstimator
        est = KoopmanWeightingEstimator(lagtime=self.tau)
        est.lagtime = 1
        np.testing.assert_equal(est.lagtime, 1)
        est.lagtime = self.tau
        np.testing.assert_equal(est.lagtime, self.tau)

        data_lagged = timeshifted_split(self.data, lagtime=self.tau, n_splits=10)
        for traj in data_lagged:
            est.partial_fit(traj)
        m = est.fetch_model()

        np.testing.assert_allclose(m.weights_input, self.weight_obj.weights_input)
        np.testing.assert_allclose(m.const_weight_input, self.weight_obj.const_weight_input)

        # weights and transform are identical
        np.testing.assert_allclose(m.weights(self.data[0]), m.transform(self.data[0]))
        # dispatches to model
        np.testing.assert_allclose(m.weights(self.data[0]), est.transform(self.data[0]))

    def test_koopman_estimator_fit(self):
        from deeptime.covariance import KoopmanWeightingEstimator
        est = KoopmanWeightingEstimator(lagtime=self.tau)
        est.fit(self.data)
        m = est.fetch_model()

        np.testing.assert_allclose(m.weights_input, self.weight_obj.weights_input)
        np.testing.assert_allclose(m.const_weight_input, self.weight_obj.const_weight_input)
