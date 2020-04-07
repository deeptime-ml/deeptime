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


r"""Unit test for the OOM-based MSM estimation.

"""

import unittest
import warnings

import msmtools.analysis as ma
import msmtools.estimation as msmest
import numpy as np
import pkg_resources
import pytest
import scipy.linalg as scl
import scipy.sparse
from sktime.markov import score_cv
from sktime.markov.msm import MarkovStateModel
from sktime.markov.msm.koopman_reweighted_msm import OOMReweightedMSM
from sktime.numeric import sort_by_norm


def oom_transformations(Ct, C2t, rank):
    # Number of states:
    N = Ct.shape[0]
    # Get the SVD of Ctau:
    U, s, V = scl.svd(Ct, full_matrices=False)
    # Reduce:
    s = s[:rank]
    U = U[:, :rank]
    V = V[:rank, :].transpose()
    # Define transformation matrices:
    F1 = np.dot(U, np.diag(s ** (-0.5)))
    F2 = np.dot(V, np.diag(s ** (-0.5)))
    # Compute observable operators:
    Xi = np.zeros((rank, N, rank))
    for n in range(N):
        Xi[:, n, :] = np.dot(F1.T, np.dot(C2t[:, :, n], F2))
    Xi_full = np.sum(Xi, axis=1)
    # Compute evaluator:
    c = np.sum(Ct, axis=1)
    sigma = np.dot(F1.T, c)
    # Compute information state:
    l, R = scl.eig(Xi_full.T)
    # Restrict eigenvalues to reasonable range:
    ind = np.where(np.logical_and(np.abs(l) <= (1 + 1e-2), np.real(l) >= 0.0))[0]
    l = l[ind]
    R = R[:, ind]
    l, R = sort_by_norm(l, R)
    omega = np.real(R[:, 0])
    omega = omega / np.dot(omega, sigma)

    return Xi, omega, sigma, l


def TransitionMatrix(Xi, omega, sigma, reversible=True):
    N = Xi.shape[1]
    Ct_Eq = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            Ct_Eq[i, j] = np.dot(omega.T, np.dot(Xi[:, i, :], np.dot(Xi[:, j, :], sigma)))
    Ct_Eq[Ct_Eq < 0.0] = 0.0
    pi_r = np.sum(Ct_Eq, axis=1)
    if reversible:
        pi_c = np.sum(Ct_Eq, axis=0)
        pi = pi_r + pi_c
        Tt_Eq = (Ct_Eq + Ct_Eq.T) / pi[:, None]
    else:
        Tt_Eq = Ct_Eq / pi_r[:, None]

    return Tt_Eq


class TestMSMFiveState(object):

    @classmethod
    def setup_class(cls):
        # Load the data:
        data = np.load(pkg_resources.resource_filename('pyemma.msm.tests', "data/TestData_OOM_MSM.npz"))
        cls.dtrajs = [data['arr_%d' % k] for k in range(1000)]

        # Number of states:
        cls.N = 5
        # Lag time:
        cls.tau = 5
        # Rank:
        cls.rank = 3
        # Build models:
        cls.msmrev = OOMReweightedMSM(lagtime=cls.tau, rank_mode='bootstrap_trajs').fit(cls.dtrajs)
        cls.msm = OOMReweightedMSM(lagtime=cls.tau, reversible=False, rank_mode='bootstrap_trajs').fit(cls.dtrajs)
        cls.msmrev_sparse = OOMReweightedMSM(lagtime=cls.tau, sparse=True, rank_mode='bootstrap_trajs').fit(cls.dtrajs)
        cls.msm_sparse = OOMReweightedMSM(lagtime=cls.tau, reversible=False, sparse=True, rank_mode='bootstrap_trajs') \
            .fit(cls.dtrajs)
        cls.msms = [cls.msmrev, cls.msm, cls.msmrev_sparse, cls.msm_sparse]

        # Reference count matrices at lag time tau and 2*tau:
        cls.C2t = data['C2t']
        cls.Ct = np.sum(cls.C2t, axis=1)

        # Compute OOM-components:
        cls.Xi, cls.omega, cls.sigma, cls.l = oom_transformations(cls.Ct, cls.C2t, cls.rank)
        # Compute corrected transition matrix:
        Tt_rev = TransitionMatrix(cls.Xi, cls.omega, cls.sigma, reversible=True)
        Tt = TransitionMatrix(cls.Xi, cls.omega, cls.sigma, reversible=False)

        # Build reference models:
        cls.rmsmrev = MarkovStateModel(Tt_rev)
        cls.rmsm = MarkovStateModel(Tt)

        " Compute further reference quantities:"
        # Commitor and MFPT:
        a = np.array([0, 1])
        b = np.array([4])
        cls.comm_forward = cls.rmsm.committor_forward(a, b)
        cls.comm_forward_rev = cls.rmsmrev.committor_forward(a, b)
        cls.comm_backward = cls.rmsm.committor_backward(a, b)
        cls.comm_backward_rev = cls.rmsmrev.committor_backward(a, b)
        cls.mfpt = cls.tau * cls.rmsm.mfpt(a, b)
        cls.mfpt_rev = cls.tau * cls.rmsmrev.mfpt(a, b)
        # PCCA:
        pcca = cls.rmsmrev.pcca(3)
        cls.pcca_ass = pcca.assignments
        cls.pcca_dist = pcca.metastable_distributions
        cls.pcca_mem = pcca.memberships
        cls.pcca_sets = pcca.sets
        # Experimental quantities:
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([1, -1, 0, -2, 4])
        p0 = np.array([0.5, 0.2, 0.2, 0.1, 0.0])
        pi = cls.rmsm.stationary_distribution
        pi_rev = cls.rmsmrev.stationary_distribution
        _, _, L_rev = ma.rdl_decomposition(Tt_rev)
        cls.exp = np.dot(cls.rmsm.stationary_distribution, a)
        cls.exp_rev = np.dot(cls.rmsmrev.stationary_distribution, a)
        cls.corr_rev = np.zeros(10)
        cls.rel = np.zeros(10)
        cls.rel_rev = np.zeros(10)
        for k in range(10):
            Ck_rev = np.dot(np.diag(pi_rev), np.linalg.matrix_power(Tt_rev, k))
            cls.corr_rev[k] = np.dot(a.T, np.dot(Ck_rev, b))
            cls.rel[k] = np.dot(p0.T, np.dot(np.linalg.matrix_power(Tt, k), a))
            cls.rel_rev[k] = np.dot(p0.T, np.dot(np.linalg.matrix_power(Tt_rev, k), a))
        cls.fing_cor = np.dot(a.T, L_rev.T) * np.dot(b.T, L_rev.T)
        cls.fing_rel = np.dot(a.T, L_rev.T) * np.dot((p0 / pi_rev).T, L_rev.T)

    # ---------------------------------
    # SCORE
    # ---------------------------------

    def test_score(self):
        for msm in self.msms:
            dtrajs_test = self.dtrajs[0:500]
            s1 = msm.fetch_model().score(dtrajs_test, score_method='VAMP1', score_k=2)
            assert 1.0 <= s1 <= 2.0
            s2 = msm.fetch_model().score(dtrajs_test, score_method='VAMP2', score_k=2)
            assert 1.0 <= s2 <= 2.0
            # se = msm.score(dtrajs_test, score_method='VAMPE', score_k=2)
            # se_inf = msm.score(dtrajs_test, score_method='VAMPE', score_k=None)

    @pytest.mark.parametrize("reversible,sparse", [(True, True), (True, False), (False, True), (False, False)])
    def test_score_cv(self, reversible, sparse):
        msm = OOMReweightedMSM(lagtime=5, reversible=reversible, sparse=sparse)
        s1 = score_cv(msm, self.dtrajs[:500], lagtime=1, n=2, score_method='VAMP1', score_k=2,
                      fit_fetch=lambda dtrajs: msm.fit(dtrajs).fetch_model()).mean()
        np.testing.assert_(1.0 <= s1 <= 2.0)
        s2 = score_cv(msm, self.dtrajs[:500], lagtime=1, n=2, score_method='VAMP2', score_k=2,
                      fit_fetch=lambda dtrajs: msm.fit(dtrajs).fetch_model()).mean()
        np.testing.assert_(1.0 <= s2 <= 2.0)
        # se = estimator.score_cv(cls.dtrajs[:500], n=2, score_method='VAMPE', score_k=2).mean()
        # se_inf = estimator.score_cv(cls.dtrajs[:500], n=2, score_method='VAMPE', score_k=None).mean()

    # ---------------------------------
    # BASIC PROPERTIES
    # ---------------------------------

    def test_reversible(self):
        # Reversible
        assert self.msmrev.is_reversible
        assert self.msmrev_sparse.is_reversible
        assert self.msmrev_eff.is_reversible
        # Non-reversible
        assert not self.msm.is_reversible
        assert not self.msm_sparse.is_reversible

    def test_sparse(self):
        np.testing.assert_(not self.msmrev_model.sparse)
        np.testing.assert_(not self.msm_model.sparse)

    def _lagtime(self, msm):
        assert (msm.lagtime == self.tau)

    def test_lagtime(self):
        for estimator in [self.msm, self.msmrev, self.msmrev_sparse, self.msm_sparse, self.msmrev_eff]:
            np.testing.assert_equal(estimator.lagtime, self.tau)
            np.testing.assert_equal(estimator.fetch_model().lagtime, self.tau)

    def test_active_set(self):
        assert np.all(self.msmrev.active_set == np.arange(self.N, dtype=int))
        assert np.all(self.msmrev_sparse.active_set == np.arange(self.N, dtype=int))
        assert np.all(self.msm.active_set == np.arange(self.N, dtype=int))
        assert np.all(self.msm_sparse.active_set == np.arange(self.N, dtype=int))
        assert np.all(self.msmrev_eff.active_set == np.arange(self.N, dtype=int))

    def test_largest_connected_set(self):
        assert np.all(self.msmrev.largest_connected_set == np.arange(self.N, dtype=int))
        assert np.all(self.msmrev_sparse.largest_connected_set == np.arange(self.N, dtype=int))
        assert np.all(self.msm.largest_connected_set == np.arange(self.N, dtype=int))
        assert np.all(self.msm_sparse.largest_connected_set == np.arange(self.N, dtype=int))
        assert np.all(self.msmrev_eff.largest_connected_set == np.arange(self.N, dtype=int))

    def _nstates(self, msm):
        # should always be <= full
        assert (msm.nstates <= msm.nstates_full)
        # THIS DATASET:
        assert (msm.nstates == 5)

    def test_nstates(self):
        self._nstates(self.msmrev)
        self._nstates(self.msm)
        self._nstates(self.msmrev_sparse)
        self._nstates(self.msm_sparse)
        self._nstates(self.msmrev_eff)

    def _connected_sets(self, msm):
        cs = msm.connected_sets
        assert (len(cs) >= 1)
        # MODE LARGEST:
        assert (np.all(cs[0] == msm.active_set))

    def test_connected_sets(self):
        self._connected_sets(self.msmrev)
        self._connected_sets(self.msm)
        self._connected_sets(self.msmrev_sparse)
        self._connected_sets(self.msm_sparse)
        self._connected_sets(self.msmrev_eff)

    def _connectivity(self, msm):
        # HERE:
        assert (msm.connectivity == 'largest')

    def test_connectivity(self):
        self._connectivity(self.msmrev)
        self._connectivity(self.msm)
        self._connectivity(self.msmrev_sparse)
        self._connectivity(self.msm_sparse)
        self._connectivity(self.msmrev_eff)

    def _count_matrix_active(self, msm, sparse=False):
        if sparse:
            C = msm.count_matrix_active.toarray()
        else:
            C = msm.count_matrix_active
        assert np.allclose(C, self.Ct)

    def test_count_matrix_active(self):
        self._count_matrix_active(self.msmrev)
        self._count_matrix_active(self.msm)
        self._count_matrix_active(self.msmrev_sparse, sparse=True)
        self._count_matrix_active(self.msm_sparse, sparse=True)
        self._count_matrix_active(self.msmrev_eff)

    def _count_matrix_full(self, msm, sparse=False):
        if sparse:
            C = msm.count_matrix_full.toarray()
        else:
            C = msm.count_matrix_full
        assert np.allclose(C, self.Ct)

    def test_count_matrix_full(self):
        self._count_matrix_full(self.msmrev)
        self._count_matrix_full(self.msm)
        self._count_matrix_full(self.msmrev_sparse, sparse=True)
        self._count_matrix_full(self.msm_sparse, sparse=True)
        self._count_matrix_full(self.msmrev_eff)

    def _discrete_trajectories_full(self, msm):
        assert (np.all(self.dtrajs[0] == msm.discrete_trajectories_full[0]))
        assert len(self.dtrajs) == len(msm.discrete_trajectories_full)

    def test_discrete_trajectories_full(self):
        self._discrete_trajectories_full(self.msmrev)
        self._discrete_trajectories_full(self.msm)
        self._discrete_trajectories_full(self.msmrev_sparse)
        self._discrete_trajectories_full(self.msm_sparse)
        self._discrete_trajectories_full(self.msmrev_eff)

    def _discrete_trajectories_active(self, msm):
        assert (np.all(self.dtrajs[0] == msm.discrete_trajectories_active[0]))
        assert len(self.dtrajs) == len(msm.discrete_trajectories_active)

    def test_discrete_trajectories_active(self):
        self._discrete_trajectories_active(self.msmrev)
        self._discrete_trajectories_active(self.msm)
        self._discrete_trajectories_active(self.msmrev_sparse)
        self._discrete_trajectories_active(self.msm_sparse)
        self._discrete_trajectories_active(self.msmrev_eff)

    def _timestep(self, msm):
        assert (msm.timestep_model.startswith('5'))
        assert (msm.timestep_model.endswith('step'))

    def test_timestep(self):
        self._timestep(self.msmrev)
        self._timestep(self.msm)
        self._timestep(self.msmrev_sparse)
        self._timestep(self.msm_sparse)
        self._timestep(self.msmrev_eff)

    def _transition_matrix(self, msm):
        P = msm.transition_matrix
        # should be ndarray by default
        assert (isinstance(P, np.ndarray) or isinstance(P, scipy.sparse.csr_matrix))
        # shape
        assert (np.all(P.shape == (msm.nstates, msm.nstates)))
        # test transition matrix properties
        import msmtools.analysis as msmana
        assert (msmana.is_transition_matrix(P))
        assert (msmana.is_connected(P))
        # REVERSIBLE
        if msm.is_reversible:
            assert (msmana.is_reversible(P))
        # Test equality with model:
        if isinstance(P, scipy.sparse.csr_matrix):
            P = P.toarray()
        if msm.is_reversible:
            assert np.allclose(P, self.rmsmrev.transition_matrix)
        else:
            assert np.allclose(P, self.rmsm.transition_matrix)

    def test_transition_matrix(self):
        self._transition_matrix(self.msmrev)
        self._transition_matrix(self.msm)
        self._transition_matrix(self.msmrev_sparse)
        self._transition_matrix(self.msm_sparse)
        self._transition_matrix(self.msmrev_eff)

    # ---------------------------------
    # SIMPLE STATISTICS
    # ---------------------------------

    def _active_state_fraction(self, msm):
        # should always be a fraction
        assert (0.0 <= msm.active_state_fraction <= 1.0)
        # For this data set:
        assert (msm.active_state_fraction == 1.0)

    def test_active_state_fraction(self):
        # should always be a fraction
        self._active_state_fraction(self.msmrev)
        self._active_state_fraction(self.msm)
        self._active_state_fraction(self.msmrev_sparse)
        self._active_state_fraction(self.msm_sparse)
        self._active_state_fraction(self.msmrev_eff)

    def _active_count_fraction(self, msm):
        # should always be a fraction
        assert (0.0 <= msm.active_count_fraction <= 1.0)
        # special case for this data set:
        assert (msm.active_count_fraction == 1.0)

    def test_active_count_fraction(self):
        self._active_count_fraction(self.msmrev)
        self._active_count_fraction(self.msm)
        self._active_count_fraction(self.msmrev_sparse)
        self._active_count_fraction(self.msm_sparse)
        self._active_count_fraction(self.msmrev_eff)

    # ---------------------------------
    # EIGENVALUES, EIGENVECTORS
    # ---------------------------------

    def _statdist(self, msm):
        mu = msm.stationary_distribution
        # should strictly positive (irreversibility)
        assert (np.all(mu > 0))
        # should sum to one
        assert (np.abs(np.sum(mu) - 1.0) < 1e-10)
        # Should match model:
        if msm.is_reversible:
            assert np.allclose(mu, self.rmsmrev.stationary_distribution)
        else:
            assert np.allclose(mu, self.rmsm.stationary_distribution)

    def test_statdist(self):
        self._statdist(self.msmrev)
        self._statdist(self.msm)
        self._statdist(self.msmrev_sparse)
        self._statdist(self.msm_sparse)
        self._statdist(self.msmrev_eff)

    def _eigenvalues(self, msm):
        ev = msm.eigenvalues()
        # stochasticity
        assert (np.max(np.abs(ev)) <= 1 + 1e-12)
        # irreducible
        assert (np.max(np.abs(ev[1:])) < 1)
        # ordered?
        evabs = np.abs(ev)
        for i in range(0, len(evabs) - 1):
            assert (evabs[i] >= evabs[i + 1])
        # REVERSIBLE:
        if msm.is_reversible:
            assert (np.all(np.isreal(ev)))

    def test_eigenvalues(self):
        self._eigenvalues(self.msmrev)
        self._eigenvalues(self.msm)
        self._eigenvalues(self.msmrev_sparse)
        self._eigenvalues(self.msm_sparse)
        self._eigenvalues(self.msmrev_eff)

    def _eigenvectors_left(self, msm):
        L = msm.eigenvectors_left()
        k = msm.nstates
        # shape should be right
        assert (np.all(L.shape == (k, msm.nstates)))
        # first one should be identical to stat.dist
        l1 = L[0, :]
        err = msm.stationary_distribution - l1
        assert (np.max(np.abs(err)) < 1e-10)
        # sums should be 1, 0, 0, ...
        assert (np.allclose(np.sum(L[1:, :], axis=1), np.zeros(k - 1)))
        # REVERSIBLE:
        if msm.is_reversible:
            assert (np.all(np.isreal(L)))

    def test_eigenvectors_left(self):
        self._eigenvectors_left(self.msmrev)
        self._eigenvectors_left(self.msm)
        self._eigenvectors_left(self.msmrev_sparse)
        self._eigenvectors_left(self.msm_sparse)
        self._eigenvectors_left(self.msmrev_eff)

    def _eigenvectors_right(self, msm):
        R = msm.eigenvectors_right()
        k = msm.nstates
        # shape should be right
        assert (np.all(R.shape == (msm.nstates, k)))
        # should be all ones
        r1 = R[:, 0]
        assert (np.allclose(r1, np.ones(msm.nstates)))
        # REVERSIBLE:
        if msm.is_reversible:
            assert (np.all(np.isreal(R)))

    def test_eigenvectors_right(self):
        self._eigenvectors_right(self.msmrev)
        self._eigenvectors_right(self.msm)
        self._eigenvectors_right(self.msmrev_sparse)
        self._eigenvectors_right(self.msm_sparse)
        self._eigenvectors_right(self.msmrev_eff)

    def _eigenvectors_RDL(self, msm):
        R = msm.eigenvectors_right()
        D = np.diag(msm.eigenvalues())
        L = msm.eigenvectors_left()
        # orthogonality constraint
        assert (np.allclose(np.dot(R, L), np.eye(msm.nstates)))
        # REVERSIBLE: also true for LR because reversible matrix
        if msm.is_reversible:
            assert (np.allclose(np.dot(L, R), np.eye(msm.nstates)))
        # recover transition matrix
        assert (np.allclose(np.dot(R, np.dot(D, L)), msm.transition_matrix))

    def test_eigenvectors_RDL(self):
        self._eigenvectors_RDL(self.msmrev)
        self._eigenvectors_RDL(self.msm)
        self._eigenvectors_RDL(self.msmrev_sparse)
        self._eigenvectors_RDL(self.msm_sparse)
        self._eigenvectors_RDL(self.msmrev_eff)

    def _timescales(self, msm):
        if not msm.is_reversible:
            with warnings.catch_warnings(record=True) as w:
                ts = msm.timescales()
        else:
            ts = msm.timescales()

        # should be all positive
        assert (np.all(ts > 0))
        # REVERSIBLE: should be all real
        if msm.is_reversible:
            ts_ref = self.rmsmrev.timescales()
            assert (np.all(np.isreal(ts)))
            # HERE:
            np.testing.assert_almost_equal(ts, self.tau * ts_ref, decimal=2)
        else:
            ts_ref = self.rmsm.timescales()
            # HERE:
            np.testing.assert_almost_equal(ts, self.tau * ts_ref, decimal=2)

    def test_timescales(self):
        self._timescales(self.msmrev)
        self._timescales(self.msm)
        self._timescales(self.msmrev_sparse)
        self._timescales(self.msm_sparse)
        self._timescales(self.msmrev_eff)

    def _eigenvalues_OOM(self, msm):
        assert np.allclose(msm.eigenvalues_oom, self.l)

    def test_eigenvalues_OOM(self):
        self._eigenvalues_OOM(self.msmrev)
        self._eigenvalues_OOM(self.msm)
        self._eigenvalues_OOM(self.msmrev_sparse)
        self._eigenvalues_OOM(self.msm_sparse)
        self._eigenvalues_OOM(self.msmrev_eff)

    def _oom_components(self, msm):
        Xi = msm.oom_components
        omega = msm.oom_omega
        sigma = msm.oom_sigma
        assert np.allclose(Xi, self.Xi)
        assert np.allclose(omega, self.omega)
        assert np.allclose(sigma, self.sigma)

    def test_oom_components(self):
        self._oom_components(self.msmrev)
        self._oom_components(self.msm)
        self._oom_components(self.msmrev_sparse)
        self._oom_components(self.msm_sparse)
        self._oom_components(self.msmrev_eff)

    # ---------------------------------
    # FIRST PASSAGE PROBLEMS
    # ---------------------------------

    def _committor(self, msm):
        a = np.array([0, 1])
        b = np.array([4])
        q_forward = msm.committor_forward(a, b)
        if msm.is_reversible:
            assert np.allclose(q_forward, self.comm_forward_rev)
        else:
            assert np.allclose(q_forward, self.comm_forward)
        q_backward = msm.committor_backward(a, b)
        if msm.is_reversible:
            assert np.allclose(q_backward, self.comm_backward_rev)
        else:
            assert np.allclose(q_backward, self.comm_backward)
        # REVERSIBLE:
        if msm.is_reversible:
            assert (np.allclose(q_forward + q_backward, np.ones(msm.nstates)))

    def test_committor(self):
        self._committor(self.msmrev)
        self._committor(self.msm)
        self._committor(self.msmrev_sparse)
        self._committor(self.msm_sparse)
        self._committor(self.msmrev_eff)

    def _mfpt(self, msm):
        a = np.array([0, 1])
        b = np.array([4])
        t = msm.mfpt(a, b)
        assert (t > 0)
        # HERE:
        if msm.is_reversible:
            np.testing.assert_allclose(t, self.mfpt_rev, rtol=1e-3, atol=1e-6)
        else:
            np.testing.assert_allclose(t, self.mfpt, rtol=1e-3, atol=1e-6)

    def test_mfpt(self):
        self._mfpt(self.msmrev)
        self._mfpt(self.msm)
        self._mfpt(self.msmrev_sparse)
        self._mfpt(self.msm_sparse)
        self._mfpt(self.msmrev_eff)

    # ---------------------------------
    # PCCA
    # ---------------------------------

    def _pcca_assignment(self, msm):
        if msm.is_reversible:
            msm.pcca(3)
            ass = msm.metastable_assignments
            # test: number of states
            assert (len(ass) == msm.nstates)
            # should be equal (zero variance) within metastable sets
            assert np.all(ass == self.pcca_ass)
        else:
            with self.assertRaises(ValueError):
                msm.pcca(3)

    def test_pcca_assignment(self):
        self._pcca_assignment(self.msmrev)
        self._pcca_assignment(self.msm)
        with warnings.catch_warnings(record=True) as w:
            self._pcca_assignment(self.msmrev_sparse)
        with warnings.catch_warnings(record=True) as w:
            self._pcca_assignment(self.msm_sparse)
        self._pcca_assignment(self.msmrev_eff)

    def _pcca_distributions(self, msm):
        if msm.is_reversible:
            msm.pcca(3)
            pccadist = msm.metastable_distributions
            # should be right size
            assert (np.all(pccadist.shape == (3, msm.nstates)))
            # should be nonnegative
            assert (np.all(pccadist >= 0))
            # check equality:
            assert np.allclose(pccadist, self.pcca_dist)
        else:
            with self.assertRaises(ValueError):
                msm.pcca(3)

    def test_pcca_distributions(self):
        self._pcca_distributions(self.msmrev)
        self._pcca_distributions(self.msm)
        self._pcca_distributions(self.msmrev_sparse)
        self._pcca_distributions(self.msm_sparse)
        self._pcca_distributions(self.msmrev_eff)

    def _pcca_memberships(self, msm):
        if msm.is_reversible:
            msm.pcca(3)
            M = msm.metastable_memberships
            # should be right size
            assert (np.all(M.shape == (msm.nstates, 3)))
            # should be nonnegative
            assert (np.all(M >= 0))
            # should add up to one:
            assert (np.allclose(np.sum(M, axis=1), np.ones(msm.nstates)))
            # check equality:
            assert np.allclose(M, self.pcca_mem)
        else:
            with self.assertRaises(ValueError):
                msm.pcca(3)

    def test_pcca_memberships(self):
        self._pcca_memberships(self.msmrev)
        self._pcca_memberships(self.msm)
        self._pcca_memberships(self.msmrev_sparse)
        self._pcca_memberships(self.msm_sparse)
        self._pcca_memberships(self.msmrev_eff)

    def _pcca_sets(self, msm):
        if msm.is_reversible:
            msm.pcca(3)
            S = msm.metastable_sets
            assignment = msm.metastable_assignments
            # should coincide with assignment
            for i, s in enumerate(S):
                for j in range(len(s)):
                    assert (assignment[s[j]] == i)
        else:
            with self.assertRaises(ValueError):
                msm.pcca(3)

    def test_pcca_sets(self):
        self._pcca_sets(self.msmrev)
        self._pcca_sets(self.msm)
        self._pcca_sets(self.msmrev_sparse)
        self._pcca_sets(self.msm_sparse)
        self._pcca_sets(self.msmrev_eff)

    # ---------------------------------
    # EXPERIMENTAL STUFF
    # ---------------------------------

    def _expectation(self, msm):
        a = np.array([1, 2, 3, 4, 5])
        e = msm.expectation(a)
        # approximately equal for both
        if msm.is_reversible:
            assert np.allclose(e, self.exp_rev)
        else:
            assert np.allclose(e, self.exp)

    def test_expectation(self):
        self._expectation(self.msmrev)
        self._expectation(self.msm)
        self._expectation(self.msmrev_sparse)
        self._expectation(self.msm_sparse)
        self._expectation(self.msmrev_eff)

    def _correlation(self, msm):
        a = [1, 2, 3, 4, 5]
        b = [1, -1, 0, -2, 4]
        with self.assertRaises(AssertionError):
            msm.correlation(a, 1)
        # test equality:
        _, cor = msm.correlation(a, b, maxtime=50)
        if msm.is_reversible:
            assert np.allclose(cor, self.corr_rev)

    def test_correlation(self):
        self._correlation(self.msmrev)
        self._correlation(self.msmrev_eff)

    def _relaxation(self, msm):
        a = [1, 2, 3, 4, 5]
        p0 = [0.5, 0.2, 0.2, 0.1, 0.0]
        times, rel1 = msm.relaxation(msm.stationary_distribution, a, maxtime=50, k=5)
        # should be constant because we are in equilibrium
        assert (np.allclose(rel1 - rel1[0], np.zeros((np.shape(rel1)[0]))))
        times, rel2 = msm.relaxation(p0, a, maxtime=50, k=5)
        # check equality:
        if msm.is_reversible:
            assert np.allclose(rel2, self.rel_rev)
        else:
            assert np.allclose(rel2, self.rel)

    def test_relaxation(self):
        self._relaxation(self.msmrev)
        self._relaxation(self.msm)
        self._relaxation(self.msmrev_sparse)
        self._relaxation(self.msm_sparse)
        self._relaxation(self.msmrev_eff)

    def _fingerprint_correlation(self, msm):
        a = [1, 2, 3, 4, 5]
        b = np.array([1, -1, 0, -2, 4])
        if msm.is_reversible:
            fp1 = msm.fingerprint_correlation(a, k=5)
            # first timescale is infinite
            assert (fp1[0][0] == np.inf)
            # next timescales are identical to timescales:
            assert (np.allclose(fp1[0][1:], msm.timescales(4)))
            # all amplitudes nonnegative (for autocorrelation)
            assert (np.all(fp1[1][:] >= 0))
            fp2 = msm.fingerprint_correlation(a, b)
            assert np.allclose(fp2[1], self.fing_cor)
        else:  # raise ValueError, because fingerprints are not defined for nonreversible
            with self.assertRaises(ValueError):
                msm.fingerprint_correlation(a, k=5)
            with self.assertRaises(ValueError):
                msm.fingerprint_correlation(a, b, k=5)

    def test_fingerprint_correlation(self):
        self._fingerprint_correlation(self.msmrev)
        self._fingerprint_correlation(self.msm)
        self._fingerprint_correlation(self.msmrev_sparse)
        self._fingerprint_correlation(self.msm_sparse)
        self._fingerprint_correlation(self.msmrev_eff)

    def _fingerprint_relaxation(self, msm):
        a = [1, 2, 3, 4, 5]
        p0 = [0.5, 0.2, 0.2, 0.1, 0.0]
        if msm.is_reversible:
            # raise assertion error because size is wrong:
            with self.assertRaises(AssertionError):
                msm.fingerprint_relaxation(msm.stationary_distribution, [0, 1], k=5)
            # equilibrium relaxation should be constant
            fp1 = msm.fingerprint_relaxation(msm.stationary_distribution, a, k=5)
            # first timescale is infinite
            assert (fp1[0][0] == np.inf)
            # next timescales are identical to timescales:
            np.testing.assert_allclose(fp1[0][1:], msm.timescales(4), atol=1E-02)
            # dynamical amplitudes should be near 0 because we are in equilibrium
            assert (np.max(np.abs(fp1[1][1:])) < 1e-10)
            # off-equilibrium relaxation
            fp2 = msm.fingerprint_relaxation(p0, a, k=5)
            # first timescale is infinite
            assert (fp2[0][0] == np.inf)
            # next timescales are identical to timescales:
            assert (np.allclose(fp2[0][1:], msm.timescales(4)))
            # check equality
            assert np.allclose(fp2[1], self.fing_rel)
        else:  # raise ValueError, because fingerprints are not defined for nonreversible
            with self.assertRaises(ValueError):
                msm.fingerprint_relaxation(msm.stationary_distribution, a, k=5)
            with self.assertRaises(ValueError):
                msm.fingerprint_relaxation(p0, a)

    def test_fingerprint_relaxation(self):
        self._fingerprint_relaxation(self.msmrev)
        self._fingerprint_relaxation(self.msm)
        self._fingerprint_relaxation(self.msmrev_sparse)
        self._fingerprint_relaxation(self.msm_sparse)
        self._fingerprint_relaxation(self.msmrev_eff)

    # ---------------------------------
    # STATISTICS, SAMPLING
    # ---------------------------------

    def _active_state_indexes(self, msm):
        I = msm.active_state_indexes
        assert (len(I) == msm.nstates)
        # compare to histogram
        import pyemma.util.discrete_trajectories as dt
        hist = dt.count_states(msm.discrete_trajectories_full)
        # number of frames should match on active subset
        A = msm.active_set
        for i in range(A.shape[0]):
            assert (I[i].shape[0] == hist[A[i]])
            assert (I[i].shape[1] == 2)

    def test_active_state_indexes(self):
        self._active_state_indexes(self.msmrev)
        self._active_state_indexes(self.msm)
        self._active_state_indexes(self.msmrev_sparse)
        self._active_state_indexes(self.msm_sparse)
        self._active_state_indexes(self.msmrev_eff)

    def _generate_traj(self, msm):
        T = 10
        gt = msm.generate_traj(T)
        # Test: should have the right dimension
        assert (np.all(gt.shape == (T, 2)))

    def test_generate_traj(self):
        self._generate_traj(self.msmrev)
        self._generate_traj(self.msm)
        with warnings.catch_warnings(record=True) as w:
            self._generate_traj(self.msmrev_sparse)
        with warnings.catch_warnings(record=True) as w:
            self._generate_traj(self.msm_sparse)
        self._generate_traj(self.msmrev_eff)

    def _sample_by_state(self, msm):
        nsample = 100
        ss = msm.sample_by_state(nsample)
        # must have the right size
        assert (len(ss) == msm.nstates)
        # must be correctly assigned
        dtrajs_active = msm.discrete_trajectories_active
        for i, samples in enumerate(ss):
            # right shape
            assert (np.all(samples.shape == (nsample, 2)))
            for row in samples:
                assert (dtrajs_active[row[0]][row[1]] == i)

    def test_sample_by_state(self):
        self._sample_by_state(self.msmrev)
        self._sample_by_state(self.msm)
        self._sample_by_state(self.msmrev_sparse)
        self._sample_by_state(self.msm_sparse)
        self._sample_by_state(self.msmrev_eff)

    def _trajectory_weights(self, msm):
        W = msm.trajectory_weights()
        # should sum to 1
        wsum = 0
        for w in W:
            wsum += np.sum(w)
        assert (np.abs(wsum - 1.0) < 1e-6)

    def test_trajectory_weights(self):
        self._trajectory_weights(self.msmrev)
        self._trajectory_weights(self.msm)
        self._trajectory_weights(self.msmrev_sparse)
        self._trajectory_weights(self.msm_sparse)
        self._trajectory_weights(self.msmrev_eff)

    def test_simulate_MSM(self):
        msm = self.msm
        N = 100
        start = 1
        traj = msm.simulate(N=N, start=start)
        assert (len(traj) <= N)
        assert (len(np.unique(traj)) <= len(msm.transition_matrix))
        assert (start == traj[0])


class TestMSM_Incomplete(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the data:
        data = np.load(pkg_resources.resource_filename('pyemma.msm.tests', "data/TestData_OOM_MSM.npz"))
        indices = np.array([21, 25, 30, 40, 66, 72, 74, 91, 116, 158, 171, 175, 201, 239, 246, 280, 300, 301, 310, 318,
                            322, 323, 339, 352, 365, 368, 407, 412, 444, 475, 486, 494, 510, 529, 560, 617, 623, 637,
                            676, 689, 728, 731, 778, 780, 811, 828, 838, 845, 851, 859, 868, 874, 895, 933, 935, 938,
                            958, 961, 968, 974, 984, 990, 999])
        cls.dtrajs = []
        for k in range(1000):
            if k not in indices:
                cls.dtrajs.append(data['arr_%d' % k])

        # Number of states:
        cls.N = 5
        # Lag time:
        cls.tau = 5
        # Rank:
        cls.rank = 2
        # Build models:
        cls.msmrev = estimate_markov_model(cls.dtrajs, lag=cls.tau, weights='oom')
        cls.msm = estimate_markov_model(cls.dtrajs, lag=cls.tau, reversible=False, weights='oom')

        """Sparse"""
        cls.msmrev_sparse = estimate_markov_model(cls.dtrajs, lag=cls.tau, sparse=True, weights='oom')
        cls.msm_sparse = estimate_markov_model(cls.dtrajs, lag=cls.tau, reversible=False, sparse=True, weights='oom')

        # Reference count matrices at lag time tau and 2*tau:
        cls.C2t = data['C2t_s']
        cls.Ct = np.sum(cls.C2t, axis=1)
        # Restrict to active set:
        lcc = msmest.largest_connected_set(cls.Ct)
        cls.Ct_active = msmest.largest_connected_submatrix(cls.Ct, lcc=lcc)
        cls.C2t_active = cls.C2t[:4, :4, :4]
        cls.active_fraction = np.sum(cls.Ct_active) / np.sum(cls.Ct)

        # Compute OOM-components:
        cls.Xi, cls.omega, cls.sigma, cls.l = oom_transformations(cls.Ct_active, cls.C2t_active, cls.rank)
        # Compute corrected transition matrix:
        Tt_rev = TransitionMatrix(cls.Xi, cls.omega, cls.sigma, reversible=True)
        Tt = TransitionMatrix(cls.Xi, cls.omega, cls.sigma, reversible=False)

        # Build reference models:
        cls.rmsmrev = markov_model(Tt_rev)
        cls.rmsm = markov_model(Tt)

        "Compute further referenc quantities:"
        # Active count fraction:
        cls.hist = 1.0 * count_states(cls.dtrajs)
        cls.active_count_frac = np.sum(cls.hist[:4]) / np.sum(cls.hist)

        # Commitor and MFPT:
        a = np.array([0, 1])
        b = np.array([3])
        cls.comm_forward = cls.rmsm.committor_forward(a, b)
        cls.comm_forward_rev = cls.rmsmrev.committor_forward(a, b)
        cls.comm_backward = cls.rmsm.committor_backward(a, b)
        cls.comm_backward_rev = cls.rmsmrev.committor_backward(a, b)
        cls.mfpt = cls.tau * cls.rmsm.mfpt(a, b)
        cls.mfpt_rev = cls.tau * cls.rmsmrev.mfpt(a, b)
        # PCCA:
        cls.rmsmrev.pcca(2)
        cls.pcca_ass = cls.rmsmrev.metastable_assignments
        cls.pcca_dist = cls.rmsmrev.metastable_distributions
        cls.pcca_mem = cls.rmsmrev.metastable_memberships
        cls.pcca_sets = cls.rmsmrev.metastable_sets
        # Experimental quantities:
        a = np.array([1, 2, 3, 4])
        b = np.array([1, -1, 0, -2])
        p0 = np.array([0.5, 0.2, 0.2, 0.1])
        pi = cls.rmsm.stationary_distribution
        pi_rev = cls.rmsmrev.stationary_distribution
        _, _, L_rev = ma.rdl_decomposition(Tt_rev)
        cls.exp = np.dot(pi, a)
        cls.exp_rev = np.dot(pi_rev, a)
        cls.corr_rev = np.zeros(10)
        cls.rel = np.zeros(10)
        cls.rel_rev = np.zeros(10)
        for k in range(10):
            Ck_rev = np.dot(np.diag(pi_rev), np.linalg.matrix_power(Tt_rev, k))
            cls.corr_rev[k] = np.dot(a.T, np.dot(Ck_rev, b))
            cls.rel[k] = np.dot(p0.T, np.dot(np.linalg.matrix_power(Tt, k), a))
            cls.rel_rev[k] = np.dot(p0.T, np.dot(np.linalg.matrix_power(Tt_rev, k), a))
        cls.fing_cor = np.dot(a.T, L_rev.T) * np.dot(b.T, L_rev.T)
        cls.fing_rel = np.dot(a.T, L_rev.T) * np.dot((p0 / pi_rev).T, L_rev.T)

    # ---------------------------------
    # BASIC PROPERTIES
    # ---------------------------------

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            estimate_markov_model(self.dtrajs, lag=self.tau, weights=2)
        with self.assertRaises(ValueError):
            estimate_markov_model(self.dtrajs, lag=self.tau, weights='koopman')

    def test_reversible(self):
        # Reversible
        assert self.msmrev.is_reversible
        assert self.msmrev_sparse.is_reversible
        # Non-reversible
        assert not self.msm.is_reversible
        assert not self.msm_sparse.is_reversible

    def _sparse(self, msm):
        assert not (msm.is_sparse)

    def test_sparse(self):
        self._sparse(self.msmrev_sparse)
        self._sparse(self.msm_sparse)

    def _lagtime(self, msm):
        assert (msm.lagtime == self.tau)

    def test_lagtime(self):
        self._lagtime(self.msmrev)
        self._lagtime(self.msm)
        self._lagtime(self.msmrev_sparse)
        self._lagtime(self.msm_sparse)

    def test_active_set(self):

        assert np.all(self.msmrev.active_set == np.arange(self.N - 1, dtype=int))
        assert np.all(self.msmrev_sparse.active_set == np.arange(self.N - 1, dtype=int))
        assert np.all(self.msm.active_set == np.arange(self.N - 1, dtype=int))
        assert np.all(self.msm_sparse.active_set == np.arange(self.N - 1, dtype=int))

    def test_largest_connected_set(self):
        assert np.all(self.msmrev.largest_connected_set == np.arange(self.N - 1, dtype=int))
        assert np.all(self.msmrev_sparse.largest_connected_set == np.arange(self.N - 1, dtype=int))
        assert np.all(self.msm.largest_connected_set == np.arange(self.N - 1, dtype=int))
        assert np.all(self.msm_sparse.largest_connected_set == np.arange(self.N - 1, dtype=int))

    def _nstates(self, msm):
        # should always be <= full
        assert (msm.nstates <= msm.nstates_full)
        # THIS DATASET:
        assert (msm.nstates == 4)

    def test_nstates(self):
        self._nstates(self.msmrev)
        self._nstates(self.msm)
        self._nstates(self.msmrev_sparse)
        self._nstates(self.msm_sparse)

    def _connected_sets(self, msm):
        cs = msm.connected_sets
        assert (len(cs) >= 1)
        # MODE LARGEST:
        assert (np.all(cs[0] == msm.active_set))

    def test_connected_sets(self):
        self._connected_sets(self.msmrev)
        self._connected_sets(self.msm)
        self._connected_sets(self.msmrev_sparse)
        self._connected_sets(self.msm_sparse)

    def _connectivity(self, msm):
        # HERE:
        assert (msm.connectivity == 'largest')

    def test_connectivity(self):
        self._connectivity(self.msmrev)
        self._connectivity(self.msm)
        self._connectivity(self.msmrev_sparse)
        self._connectivity(self.msm_sparse)

    def _count_matrix_active(self, msm, sparse=False):
        if sparse:
            C = msm.count_matrix_active.toarray()
        else:
            C = msm.count_matrix_active
        assert np.allclose(C, self.Ct_active)

    def test_count_matrix_active(self):
        self._count_matrix_active(self.msmrev)
        self._count_matrix_active(self.msm)
        self._count_matrix_active(self.msmrev_sparse, sparse=True)
        self._count_matrix_active(self.msm_sparse, sparse=True)

    def _count_matrix_full(self, msm, sparse=False):
        if sparse:
            C = msm.count_matrix_full.toarray()
        else:
            C = msm.count_matrix_full
        assert np.allclose(C, self.Ct)

    def test_count_matrix_full(self):
        self._count_matrix_full(self.msmrev)
        self._count_matrix_full(self.msm)
        self._count_matrix_full(self.msmrev_sparse, sparse=True)
        self._count_matrix_full(self.msm_sparse, sparse=True)

    def _discrete_trajectories_full(self, msm):
        assert (np.all(self.dtrajs[0] == msm.discrete_trajectories_full[0]))
        assert len(self.dtrajs) == len(msm.discrete_trajectories_full)

    def test_discrete_trajectories_full(self):
        self._discrete_trajectories_full(self.msmrev)
        self._discrete_trajectories_full(self.msm)
        self._discrete_trajectories_full(self.msmrev_sparse)
        self._discrete_trajectories_full(self.msm_sparse)

    def _discrete_trajectories_active(self, msm):
        dtraj = self.dtrajs[15].copy()
        dtraj[dtraj == 4] = -1
        assert (np.all(dtraj == msm.discrete_trajectories_active[15]))
        assert len(self.dtrajs) == len(msm.discrete_trajectories_active)

    def test_discrete_trajectories_active(self):
        self._discrete_trajectories_active(self.msmrev)
        self._discrete_trajectories_active(self.msm)
        self._discrete_trajectories_active(self.msmrev_sparse)
        self._discrete_trajectories_active(self.msm_sparse)

    def _timestep(self, msm):
        assert (msm.timestep_model.startswith('5'))
        assert (msm.timestep_model.endswith('step'))

    def test_timestep(self):
        self._timestep(self.msmrev)
        self._timestep(self.msm)
        self._timestep(self.msmrev_sparse)
        self._timestep(self.msm_sparse)

    def _transition_matrix(self, msm):
        P = msm.transition_matrix
        # should be ndarray by default
        assert (isinstance(P, np.ndarray) or isinstance(P, scipy.sparse.csr_matrix))
        # shape
        assert (np.all(P.shape == (msm.nstates, msm.nstates)))
        # test transition matrix properties
        import msmtools.analysis as msmana
        assert (msmana.is_transition_matrix(P))
        assert (msmana.is_connected(P))
        # REVERSIBLE
        if msm.is_reversible:
            assert (msmana.is_reversible(P))
        # Test equality with model:
        if isinstance(P, scipy.sparse.csr_matrix):
            P = P.toarray()
        if msm.is_reversible:
            assert np.allclose(P, self.rmsmrev.transition_matrix)
        else:
            assert np.allclose(P, self.rmsm.transition_matrix)

    def test_transition_matrix(self):
        self._transition_matrix(self.msmrev)
        self._transition_matrix(self.msm)
        self._transition_matrix(self.msmrev_sparse)
        self._transition_matrix(self.msm_sparse)

    # ---------------------------------
    # SIMPLE STATISTICS
    # ---------------------------------

    def _active_state_fraction(self, msm):
        # should always be a fraction
        assert (0.0 <= msm.active_state_fraction <= 1.0)
        # For this data set:
        assert (msm.active_state_fraction == 0.8)

    def test_active_state_fraction(self):
        # should always be a fraction
        self._active_state_fraction(self.msmrev)
        self._active_state_fraction(self.msm)
        self._active_state_fraction(self.msmrev_sparse)
        self._active_state_fraction(self.msm_sparse)

    def _active_count_fraction(self, msm):
        # should always be a fraction
        assert (0.0 <= msm.active_count_fraction <= 1.0)
        # special case for this data set:
        assert (msm.active_count_fraction == self.active_count_frac)

    def test_active_count_fraction(self):
        self._active_count_fraction(self.msmrev)
        self._active_count_fraction(self.msm)
        self._active_count_fraction(self.msmrev_sparse)
        self._active_count_fraction(self.msm_sparse)

    # ---------------------------------
    # EIGENVALUES, EIGENVECTORS
    # ---------------------------------

    def _statdist(self, msm):
        mu = msm.stationary_distribution
        # should strictly positive (irreversibility)
        assert (np.all(mu > 0))
        # should sum to one
        assert (np.abs(np.sum(mu) - 1.0) < 1e-10)
        # Should match model:
        if msm.is_reversible:
            assert np.allclose(mu, self.rmsmrev.stationary_distribution)
        else:
            assert np.allclose(mu, self.rmsm.stationary_distribution)

    def test_statdist(self):
        self._statdist(self.msmrev)
        self._statdist(self.msm)
        self._statdist(self.msmrev_sparse)
        self._statdist(self.msm_sparse)

    def _eigenvalues(self, msm):
        ev = msm.eigenvalues()
        # stochasticity
        assert (np.max(np.abs(ev)) <= 1 + 1e-12)
        # irreducible
        assert (np.max(np.abs(ev[1:])) < 1)
        # ordered?
        evabs = np.abs(ev)
        for i in range(0, len(evabs) - 1):
            assert (evabs[i] >= evabs[i + 1])
        # REVERSIBLE:
        if msm.is_reversible:
            assert (np.all(np.isreal(ev)))

    def test_eigenvalues(self):
        self._eigenvalues(self.msmrev)
        self._eigenvalues(self.msm)
        self._eigenvalues(self.msmrev_sparse)
        self._eigenvalues(self.msm_sparse)

    def _eigenvectors_left(self, msm):
        L = msm.eigenvectors_left()
        k = msm.nstates
        # shape should be right
        assert (np.all(L.shape == (k, msm.nstates)))
        # first one should be identical to stat.dist
        l1 = L[0, :]
        err = msm.stationary_distribution - l1
        assert (np.max(np.abs(err)) < 1e-10)
        # sums should be 1, 0, 0, ...
        assert (np.allclose(np.sum(L[1:, :], axis=1), np.zeros(k - 1)))
        # REVERSIBLE:
        if msm.is_reversible:
            assert (np.all(np.isreal(L)))

    def test_eigenvectors_left(self):
        self._eigenvectors_left(self.msmrev)
        self._eigenvectors_left(self.msm)
        self._eigenvectors_left(self.msmrev_sparse)
        self._eigenvectors_left(self.msm_sparse)

    def _eigenvectors_right(self, msm):
        R = msm.eigenvectors_right()
        k = msm.nstates
        # shape should be right
        assert (np.all(R.shape == (msm.nstates, k)))
        # should be all ones
        r1 = R[:, 0]
        assert (np.allclose(r1, np.ones(msm.nstates)))
        # REVERSIBLE:
        if msm.is_reversible:
            assert (np.all(np.isreal(R)))

    def test_eigenvectors_right(self):
        self._eigenvectors_right(self.msmrev)
        self._eigenvectors_right(self.msm)
        self._eigenvectors_right(self.msmrev_sparse)
        self._eigenvectors_right(self.msm_sparse)

    def _eigenvectors_RDL(self, msm):
        R = msm.eigenvectors_right()
        D = np.diag(msm.eigenvalues())
        L = msm.eigenvectors_left()
        # orthogonality constraint
        assert (np.allclose(np.dot(R, L), np.eye(msm.nstates)))
        # REVERSIBLE: also true for LR because reversible matrix
        if msm.is_reversible:
            assert (np.allclose(np.dot(L, R), np.eye(msm.nstates)))
        # recover transition matrix
        assert (np.allclose(np.dot(R, np.dot(D, L)), msm.transition_matrix))

    def test_eigenvectors_RDL(self):
        self._eigenvectors_RDL(self.msmrev)
        self._eigenvectors_RDL(self.msm)
        self._eigenvectors_RDL(self.msmrev_sparse)
        self._eigenvectors_RDL(self.msm_sparse)

    def _timescales(self, msm):
        if not msm.is_reversible:
            with warnings.catch_warnings(record=True) as w:
                ts = msm.timescales()
        else:
            ts = msm.timescales()

        # should be all positive
        assert (np.all(ts > 0))
        # REVERSIBLE: should be all real
        if msm.is_reversible:
            ts_ref = self.rmsmrev.timescales()
            assert (np.all(np.isreal(ts)))
            # HERE:
            np.testing.assert_almost_equal(ts, self.tau * ts_ref, decimal=2)
        else:
            ts_ref = self.rmsm.timescales()
            # HERE:
            np.testing.assert_almost_equal(ts, self.tau * ts_ref, decimal=2)

    def test_timescales(self):
        self._timescales(self.msmrev)
        self._timescales(self.msm)
        self._timescales(self.msmrev_sparse)
        self._timescales(self.msm_sparse)

    def _eigenvalues_OOM(self, msm):
        assert np.allclose(msm.eigenvalues_oom, self.l)

    def test_eigenvalues_OOM(self):
        self._eigenvalues_OOM(self.msmrev)
        self._eigenvalues_OOM(self.msm)
        self._eigenvalues_OOM(self.msmrev_sparse)
        self._eigenvalues_OOM(self.msm_sparse)

    def _oom_components(self, msm):
        Xi = msm.oom_components
        omega = msm.oom_omega
        sigma = msm.oom_sigma
        assert np.allclose(Xi, self.Xi)
        assert np.allclose(omega, self.omega)
        assert np.allclose(sigma, self.sigma)

    def test_oom_components(self):
        self._oom_components(self.msmrev)
        self._oom_components(self.msm)
        self._oom_components(self.msmrev_sparse)
        self._oom_components(self.msm_sparse)

    # ---------------------------------
    # FIRST PASSAGE PROBLEMS
    # ---------------------------------

    def _committor(self, msm):
        a = np.array([0, 1])
        b = np.array([3])
        q_forward = msm.committor_forward(a, b)
        if msm.is_reversible:
            assert np.allclose(q_forward, self.comm_forward_rev)
        else:
            assert np.allclose(q_forward, self.comm_forward)
        q_backward = msm.committor_backward(a, b)
        if msm.is_reversible:
            assert np.allclose(q_backward, self.comm_backward_rev)
        else:
            assert np.allclose(q_backward, self.comm_backward)
        # REVERSIBLE:
        if msm.is_reversible:
            assert (np.allclose(q_forward + q_backward, np.ones(msm.nstates)))

    def test_committor(self):
        self._committor(self.msmrev)
        self._committor(self.msm)
        self._committor(self.msmrev_sparse)
        self._committor(self.msm_sparse)

    def _mfpt(self, msm):
        a = np.array([0, 1])
        b = np.array([3])
        t = msm.mfpt(a, b)
        assert (t > 0)
        # HERE:
        if msm.is_reversible:
            np.testing.assert_allclose(t, self.mfpt_rev, rtol=1e-3, atol=1e-6)
        else:
            np.testing.assert_allclose(t, self.mfpt, rtol=1e-3, atol=1e-6)

    def test_mfpt(self):
        self._mfpt(self.msmrev)
        self._mfpt(self.msm)
        self._mfpt(self.msmrev_sparse)
        self._mfpt(self.msm_sparse)

    # ---------------------------------
    # PCCA
    # ---------------------------------

    def _pcca_assignment(self, msm):
        if msm.is_reversible:
            msm.pcca(2)
            ass = msm.metastable_assignments
            # test: number of states
            assert (len(ass) == msm.nstates)
            # should be equal (zero variance) within metastable sets
            assert np.all(ass == self.pcca_ass)
        else:
            with self.assertRaises(ValueError):
                msm.pcca(2)

    def test_pcca_assignment(self):
        self._pcca_assignment(self.msmrev)
        self._pcca_assignment(self.msm)
        with warnings.catch_warnings(record=True) as w:
            self._pcca_assignment(self.msmrev_sparse)
        with warnings.catch_warnings(record=True) as w:
            self._pcca_assignment(self.msm_sparse)

    def _pcca_distributions(self, msm):
        if msm.is_reversible:
            msm.pcca(2)
            pccadist = msm.metastable_distributions
            # should be right size
            assert (np.all(pccadist.shape == (2, msm.nstates)))
            # should be nonnegative
            assert (np.all(pccadist >= 0))
            # check equality:
            assert np.allclose(pccadist, self.pcca_dist)
        else:
            with self.assertRaises(ValueError):
                msm.pcca(2)

    def test_pcca_distributions(self):
        self._pcca_distributions(self.msmrev)
        self._pcca_distributions(self.msm)
        self._pcca_distributions(self.msmrev_sparse)
        self._pcca_distributions(self.msm_sparse)

    def _pcca_memberships(self, msm):
        if msm.is_reversible:
            msm.pcca(2)
            M = msm.metastable_memberships
            # should be right size
            assert (np.all(M.shape == (msm.nstates, 2)))
            # should be nonnegative
            assert (np.all(M >= 0))
            # should add up to one:
            assert (np.allclose(np.sum(M, axis=1), np.ones(msm.nstates)))
            # check equality:
            assert np.allclose(M, self.pcca_mem)
        else:
            with self.assertRaises(ValueError):
                msm.pcca(2)

    def test_pcca_memberships(self):
        self._pcca_memberships(self.msmrev)
        self._pcca_memberships(self.msm)
        self._pcca_memberships(self.msmrev_sparse)
        self._pcca_memberships(self.msm_sparse)

    def _pcca_sets(self, msm):
        if msm.is_reversible:
            msm.pcca(2)
            S = msm.metastable_sets
            assignment = msm.metastable_assignments
            # should coincide with assignment
            for i, s in enumerate(S):
                for j in range(len(s)):
                    assert (assignment[s[j]] == i)
        else:
            with self.assertRaises(ValueError):
                msm.pcca(2)

    def test_pcca_sets(self):
        self._pcca_sets(self.msmrev)
        self._pcca_sets(self.msm)
        self._pcca_sets(self.msmrev_sparse)
        self._pcca_sets(self.msm_sparse)

    # ---------------------------------
    # EXPERIMENTAL STUFF
    # ---------------------------------

    def _expectation(self, msm):
        a = np.array([1, 2, 3, 4])
        e = msm.expectation(a)
        # approximately equal for both
        if msm.is_reversible:
            assert np.allclose(e, self.exp_rev)
        else:
            assert np.allclose(e, self.exp)

    def test_expectation(self):
        self._expectation(self.msmrev)
        self._expectation(self.msm)
        self._expectation(self.msmrev_sparse)
        self._expectation(self.msm_sparse)

    def _correlation(self, msm):
        a = [1, 2, 3, 4]
        b = [1, -1, 0, -2]
        with self.assertRaises(AssertionError):
            msm.correlation(a, 1)
        # test equality:
        _, cor = msm.correlation(a, b, maxtime=50)
        if msm.is_reversible:
            assert np.allclose(cor, self.corr_rev)

    def test_correlation(self):
        self._correlation(self.msmrev)

    def _relaxation(self, msm):
        a = [1, 2, 3, 4]
        p0 = [0.5, 0.2, 0.2, 0.1]
        times, rel1 = msm.relaxation(msm.stationary_distribution, a, maxtime=50, k=4)
        # should be constant because we are in equilibrium
        assert (np.allclose(rel1 - rel1[0], np.zeros((np.shape(rel1)[0]))))
        times, rel2 = msm.relaxation(p0, a, maxtime=50, k=4)
        # check equality:
        if msm.is_reversible:
            assert np.allclose(rel2, self.rel_rev)
        else:
            assert np.allclose(rel2, self.rel)

    def test_relaxation(self):
        self._relaxation(self.msmrev)
        self._relaxation(self.msm)
        self._relaxation(self.msmrev_sparse)
        self._relaxation(self.msm_sparse)

    def _fingerprint_correlation(self, msm):
        a = [1, 2, 3, 4]
        b = np.array([1, -1, 0, -2])
        if msm.is_reversible:
            fp1 = msm.fingerprint_correlation(a, k=4)
            # first timescale is infinite
            assert (fp1[0][0] == np.inf)
            # next timescales are identical to timescales:
            assert (np.allclose(fp1[0][1:], msm.timescales(3)))
            # all amplitudes nonnegative (for autocorrelation)
            assert (np.all(fp1[1][:] >= 0))
            fp2 = msm.fingerprint_correlation(a, b)
            assert np.allclose(fp2[1], self.fing_cor)
        else:  # raise ValueError, because fingerprints are not defined for nonreversible
            with self.assertRaises(ValueError):
                msm.fingerprint_correlation(a, k=4)
            with self.assertRaises(ValueError):
                msm.fingerprint_correlation(a, b, k=4)

    def test_fingerprint_correlation(self):
        self._fingerprint_correlation(self.msmrev)
        self._fingerprint_correlation(self.msm)
        self._fingerprint_correlation(self.msmrev_sparse)
        self._fingerprint_correlation(self.msm_sparse)

    def _fingerprint_relaxation(self, msm):
        a = [1, 2, 3, 4]
        p0 = [0.5, 0.2, 0.2, 0.1]
        if msm.is_reversible:
            # raise assertion error because size is wrong:
            with self.assertRaises(AssertionError):
                msm.fingerprint_relaxation(msm.stationary_distribution, [0, 1], k=4)
            # equilibrium relaxation should be constant
            fp1 = msm.fingerprint_relaxation(msm.stationary_distribution, a, k=4)
            # first timescale is infinite
            assert (fp1[0][0] == np.inf)
            # next timescales are identical to timescales:
            np.testing.assert_allclose(fp1[0][1:], msm.timescales(3), atol=1e-2)
            # dynamical amplitudes should be near 0 because we are in equilibrium
            assert (np.max(np.abs(fp1[1][1:])) < 1e-10)
            # off-equilibrium relaxation
            fp2 = msm.fingerprint_relaxation(p0, a, k=4)
            # first timescale is infinite
            assert (fp2[0][0] == np.inf)
            # next timescales are identical to timescales:
            np.testing.assert_allclose(fp2[0][1:], msm.timescales(3), atol=1e-2)
            # check equality
            assert np.allclose(fp2[1], self.fing_rel)
        else:  # raise ValueError, because fingerprints are not defined for nonreversible
            with self.assertRaises(ValueError):
                msm.fingerprint_relaxation(msm.stationary_distribution, a, k=4)
            with self.assertRaises(ValueError):
                msm.fingerprint_relaxation(p0, a)

    def test_fingerprint_relaxation(self):
        self._fingerprint_relaxation(self.msmrev)
        self._fingerprint_relaxation(self.msm)
        self._fingerprint_relaxation(self.msmrev_sparse)
        self._fingerprint_relaxation(self.msm_sparse)

        # ---------------------------------

    # STATISTICS, SAMPLING
    # ---------------------------------

    def _active_state_indexes(self, msm):
        I = msm.active_state_indexes
        assert (len(I) == msm.nstates)
        # compare to histogram
        import pyemma.util.discrete_trajectories as dt
        hist = dt.count_states(msm.discrete_trajectories_full)
        # number of frames should match on active subset
        A = msm.active_set
        for i in range(A.shape[0]):
            assert (I[i].shape[0] == hist[A[i]])
            assert (I[i].shape[1] == 2)

    def test_active_state_indexes(self):
        self._active_state_indexes(self.msmrev)
        self._active_state_indexes(self.msm)
        self._active_state_indexes(self.msmrev_sparse)
        self._active_state_indexes(self.msm_sparse)

    def _generate_traj(self, msm):
        T = 10
        gt = msm.generate_traj(T)
        # Test: should have the right dimension
        assert (np.all(gt.shape == (T, 2)))

    def test_generate_traj(self):
        self._generate_traj(self.msmrev)
        self._generate_traj(self.msm)
        with warnings.catch_warnings(record=True) as w:
            self._generate_traj(self.msmrev_sparse)
        with warnings.catch_warnings(record=True) as w:
            self._generate_traj(self.msm_sparse)

    def _sample_by_state(self, msm):
        nsample = 100
        ss = msm.sample_by_state(nsample)
        # must have the right size
        assert (len(ss) == msm.nstates)
        # must be correctly assigned
        dtrajs_active = msm.discrete_trajectories_active
        for i, samples in enumerate(ss):
            # right shape
            assert (np.all(samples.shape == (nsample, 2)))
            for row in samples:
                assert (dtrajs_active[row[0]][row[1]] == i)

    def test_sample_by_state(self):
        self._sample_by_state(self.msmrev)
        self._sample_by_state(self.msm)
        self._sample_by_state(self.msmrev_sparse)
        self._sample_by_state(self.msm_sparse)

    def _trajectory_weights(self, msm):
        dtr = msm.discrete_trajectories_full
        W = msm.trajectory_weights()
        # should sum to 1
        wsum = 0
        for w in W:
            wsum += np.sum(w)
        assert (np.abs(wsum - 1.0) < 1e-6)

    def test_trajectory_weights(self):
        self._trajectory_weights(self.msmrev)
        self._trajectory_weights(self.msm)
        self._trajectory_weights(self.msmrev_sparse)
        self._trajectory_weights(self.msm_sparse)

    def test_simulate_MSM(self):
        msm = self.msm
        N = 100
        start = 1
        traj = msm.simulate(N=N, start=start)
        assert (len(traj) <= N)
        assert (len(np.unique(traj)) <= len(msm.transition_matrix))
        assert (start == traj[0])


if __name__ == "__main__":
    unittest.main()
