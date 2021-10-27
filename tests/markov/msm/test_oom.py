r"""Unit test for the OOM-based MSM estimation.

"""
import os

import deeptime.markov.tools.analysis as ma
import deeptime.markov.tools.estimation as msmest
import numpy as np
import pytest
import scipy.linalg as scl
import scipy.sparse

from deeptime.decomposition import vamp_score_cv
from deeptime.markov import sample, count_states
from deeptime.markov.msm import MarkovStateModel, OOMReweightedMSM
from deeptime.numeric import sort_eigs


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
    l, R = sort_eigs(l, R)
    omega = np.real(R[:, 0])
    omega = omega / np.dot(omega, sigma)

    return Xi, omega, sigma, l


def compute_transition_matrix(Xi, omega, sigma, reversible=True):
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


class FiveStateSetup(object):
    def __init__(self, complete: bool = True):
        self.complete = complete
        data = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'TestData_OOM_MSM.npz'))
        if complete:
            self.dtrajs = [data['arr_%d' % k] for k in range(1000)]
        else:
            excluded = [
                21, 25, 30, 40, 66, 72, 74, 91, 116, 158, 171, 175, 201, 239, 246, 280, 300, 301, 310, 318,
                322, 323, 339, 352, 365, 368, 407, 412, 444, 475, 486, 494, 510, 529, 560, 617, 623, 637,
                676, 689, 728, 731, 778, 780, 811, 828, 838, 845, 851, 859, 868, 874, 895, 933, 935, 938,
                958, 961, 968, 974, 984, 990, 999
            ]
            self.dtrajs = [data['arr_%d' % k] for k in np.setdiff1d(np.arange(1000), excluded)]
        # Number of states:
        self.N = 5
        # Lag time:
        self.tau = 5
        self.dtrajs_lag = [traj[:-self.tau] for traj in self.dtrajs]
        # Rank:
        if complete:
            self.rank = 3
        else:
            self.rank = 2

        # Build models:
        self.msmrev = OOMReweightedMSM(lagtime=self.tau, rank_mode='bootstrap_trajs').fit(self.dtrajs)
        self.msmrev_sparse = OOMReweightedMSM(lagtime=self.tau, sparse=True, rank_mode='bootstrap_trajs') \
            .fit(self.dtrajs)
        self.msm = OOMReweightedMSM(lagtime=self.tau, reversible=False, rank_mode='bootstrap_trajs').fit(self.dtrajs)
        self.msm_sparse = OOMReweightedMSM(lagtime=self.tau, reversible=False, sparse=True,
                                           rank_mode='bootstrap_trajs').fit(self.dtrajs)
        self.estimators = [self.msmrev, self.msm, self.msmrev_sparse, self.msm_sparse]
        self.msms = [est.fetch_model() for est in self.estimators]

        # Reference count matrices at lag time tau and 2*tau:
        if complete:
            self.C2t = data['C2t']
        else:
            self.C2t = data['C2t_s']
        self.Ct = np.sum(self.C2t, axis=1)

        if complete:
            self.Ct_active = self.Ct
            self.C2t_active = self.C2t
            self.active_faction = 1.
        else:
            lcc = msmest.largest_connected_set(self.Ct)
            self.Ct_active = msmest.largest_connected_submatrix(self.Ct, lcc=lcc)
            self.C2t_active = self.C2t[:4, :4, :4]
            self.active_fraction = np.sum(self.Ct_active) / np.sum(self.Ct)

        # Compute OOM-components:
        self.Xi, self.omega, self.sigma, self.l = oom_transformations(self.Ct_active, self.C2t_active, self.rank)

        # Compute corrected transition matrix:
        Tt_rev = compute_transition_matrix(self.Xi, self.omega, self.sigma, reversible=True)
        Tt = compute_transition_matrix(self.Xi, self.omega, self.sigma, reversible=False)

        # Build reference models:
        self.rmsmrev = MarkovStateModel(Tt_rev)
        self.rmsm = MarkovStateModel(Tt)

        # Active count fraction:
        self.hist = count_states(self.dtrajs)
        self.active_hist = self.hist[:-1] if not complete else self.hist

        self.active_count_frac = float(np.sum(self.active_hist)) / np.sum(self.hist) if not complete else 1.
        self.active_state_frac = 0.8 if not complete else 1.

        # Commitor and MFPT:
        a = np.array([0, 1])
        b = np.array([4]) if complete else np.array([3])
        self.comm_forward = self.rmsm.committor_forward(a, b)
        self.comm_forward_rev = self.rmsmrev.committor_forward(a, b)
        self.comm_backward = self.rmsm.committor_backward(a, b)
        self.comm_backward_rev = self.rmsmrev.committor_backward(a, b)
        self.mfpt = self.tau * self.rmsm.mfpt(a, b)
        self.mfpt_rev = self.tau * self.rmsmrev.mfpt(a, b)
        # PCCA:
        pcca = self.rmsmrev.pcca(3 if complete else 2)
        self.pcca_ass = pcca.assignments
        self.pcca_dist = pcca.metastable_distributions
        self.pcca_mem = pcca.memberships
        self.pcca_sets = pcca.sets
        # Experimental quantities:
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([1, -1, 0, -2, 4])
        p0 = np.array([0.5, 0.2, 0.2, 0.1, 0.0])
        if not complete:
            a = a[:-1]
            b = b[:-1]
            p0 = p0[:-1]
        pi = self.rmsm.stationary_distribution
        pi_rev = self.rmsmrev.stationary_distribution
        _, _, L_rev = ma.rdl_decomposition(Tt_rev)
        self.exp = np.dot(self.rmsm.stationary_distribution, a)
        self.exp_rev = np.dot(self.rmsmrev.stationary_distribution, a)
        self.corr_rev = np.zeros(10)
        self.rel = np.zeros(10)
        self.rel_rev = np.zeros(10)
        for k in range(10):
            Ck_rev = np.dot(np.diag(pi_rev), np.linalg.matrix_power(Tt_rev, k))
            self.corr_rev[k] = np.dot(a.T, np.dot(Ck_rev, b))
            self.rel[k] = np.dot(p0.T, np.dot(np.linalg.matrix_power(Tt, k), a))
            self.rel_rev[k] = np.dot(p0.T, np.dot(np.linalg.matrix_power(Tt_rev, k), a))

        self.fing_cor = np.dot(a.T, L_rev.T) * np.dot(b.T, L_rev.T)
        self.fing_rel = np.dot(a.T, L_rev.T) * np.dot((p0 / pi_rev).T, L_rev.T)


@pytest.fixture(scope="module")
def five_state_msm():
    return FiveStateSetup()


@pytest.fixture(scope="module")
def five_state_msm_incomplete():
    return FiveStateSetup(complete=False)


@pytest.fixture(params=['five_state_msm', 'five_state_msm_incomplete'])
def oom_msm_scenario(request):
    return request.getfixturevalue(request.param)


# ---------------------------------
# SCORE
# ---------------------------------

def test_score(five_state_msm):
    for msm in five_state_msm.estimators:
        dtrajs_test = five_state_msm.dtrajs[0:500]
        s1 = msm.fetch_model().score(dtrajs_test, r=1, dim=2)
        assert 1.0 <= s1 <= 2.0
        s2 = msm.fetch_model().score(dtrajs_test, r=2, dim=2)
        assert 1.0 <= s2 <= 2.0


@pytest.mark.parametrize("reversible,sparse", [(True, True), (True, False), (False, True), (False, False)])
def test_score_cv(five_state_msm, reversible, sparse):
    msm = OOMReweightedMSM(lagtime=5, reversible=reversible, sparse=sparse)
    s1 = vamp_score_cv(msm, trajs=five_state_msm.dtrajs[:500], lagtime=5, n=2, r=1, dim=2, blocksplit=False).mean()
    np.testing.assert_(1.0 <= s1 <= 2.0)
    s2 = vamp_score_cv(msm, trajs=five_state_msm.dtrajs[:500], lagtime=5, n=2, r=2, dim=2, blocksplit=False).mean()
    np.testing.assert_(1.0 <= s2 <= 2.0)


# ---------------------------------
# BASIC PROPERTIES
# ---------------------------------


def test_basic_oom_properties(oom_msm_scenario):
    for est in oom_msm_scenario.estimators:
        model = est.fetch_model()
        np.testing.assert_equal(est.lagtime, oom_msm_scenario.tau)
        np.testing.assert_equal(model.lagtime, oom_msm_scenario.tau)
        np.testing.assert_(model.count_model.is_full_model == oom_msm_scenario.complete)
        np.testing.assert_equal(len(model.count_model.connected_sets()), 1)
        np.testing.assert_equal(model.n_states, 5 if oom_msm_scenario.complete else 4)
        np.testing.assert_equal(model.count_model.state_symbols, np.arange(5 if oom_msm_scenario.complete else 4))
        if est.sparse:
            np.testing.assert_allclose(oom_msm_scenario.Ct_active, model.count_model.count_matrix.toarray(),
                                       rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(oom_msm_scenario.Ct, model.count_model.count_matrix_full.toarray(),
                                       rtol=1e-5, atol=1e-8)
        else:
            np.testing.assert_allclose(oom_msm_scenario.Ct_active, model.count_model.count_matrix,
                                       rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(oom_msm_scenario.Ct, model.count_model.count_matrix_full,
                                       rtol=1e-5, atol=1e-8)
        np.testing.assert_equal(model.count_model.selected_state_fraction, oom_msm_scenario.active_state_frac)
        np.testing.assert_array_equal(model.count_model.state_histogram, oom_msm_scenario.active_hist)
        np.testing.assert_equal(model.count_model.selected_count_fraction, oom_msm_scenario.active_count_frac)


# ---------------------------------
# EIGENVALUES, EIGENVECTORS
# ---------------------------------


def test_transition_matrix(oom_msm_scenario):
    for msm in oom_msm_scenario.msms:
        P = msm.transition_matrix
        # should be ndarray by default
        np.testing.assert_(isinstance(P, np.ndarray) or isinstance(P, scipy.sparse.csr_matrix))
        # shape
        np.testing.assert_equal(P.shape, (msm.n_states, msm.n_states))
        # test transition matrix properties
        import deeptime.markov.tools.analysis as msmana
        np.testing.assert_(msmana.is_transition_matrix(P))
        np.testing.assert_(msmana.is_connected(P))
        # REVERSIBLE
        if msm.reversible:
            np.testing.assert_(msmana.is_reversible(P))
        # Test equality with model:
        from scipy.sparse import issparse
        if issparse(P):
            P = P.toarray()
        if msm.reversible:
            np.testing.assert_allclose(P, oom_msm_scenario.rmsmrev.transition_matrix)
        else:
            np.testing.assert_allclose(P, oom_msm_scenario.rmsm.transition_matrix)


def test_stationary_distribution(oom_msm_scenario):
    for msm in oom_msm_scenario.msms:
        # should strictly positive (irreversibility)
        np.testing.assert_(np.all(msm.stationary_distribution > 0))
        # should sum to one
        np.testing.assert_almost_equal(np.sum(msm.stationary_distribution), 1.)
        # Should match model:
        if msm.reversible:
            np.testing.assert_array_almost_equal(msm.stationary_distribution,
                                                 oom_msm_scenario.rmsmrev.stationary_distribution)
        else:
            np.testing.assert_array_almost_equal(msm.stationary_distribution,
                                                 oom_msm_scenario.rmsm.stationary_distribution)


def test_eigenvalues(oom_msm_scenario):
    for msm in oom_msm_scenario.msms:
        ev = msm.eigenvalues()
        # stochasticity
        np.testing.assert_(np.max(np.abs(ev)) <= 1 + 1e-12)
        # irreducible
        np.testing.assert_(np.max(np.abs(ev[1:])) < 1)
        # ordered?
        evabs = np.abs(ev)
        for i in range(0, len(evabs) - 1):
            np.testing.assert_(evabs[i] >= evabs[i + 1])
        # REVERSIBLE:
        if msm.reversible:
            np.testing.assert_(np.all(np.isreal(ev)))


def test_eigenvectors(oom_msm_scenario):
    for msm in oom_msm_scenario.msms:
        L = msm.eigenvectors_left()
        D = np.diag(msm.eigenvalues())
        R = msm.eigenvectors_right()
        np.testing.assert_equal(L.shape, (msm.n_states, msm.n_states))
        np.testing.assert_equal(R.shape, (msm.n_states, msm.n_states))
        np.testing.assert_allclose(np.dot(R, L), np.eye(msm.n_states), rtol=1e-5, atol=1e-8,
                                   err_msg="orthogonality constraint")

        l1 = L[0, :]
        r1 = R[:, 0]
        np.testing.assert_allclose(r1, np.ones(msm.n_states), rtol=1e-5, atol=1e-8, err_msg="should be all ones")
        err = msm.stationary_distribution - l1
        np.testing.assert_(np.max(np.abs(err)) < 1e-10)
        np.testing.assert_allclose(np.sum(L[1:, :], axis=1), np.zeros(msm.n_states - 1), rtol=1e-5, atol=1e-8,
                                   err_msg="sums should be 1, 0, 0, ...")
        if msm.reversible:
            np.testing.assert_(np.all(np.isreal(L)))
            np.testing.assert_(np.all(np.isreal(R)))
            np.testing.assert_allclose(np.dot(L, R), np.eye(msm.n_states), rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(np.dot(R, np.dot(D, L)), msm.transition_matrix, rtol=1e-5, atol=1e-8,
                                   err_msg="recover transition matrix")


def test_timescales(oom_msm_scenario):
    for msm in oom_msm_scenario.msms:
        ts = msm.timescales()

        np.testing.assert_(np.all(ts > 0), msg="should be all positive")
        if msm.reversible:
            ts_ref = oom_msm_scenario.rmsmrev.timescales()
            np.testing.assert_(np.all(np.isreal(ts)), msg="REVERSIBLE: should be all real")
        else:
            ts_ref = oom_msm_scenario.rmsm.timescales()
        np.testing.assert_almost_equal(ts, oom_msm_scenario.tau * ts_ref, decimal=2)


def test_oom_properties(oom_msm_scenario):
    for msm in oom_msm_scenario.msms:
        np.testing.assert_array_almost_equal(msm.oom_eigenvalues, oom_msm_scenario.l)
        np.testing.assert_array_almost_equal(msm.oom_components, oom_msm_scenario.Xi)
        np.testing.assert_array_almost_equal(msm.oom_information_state_vector, oom_msm_scenario.omega)
        np.testing.assert_array_almost_equal(msm.oom_evaluator, oom_msm_scenario.sigma)


def test_committor(oom_msm_scenario):
    for msm in oom_msm_scenario.msms:
        a = np.array([0, 1])
        b = np.array([4]) if oom_msm_scenario.complete else np.array([3])
        q_forward = msm.committor_forward(a, b)
        if msm.reversible:
            np.testing.assert_allclose(q_forward, oom_msm_scenario.comm_forward_rev)
        else:
            np.testing.assert_allclose(q_forward, oom_msm_scenario.comm_forward)
        q_backward = msm.committor_backward(a, b)
        if msm.reversible:
            np.testing.assert_allclose(q_backward, oom_msm_scenario.comm_backward_rev)
        else:
            np.testing.assert_allclose(q_backward, oom_msm_scenario.comm_backward)
        # REVERSIBLE:
        if msm.reversible:
            np.testing.assert_allclose(q_forward + q_backward, np.ones(msm.n_states))


def test_mfpt(oom_msm_scenario):
    for msm in oom_msm_scenario.msms:
        t = msm.mfpt(A=np.array([0, 1]), B=np.array([4]) if oom_msm_scenario.complete else np.array([3]))
        np.testing.assert_(t > 0)
        if msm.reversible:
            np.testing.assert_allclose(t, oom_msm_scenario.mfpt_rev, rtol=1e-3, atol=1e-6)
        else:
            np.testing.assert_allclose(t, oom_msm_scenario.mfpt, rtol=1e-3, atol=1e-6)


def test_pcca(oom_msm_scenario):
    for msm in oom_msm_scenario.msms:
        if msm.reversible:
            n_coarse = 3 if oom_msm_scenario.complete else 2
            pcca = msm.pcca(n_coarse)
            np.testing.assert_equal(len(pcca.assignments), msm.n_states)
            np.testing.assert_array_almost_equal(pcca.assignments, oom_msm_scenario.pcca_ass)

            np.testing.assert_equal(pcca.metastable_distributions.shape, (n_coarse, msm.n_states))
            np.testing.assert_(np.all(pcca.metastable_distributions >= 0))
            np.testing.assert_array_almost_equal(pcca.metastable_distributions, oom_msm_scenario.pcca_dist)

            np.testing.assert_equal(pcca.memberships.shape, (msm.n_states, n_coarse))
            np.testing.assert_(np.all(pcca.memberships >= 0))
            np.testing.assert_allclose(pcca.memberships.sum(axis=1), 1.)
            np.testing.assert_array_almost_equal(pcca.memberships, oom_msm_scenario.pcca_mem)

            for i, s in enumerate(pcca.sets):
                for j in range(len(s)):
                    assert (pcca.assignments[s[j]] == i)
        else:
            with np.testing.assert_raises(ValueError):
                msm.pcca(3)


def test_expectation_correlation_relaxation(oom_msm_scenario):
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([1, -1, 0, -2, 4])
    p0 = np.array([0.5, 0.2, 0.2, 0.1, 0.0])
    if not oom_msm_scenario.complete:
        a = a[:-1]
        b = b[:-1]
        p0 = p0[:-1]
    for msm in oom_msm_scenario.msms:
        expectation = msm.expectation(a)
        if msm.reversible:
            np.testing.assert_allclose(expectation, oom_msm_scenario.exp_rev)
        else:
            np.testing.assert_allclose(expectation, oom_msm_scenario.exp)

        with np.testing.assert_raises(ValueError):
            msm.correlation(a, 1)
        # test equality:
        _, cor = msm.correlation(a, b, maxtime=50)
        if msm.reversible:
            np.testing.assert_allclose(cor, oom_msm_scenario.corr_rev)

        times, rel1 = msm.relaxation(msm.stationary_distribution, a, maxtime=50, k=5)
        # should be constant because we are in equilibrium
        np.testing.assert_allclose(rel1 - rel1[0], 0, atol=1e-5)
        times, rel2 = msm.relaxation(p0, a, maxtime=50, k=5)
        # check equality:
        if msm.reversible:
            np.testing.assert_allclose(rel2, oom_msm_scenario.rel_rev)
        else:
            np.testing.assert_allclose(rel2, oom_msm_scenario.rel)


def test_fingerprint_correlation(oom_msm_scenario):
    a = [1, 2, 3, 4, 5]
    b = np.array([1, -1, 0, -2, 4])
    k = 5
    if not oom_msm_scenario.complete:
        a = a[:-1]
        b = b[:-1]
        k = 4
    for msm in oom_msm_scenario.msms:
        if msm.reversible:
            fp1 = msm.fingerprint_correlation(a, k=k)
            # first timescale is infinite
            np.testing.assert_equal(fp1[0][0], np.inf)
            # next timescales are identical to timescales:
            np.testing.assert_allclose(fp1[0][1:], msm.timescales(k - 1), rtol=1e-1, atol=1e-1)
            # all amplitudes nonnegative (for autocorrelation)
            np.testing.assert_(np.all(fp1[1][:] >= 0))
            fp2 = msm.fingerprint_correlation(a, b)
            np.testing.assert_array_almost_equal(fp2[1], oom_msm_scenario.fing_cor)
        else:  # raise ValueError, because fingerprints are not defined for nonreversible
            with np.testing.assert_raises(ValueError):
                msm.fingerprint_correlation(a, k=k)
            with np.testing.assert_raises(ValueError):
                msm.fingerprint_correlation(a, b, k=k)


def test_fingerprint_relaxation(oom_msm_scenario):
    a = [1, 2, 3, 4, 5]
    p0 = [0.5, 0.2, 0.2, 0.1, 0.0]
    k = 5
    if not oom_msm_scenario.complete:
        a = a[:-1]
        p0 = p0[:-1]
        k = 4
    for msm in oom_msm_scenario.msms:
        if msm.reversible:
            # raise assertion error because size is wrong:
            with np.testing.assert_raises(ValueError):
                msm.fingerprint_relaxation(msm.stationary_distribution, [0, 1], k=k)
            # equilibrium relaxation should be constant
            fp1 = msm.fingerprint_relaxation(msm.stationary_distribution, a, k=k)
            # first timescale is infinite
            np.testing.assert_equal(fp1[0][0], np.inf)
            # next timescales are identical to timescales:
            np.testing.assert_allclose(fp1[0][1:], msm.timescales(4), atol=1e-2)
            # dynamical amplitudes should be near 0 because we are in equilibrium
            np.testing.assert_(np.max(np.abs(fp1[1][1:])) < 1e-10)
            # off-equilibrium relaxation
            fp2 = msm.fingerprint_relaxation(p0, a, k=k)
            # first timescale is infinite
            np.testing.assert_equal(fp2[0][0], np.inf)
            # next timescales are identical to timescales:
            np.testing.assert_allclose(fp2[0][1:], msm.timescales(k - 1), atol=1e-1, rtol=1e-1)
            # check equality
            np.testing.assert_allclose(fp2[1], oom_msm_scenario.fing_rel)
        else:  # raise ValueError, because fingerprints are not defined for nonreversible
            with np.testing.assert_raises(ValueError):
                msm.fingerprint_relaxation(msm.stationary_distribution, a, k=k)
            with np.testing.assert_raises(ValueError):
                msm.fingerprint_relaxation(p0, a)


def test_active_state_indices(oom_msm_scenario):
    for msm in oom_msm_scenario.msms:
        dtrajs_proj = msm.count_model.transform_discrete_trajectories_to_submodel(oom_msm_scenario.dtrajs)
        indices = sample.compute_index_states(dtrajs_proj)
        np.testing.assert_equal(len(indices), msm.n_states)
        hist = count_states(oom_msm_scenario.dtrajs)
        for state in range(msm.n_states):
            np.testing.assert_equal(indices[state].shape[0], hist[msm.count_model.state_symbols[state]])
            np.testing.assert_equal(indices[state].shape[1], 2)


def test_generate_trajectory(oom_msm_scenario):
    for msm in oom_msm_scenario.msms:
        dtrajs_proj = msm.count_model.transform_discrete_trajectories_to_submodel(oom_msm_scenario.dtrajs)
        indices = sample.compute_index_states(dtrajs_proj)

        traj = msm.simulate(10)
        ix = sample.indices_by_sequence(indices, traj)
        np.testing.assert_equal(ix.shape, (10, 2))


def test_sample_by_state(oom_msm_scenario):
    for msm in oom_msm_scenario.msms:
        n_samples = 100
        dtrajs_active = msm.count_model.transform_discrete_trajectories_to_submodel(oom_msm_scenario.dtrajs)
        ss = sample.by_state(dtrajs_active, n_samples)
        # must have the right size
        np.testing.assert_equal(len(ss), msm.n_states)
        for i, samples in enumerate(ss):
            # right shape
            np.testing.assert_equal(samples.shape, (n_samples, 2))
            for row in samples:
                np.testing.assert_equal(dtrajs_active[row[0]][row[1]], i)


def test_trajectory_weights(oom_msm_scenario):
    for msm in oom_msm_scenario.msms:
        weights = msm.compute_trajectory_weights(oom_msm_scenario.dtrajs)
        wsum = sum(np.sum(w) for w in weights)
        np.testing.assert_almost_equal(wsum, 1., err_msg="should sum to 1")


def test_simulate(oom_msm_scenario):
    for msm in oom_msm_scenario.msms:
        traj = msm.simulate(n_steps=100, start=1)
        np.testing.assert_(len(traj) <= 100)
        np.testing.assert_(len(np.unique(traj)) <= msm.n_states)
        np.testing.assert_equal(1, traj[0])
