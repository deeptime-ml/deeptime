
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


r"""Unit test for the MSM module

.. moduleauthor:: F. Noe <frank DOT noe AT fu-berlin DOT de>
.. moduleauthor:: B. Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest

import numpy as np
import scipy.sparse
import warnings

from msmtools.generation import generate_traj
from msmtools.estimation import count_matrix, largest_connected_set, largest_connected_submatrix, transition_matrix
from msmtools.analysis import stationary_distribution, timescales


class TestMSMSimple(unittest.TestCase):
    def setUp(self):
        """Store state of the rng"""
        self.state = np.random.mtrand.get_state()

        """Reseed the rng to enforce 'deterministic' behavior"""
        np.random.mtrand.seed(42)

        """Meta-stable birth-death chain"""
        b = 2
        q = np.zeros(7)
        p = np.zeros(7)
        q[1:] = 0.5
        p[0:-1] = 0.5
        q[2] = 1.0 - 10 ** (-b)
        q[4] = 10 ** (-b)
        p[2] = 10 ** (-b)
        p[4] = 1.0 - 10 ** (-b)

        bdc = BirthDeathChain(q, p)
        P = bdc.transition_matrix()
        self.dtraj = generate_traj(P, 10000, start=0)
        self.tau = 1

        """Estimate MSM"""
        self.C_MSM = count_matrix(self.dtraj, self.tau, sliding=True)
        self.lcc_MSM = largest_connected_set(self.C_MSM)
        self.Ccc_MSM = largest_connected_submatrix(self.C_MSM, lcc=self.lcc_MSM)
        self.P_MSM = transition_matrix(self.Ccc_MSM, reversible=True)
        self.mu_MSM = stationary_distribution(self.P_MSM)
        self.k = 3
        self.ts = timescales(self.P_MSM, k=self.k, tau=self.tau)

    def tearDown(self):
        """Revert the state of the rng"""
        np.random.mtrand.set_state(self.state)

    def test_MSM(self):
        msm = estimate_markov_model(self.dtraj, self.tau)
        assert_allclose(self.dtraj, msm.discrete_trajectories_full[0])
        self.assertEqual(self.tau, msm.lagtime)
        assert_allclose(self.lcc_MSM, msm.largest_connected_set)
        self.assertTrue(np.allclose(self.Ccc_MSM.toarray(), msm.count_matrix_active))
        self.assertTrue(np.allclose(self.C_MSM.toarray(), msm.count_matrix_full))
        self.assertTrue(np.allclose(self.P_MSM.toarray(), msm.transition_matrix))
        assert_allclose(self.mu_MSM, msm.stationary_distribution)
        assert_allclose(self.ts[1:], msm.timescales(self.k - 1))

    def test_MSM_sparse(self):
        msm = estimate_markov_model(self.dtraj, self.tau, sparse=True)
        assert_allclose(self.dtraj, msm.discrete_trajectories_full[0])
        self.assertEqual(self.tau, msm.lagtime)
        assert_allclose(self.lcc_MSM, msm.largest_connected_set)
        self.assertTrue(np.allclose(self.Ccc_MSM.toarray(), msm.count_matrix_active.toarray()))
        self.assertTrue(np.allclose(self.C_MSM.toarray(), msm.count_matrix_full.toarray()))
        self.assertTrue(np.allclose(self.P_MSM.toarray(), msm.transition_matrix.toarray()))
        assert_allclose(self.mu_MSM, msm.stationary_distribution)
        assert_allclose(self.ts[1:], msm.timescales(self.k - 1))

    def test_pcca_recompute(self):
        msm = estimate_markov_model(self.dtraj, self.tau)
        pcca1 = msm.pcca(2)
        msm.estimate(self.dtraj, lag=self.tau + 1)
        pcca2 = msm.pcca(2)
        assert pcca2 is not pcca1

    def test_rdl_recompute(self):
        """ test for issue 1301. Should recompute RDL decomposition in case of new transition matrix. """
        msm = estimate_markov_model(self.dtraj, self.tau)
        ev1 = msm.eigenvectors_left(2)
        msm.estimate(self.dtraj, lag=self.tau+1)
        ev2 = msm.eigenvectors_left(2)
        assert ev2 is not ev1


class TestMSMRevPi(unittest.TestCase):
    r"""Checks if the MLMSM correctly handles the active set computation
    if a stationary distribution is given"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_valid_stationary_vector(self):
        dtraj = np.array([0, 0, 1, 0, 1, 2])
        pi_valid = np.array([0.1, 0.9, 0.0])
        pi_invalid = np.array([0.1, 0.9])
        active_set = np.array([0, 1])
        msm = estimate_markov_model(dtraj, 1, statdist=pi_valid)
        self.assertTrue(np.all(msm.active_set==active_set))
        with self.assertRaises(ValueError):
            msm = estimate_markov_model(dtraj, 1, statdist=pi_invalid)

    def test_valid_trajectory(self):
        pi = np.array([0.1, 0.0, 0.9])
        dtraj_invalid = np.array([1, 1, 1, 1, 1, 1, 1])
        dtraj_valid = np.array([0, 2, 0, 2, 2, 0, 1, 1])
        msm = estimate_markov_model(dtraj_valid, 1, statdist=pi)
        self.assertTrue(np.all(msm.active_set==np.array([0, 2])))
        with self.assertRaises(ValueError):
            msm = estimate_markov_model(dtraj_invalid, 1, statdist=pi)


class TestMSMDoubleWell(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyemma.datasets
        cls.dtraj = pyemma.datasets.load_2well_discrete().dtraj_T100K_dt10
        nu = 1.*np.bincount(cls.dtraj)
        cls.statdist = nu/nu.sum()

        cls.tau = 10
        maxerr = 1e-12
        cls.msmrev = estimate_markov_model(cls.dtraj, cls.tau ,maxerr=maxerr)
        cls.msmrevpi = estimate_markov_model(cls.dtraj, cls.tau,maxerr=maxerr,
                                             statdist=cls.statdist)
        cls.msm = estimate_markov_model(cls.dtraj, cls.tau, reversible=False, maxerr=maxerr)

        """Sparse"""
        cls.msmrev_sparse = estimate_markov_model(cls.dtraj, cls.tau, sparse=True, maxerr=maxerr)
        cls.msmrevpi_sparse = estimate_markov_model(cls.dtraj, cls.tau,maxerr=maxerr,
                                                    statdist=cls.statdist,
                                                    sparse=True)
        cls.msm_sparse = estimate_markov_model(cls.dtraj, cls.tau, reversible=False, sparse=True, maxerr=maxerr)

    # ---------------------------------
    # SCORE
    # ---------------------------------
    def _score(self, msm):
        # check estimator args are not overwritten, if default arguments are used.
        old_score_k = msm.score_k
        old_score_method = msm.score_method
        dtrajs_test = self.dtraj[80000:]
        msm.score(dtrajs_test)
        assert msm.score_k == old_score_k
        assert msm.score_method == old_score_method
        s1 = msm.score(dtrajs_test, score_method='VAMP1', score_k=2)
        assert msm.score_k == 2
        assert msm.score_method == 'VAMP1'
        assert 1.0 <= s1 <= 2.0

        s2 = msm.score(dtrajs_test, score_method='VAMP2', score_k=2)
        assert 1.0 <= s2 <= 2.0
        assert msm.score_k == 2
        assert msm.score_method == 'VAMP2'
        # se = msm.score(dtrajs_test, score_method='VAMPE', score_k=2)
        # se_inf = msm.score(dtrajs_test, score_method='VAMPE', score_k=None)

    def test_score(self):
        self._score(self.msmrev)
        self._score(self.msmrevpi)
        self._score(self.msm)
        self._score(self.msmrev_sparse)
        self._score(self.msmrevpi_sparse)
        self._score(self.msm_sparse)

    def _score_cv(self, estimator):
        s1 = estimator.score_cv(self.dtraj, n=5, score_method='VAMP1', score_k=2).mean()
        assert 1.0 <= s1 <= 2.0
        s2 = estimator.score_cv(self.dtraj, n=5, score_method='VAMP2', score_k=2).mean()
        assert 1.0 <= s2 <= 2.0
        se = estimator.score_cv(self.dtraj, n=5, score_method='VAMPE', score_k=2).mean()
        se_inf = estimator.score_cv(self.dtraj, n=5, score_method='VAMPE', score_k=None).mean()

    def test_score_cv(self):
        self._score_cv(MaximumLikelihoodMSM(lag=10, reversible=True))
        self._score_cv(MaximumLikelihoodMSM(lag=10, reversible=True, statdist_constraint=self.statdist))
        self._score_cv(MaximumLikelihoodMSM(lag=10, reversible=False))
        self._score_cv(MaximumLikelihoodMSM(lag=10, reversible=True, sparse=True))
        self._score_cv(MaximumLikelihoodMSM(lag=10, reversible=True, statdist_constraint=self.statdist, sparse=True))
        self._score_cv(MaximumLikelihoodMSM(lag=10, reversible=False, sparse=True))

    # ---------------------------------
    # BASIC PROPERTIES
    # ---------------------------------

    def test_reversible(self):
        # NONREVERSIBLE
        assert self.msmrev.is_reversible
        assert self.msmrevpi.is_reversible
        assert (self.msmrev_sparse.is_reversible)
        assert self.msmrevpi_sparse.is_reversible
        # REVERSIBLE
        assert not self.msm.is_reversible
        assert not self.msm_sparse.is_reversible

    def _sparse(self, msm):
        assert (msm.is_sparse)

    def test_sparse(self):
        self._sparse(self.msmrev_sparse)
        self._sparse(self.msmrevpi_sparse)
        self._sparse(self.msm_sparse)

    def _lagtime(self, msm):
        assert (msm.lagtime == self.tau)

    def test_lagtime(self):
        self._lagtime(self.msmrev)
        self._lagtime(self.msmrevpi)
        self._lagtime(self.msm)
        self._lagtime(self.msmrev_sparse)
        self._lagtime(self.msmrevpi_sparse)
        self._lagtime(self.msm_sparse)

    def _active_set(self, msm):
        # should always be <= full set
        assert (len(msm.active_set) <= self.msm.nstates_full)
        # should be length of nstates
        assert (len(msm.active_set) == self.msm.nstates)

    def test_active_set(self):
        self._active_set(self.msmrev)
        self._active_set(self.msmrevpi)
        self._active_set(self.msm)
        self._active_set(self.msmrev_sparse)
        self._active_set(self.msmrevpi_sparse)
        self._active_set(self.msm_sparse)

    def _largest_connected_set(self, msm):
        lcs = msm.largest_connected_set
        # identical to first connected set
        assert (np.all(lcs == msm.connected_sets[0]))
        # LARGEST: identical to active set
        assert (np.all(lcs == msm.active_set))

    def test_largest_connected_set(self):
        self._largest_connected_set(self.msmrev)
        self._largest_connected_set(self.msmrevpi)
        self._largest_connected_set(self.msm)
        self._largest_connected_set(self.msmrev_sparse)
        self._largest_connected_set(self.msmrevpi_sparse)
        self._largest_connected_set(self.msm_sparse)

    def _nstates(self, msm):
        # should always be <= full
        assert (msm.nstates <= msm.nstates_full)
        # THIS DATASET:
        assert (msm.nstates == 66)

    def test_nstates(self):
        self._nstates(self.msmrev)
        self._nstates(self.msmrevpi)
        self._nstates(self.msm)
        self._nstates(self.msmrev_sparse)
        self._nstates(self.msmrevpi_sparse)
        self._nstates(self.msm_sparse)

    def _connected_sets(self, msm):
        cs = msm.connected_sets
        assert (len(cs) >= 1)
        # MODE LARGEST:
        assert (np.all(cs[0] == msm.active_set))

    def test_connected_sets(self):
        self._connected_sets(self.msmrev)
        self._connected_sets(self.msmrevpi)
        self._connected_sets(self.msm)
        self._connected_sets(self.msmrev_sparse)
        self._connected_sets(self.msmrevpi_sparse)
        self._connected_sets(self.msm_sparse)

    def _connectivity(self, msm):
        # HERE:
        assert (msm.connectivity == 'largest')

    def test_connectivity(self):
        self._connectivity(self.msmrev)
        self._connectivity(self.msmrevpi)
        self._connectivity(self.msm)
        self._connectivity(self.msmrev_sparse)
        self._connectivity(self.msmrevpi_sparse)
        self._connectivity(self.msm_sparse)

    def _count_matrix_active(self, msm):
        C = msm.count_matrix_active
        assert (np.all(C.shape == (msm.nstates, msm.nstates)))

    def test_count_matrix_active(self):
        self._count_matrix_active(self.msmrev)
        self._count_matrix_active(self.msmrevpi)
        self._count_matrix_active(self.msm)
        self._count_matrix_active(self.msmrev_sparse)
        self._count_matrix_active(self.msmrevpi_sparse)
        self._count_matrix_active(self.msm_sparse)

    def _count_matrix_full(self, msm):
        C = msm.count_matrix_full
        assert (np.all(C.shape == (msm.nstates_full, msm.nstates_full)))

    def test_count_matrix_full(self):
        self._count_matrix_full(self.msmrev)
        self._count_matrix_full(self.msmrevpi)
        self._count_matrix_full(self.msm)
        self._count_matrix_full(self.msmrev_sparse)
        self._count_matrix_full(self.msmrevpi_sparse)
        self._count_matrix_full(self.msm_sparse)

    def _discrete_trajectories_full(self, msm):
        assert (np.all(self.dtraj == msm.discrete_trajectories_full[0]))

    def test_discrete_trajectories_full(self):
        self._discrete_trajectories_full(self.msmrev)
        self._discrete_trajectories_full(self.msmrevpi)
        self._discrete_trajectories_full(self.msm)
        self._discrete_trajectories_full(self.msmrev_sparse)
        self._discrete_trajectories_full(self.msmrevpi_sparse)
        self._discrete_trajectories_full(self.msm_sparse)

    def _discrete_trajectories_active(self, msm):
        dta = msm.discrete_trajectories_active
        # HERE
        assert (len(dta) == 1)
        # HERE: states are shifted down from the beginning, because early states are missing
        assert (dta[0][0] < self.dtraj[0])

    def test_discrete_trajectories_active(self):
        self._discrete_trajectories_active(self.msmrev)
        self._discrete_trajectories_active(self.msmrevpi)
        self._discrete_trajectories_active(self.msm)
        self._discrete_trajectories_active(self.msmrev_sparse)
        self._discrete_trajectories_active(self.msmrevpi_sparse)
        self._discrete_trajectories_active(self.msm_sparse)

    def _timestep(self, msm):
        assert (msm.timestep_model.startswith('1'))
        assert (msm.timestep_model.endswith('step'))

    def test_timestep(self):
        self._timestep(self.msmrev)
        self._timestep(self.msmrevpi)
        self._timestep(self.msm)
        self._timestep(self.msmrev_sparse)
        self._timestep(self.msmrevpi_sparse)
        self._timestep(self.msm_sparse)

    def _dt_model(self, msm):
        from pyemma.util.units import TimeUnit
        tu = TimeUnit("1 step").get_scaled(self.msm.lagtime)
        self.assertEqual(msm.dt_model, tu)

    def test_dt_model(self):
        self._dt_model(self.msmrev)
        self._dt_model(self.msmrevpi)
        self._dt_model(self.msm)
        self._dt_model(self.msmrev_sparse)
        self._dt_model(self.msmrevpi_sparse)
        self._dt_model(self.msm_sparse)

    def _transition_matrix(self, msm):
        P = msm.transition_matrix
        # should be ndarray by default
        # assert (isinstance(P, np.ndarray))
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

    def test_transition_matrix(self):
        self._transition_matrix(self.msmrev)
        self._transition_matrix(self.msmrev)
        self._transition_matrix(self.msm)
        self._transition_matrix(self.msmrev_sparse)
        self._transition_matrix(self.msmrevpi_sparse)
        self._transition_matrix(self.msm_sparse)

    # ---------------------------------
    # SIMPLE STATISTICS
    # ---------------------------------

    def _active_count_fraction(self, msm):
        # should always be a fraction
        assert (0.0 <= msm.active_count_fraction <= 1.0)
        # special case for this data set:
        assert (msm.active_count_fraction == 1.0)

    def test_active_count_fraction(self):
        self._active_count_fraction(self.msmrev)
        self._active_count_fraction(self.msmrevpi)
        self._active_count_fraction(self.msm)
        self._active_count_fraction(self.msmrev_sparse)
        self._active_count_fraction(self.msmrevpi_sparse)
        self._active_count_fraction(self.msm_sparse)

    def _active_state_fraction(self, msm):
        # should always be a fraction
        assert (0.0 <= msm.active_state_fraction <= 1.0)

    def test_active_state_fraction(self):
        # should always be a fraction
        self._active_state_fraction(self.msmrev)
        self._active_state_fraction(self.msmrevpi)
        self._active_state_fraction(self.msm)
        self._active_state_fraction(self.msmrev_sparse)
        self._active_state_fraction(self.msmrevpi_sparse)
        self._active_state_fraction(self.msm_sparse)

    def _effective_count_matrix(self, msm):
        Ceff = msm.effective_count_matrix
        assert (np.all(Ceff.shape == (msm.nstates, msm.nstates)))

    def test_effective_count_matrix(self):
        self._effective_count_matrix(self.msmrev)
        self._effective_count_matrix(self.msmrevpi)
        self._effective_count_matrix(self.msm)
        self._effective_count_matrix(self.msmrev_sparse)
        self._effective_count_matrix(self.msmrevpi_sparse)
        self._effective_count_matrix(self.msm_sparse)

    # ---------------------------------
    # EIGENVALUES, EIGENVECTORS
    # ---------------------------------

    def _statdist(self, msm):
        mu = msm.stationary_distribution
        # should strictly positive (irreversibility)
        assert (np.all(mu > 0))
        # should sum to one
        assert (np.abs(np.sum(mu) - 1.0) < 1e-10)

    def test_statdist(self):
        self._statdist(self.msmrev)
        self._statdist(self.msmrevpi)
        self._statdist(self.msm)
        self._statdist(self.msmrev_sparse)
        self._statdist(self.msmrevpi_sparse)
        self._statdist(self.msm_sparse)

    def _eigenvalues(self, msm):
        if not msm.is_sparse:
            ev = msm.eigenvalues()
        else:
            k = 4
            ev = msm.eigenvalues(k)
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
        self._eigenvalues(self.msmrevpi)
        self._eigenvalues(self.msm)
        self._eigenvalues(self.msmrev_sparse)
        self._eigenvalues(self.msmrevpi_sparse)
        self._eigenvalues(self.msm_sparse)

    def _eigenvectors_left(self, msm):
        if not msm.is_sparse:
            L = msm.eigenvectors_left()
            k = msm.nstates
        else:
            k = 4
            L = msm.eigenvectors_left(k)
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
        self._eigenvectors_left(self.msmrevpi)
        self._eigenvectors_left(self.msm)
        self._eigenvectors_left(self.msmrev_sparse)
        self._eigenvectors_left(self.msmrevpi_sparse)
        self._eigenvectors_left(self.msm_sparse)

    def _eigenvectors_right(self, msm):
        if not msm.is_sparse:
            R = msm.eigenvectors_right()
            k = msm.nstates
        else:
            k = 4
            R = msm.eigenvectors_right(k)
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
        self._eigenvectors_right(self.msmrevpi)
        self._eigenvectors_right(self.msm)
        self._eigenvectors_right(self.msmrev_sparse)
        self._eigenvectors_right(self.msmrevpi_sparse)
        self._eigenvectors_right(self.msm_sparse)

    def _eigenvectors_RDL(self, msm):
        if not msm.is_sparse:
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

        else:
            k = 4
            R = msm.eigenvectors_right(k)
            D = np.diag(msm.eigenvalues(k))
            L = msm.eigenvectors_left(k)
            """Orthoginality"""
            assert (np.allclose(np.dot(L, R), np.eye(k)))
            """Reversibility"""
            if msm.is_reversible:
                mu = msm.stationary_distribution
                L_mu = mu[:,np.newaxis] * R
                assert (np.allclose(np.dot(L_mu.T, R), np.eye(k)))


    def test_eigenvectors_RDL(self):
        self._eigenvectors_RDL(self.msmrev)
        self._eigenvectors_RDL(self.msmrevpi)
        self._eigenvectors_RDL(self.msm)
        self._eigenvectors_RDL(self.msmrev_sparse)
        self._eigenvectors_RDL(self.msmrevpi_sparse)
        self._eigenvectors_RDL(self.msm_sparse)

    def _timescales(self, msm):
        if not msm.is_sparse:
            if not msm.is_reversible:
                with warnings.catch_warnings(record=True) as w:
                    ts = msm.timescales()
            else:
                ts = msm.timescales()
        else:
            k = 4
            if not msm.is_reversible:
                with warnings.catch_warnings(record=True) as w:
                    ts = msm.timescales(k)
            else:
                ts = msm.timescales(k)

        # should be all positive
        assert (np.all(ts > 0))
        # REVERSIBLE: should be all real
        if msm.is_reversible:
            ts_ref = np.array([310.87, 8.5, 5.09])
            assert (np.all(np.isreal(ts)))
            # HERE:
            np.testing.assert_almost_equal(ts[:3], ts_ref, decimal=2)
        else:
            ts_ref = np.array([310.49376926, 8.48302712, 5.02649564])
            # HERE:
            np.testing.assert_almost_equal(ts[:3], ts_ref, decimal=2)

    def test_timescales(self):
        self._timescales(self.msmrev)
        self._timescales(self.msm)
        self._timescales(self.msmrev_sparse)
        self._timescales(self.msm_sparse)

    # ---------------------------------
    # FIRST PASSAGE PROBLEMS
    # ---------------------------------

    def _committor(self, msm):
        a = 16
        b = 48
        q_forward = msm.committor_forward(a, b)
        assert (q_forward[a] == 0)
        assert (q_forward[b] == 1)
        assert (np.all(q_forward[:30] < 0.5))
        assert (np.all(q_forward[40:] > 0.5))
        q_backward = msm.committor_backward(a, b)
        assert (q_backward[a] == 1)
        assert (q_backward[b] == 0)
        assert (np.all(q_backward[:30] > 0.5))
        assert (np.all(q_backward[40:] < 0.5))
        # REVERSIBLE:
        if msm.is_reversible:
            assert (np.allclose(q_forward + q_backward, np.ones(msm.nstates)))

    def test_committor(self):
        self._committor(self.msmrev)
        self._committor(self.msm)
        self._committor(self.msmrev_sparse)
        self._committor(self.msm_sparse)

    def _mfpt(self, msm):
        a = 16
        b = 48
        t = msm.mfpt(a, b)
        assert (t > 0)
        # HERE:
        if msm.is_reversible:
            np.testing.assert_allclose(t, 872.69, rtol=1e-3, atol=1e-6)
        else:
            np.testing.assert_allclose(t, 872.07, rtol=1e-3, atol=1e-6)

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
            assert msm.n_metastable == 2
            # test: should be 0 or 1
            assert (np.all(ass >= 0))
            assert (np.all(ass <= 1))
            # should be equal (zero variance) within metastable sets
            assert (np.std(ass[:30]) == 0)
            assert (np.std(ass[40:]) == 0)
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
            # should roughly add up to stationary:
            cgdist = np.array([msm.stationary_distribution[msm.metastable_sets[0]].sum(), msm.stationary_distribution[msm.metastable_sets[1]].sum()])
            ds = cgdist[0]*pccadist[0] + cgdist[1]*pccadist[1]
            ds /= ds.sum()
            assert (np.max(np.abs(ds - msm.stationary_distribution)) < 0.001)
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
        e = msm.expectation(list(range(msm.nstates)))
        # approximately equal for both
        assert (np.abs(e - 31.73) < 0.01)

    def test_expectation(self):
        self._expectation(self.msmrev)
        self._expectation(self.msm)
        self._expectation(self.msmrev_sparse)
        self._expectation(self.msm_sparse)

    def _correlation(self, msm):
        if msm.is_sparse:
            k = 4
        else:
            k = msm.nstates
        # raise assertion error because size is wrong:
        maxtime = 100000
        a = [1, 2, 3]
        with self.assertRaises(AssertionError):
            msm.correlation(a, 1)
        # should decrease
        a = list(range(msm.nstates))
        times, corr1 = msm.correlation(a, maxtime=maxtime)
        assert (len(corr1) == maxtime / msm.lagtime)
        assert (len(times) == maxtime / msm.lagtime)
        assert (corr1[0] > corr1[-1])
        a = list(range(msm.nstates))
        times, corr2 = msm.correlation(a, a, maxtime=maxtime, k=k)
        # should be identical to autocorr
        assert (np.allclose(corr1, corr2))
        # Test: should be increasing in time
        b = list(range(msm.nstates))[::-1]
        times, corr3 = msm.correlation(a, b, maxtime=maxtime, )
        assert (len(times) == maxtime / msm.lagtime)
        assert (len(corr3) == maxtime / msm.lagtime)
        assert (corr3[0] < corr3[-1])

    def test_correlation(self):
        self._correlation(self.msmrev)
        # self._correlation(self.msm)
        # self._correlation(self.msmrev_sparse)
        # self._correlation(self.msm_sparse)

    def _relaxation(self, msm):
        if msm.is_sparse:
            k = 4
        else:
            k = msm.nstates
        pi_perturbed = (msm.stationary_distribution ** 2)
        pi_perturbed /= pi_perturbed.sum()
        a = list(range(msm.nstates))
        maxtime = 100000
        times, rel1 = msm.relaxation(msm.stationary_distribution, a, maxtime=maxtime, k=k)
        # should be constant because we are in equilibrium
        assert (np.allclose(rel1 - rel1[0], np.zeros((np.shape(rel1)[0]))))
        times, rel2 = msm.relaxation(pi_perturbed, a, maxtime=maxtime, k=k)
        # should relax
        assert (len(times) == maxtime / msm.lagtime)
        assert (len(rel2) == maxtime / msm.lagtime)
        assert (rel2[0] < rel2[-1])

    def test_relaxation(self):
        self._relaxation(self.msmrev)
        self._relaxation(self.msm)
        self._relaxation(self.msmrev_sparse)
        self._relaxation(self.msm_sparse)

    def _fingerprint_correlation(self, msm):
        if msm.is_sparse:
            k = 4
        else:
            k = msm.nstates

        if msm.is_reversible:
            # raise assertion error because size is wrong:
            a = [1, 2, 3]
            with self.assertRaises(AssertionError):
                msm.fingerprint_correlation(a, 1, k=k)
            # should decrease
            a = list(range(self.msm.nstates))
            fp1 = msm.fingerprint_correlation(a, k=k)
            # first timescale is infinite
            assert (fp1[0][0] == np.inf)
            # next timescales are identical to timescales:
            assert (np.allclose(fp1[0][1:], msm.timescales(k-1)))
            # all amplitudes nonnegative (for autocorrelation)
            assert (np.all(fp1[1][:] >= 0))
            # identical call
            b = list(range(msm.nstates))
            fp2 = msm.fingerprint_correlation(a, b, k=k)
            assert (np.allclose(fp1[0], fp2[0]))
            assert (np.allclose(fp1[1], fp2[1]))
            # should be - of the above, apart from the first
            b = list(range(msm.nstates))[::-1]
            fp3 = msm.fingerprint_correlation(a, b, k=k)
            assert (np.allclose(fp1[0], fp3[0]))
            assert (np.allclose(fp1[1][1:], -fp3[1][1:]))
        else:  # raise ValueError, because fingerprints are not defined for nonreversible
            with self.assertRaises(ValueError):
                a = list(range(self.msm.nstates))
                msm.fingerprint_correlation(a, k=k)
            with self.assertRaises(ValueError):
                a = list(range(self.msm.nstates))
                b = list(range(msm.nstates))
                msm.fingerprint_correlation(a, b, k=k)

    def test_fingerprint_correlation(self):
        self._fingerprint_correlation(self.msmrev)
        self._fingerprint_correlation(self.msm)
        self._fingerprint_correlation(self.msmrev_sparse)
        self._fingerprint_correlation(self.msm_sparse)

    def _fingerprint_relaxation(self, msm):
        if msm.is_sparse:
            k = 4
        else:
            k = msm.nstates

        if msm.is_reversible:
            # raise assertion error because size is wrong:
            a = [1, 2, 3]
            with self.assertRaises(AssertionError):
                msm.fingerprint_relaxation(msm.stationary_distribution, a, k=k)
            # equilibrium relaxation should be constant
            a = list(range(msm.nstates))
            fp1 = msm.fingerprint_relaxation(msm.stationary_distribution, a, k=k)
            # first timescale is infinite
            assert (fp1[0][0] == np.inf)
            # next timescales are identical to timescales:
            assert (np.allclose(fp1[0][1:], msm.timescales(k-1)))
            # dynamical amplitudes should be near 0 because we are in equilibrium
            assert (np.max(np.abs(fp1[1][1:])) < 1e-10)
            # off-equilibrium relaxation
            pi_perturbed = (msm.stationary_distribution ** 2)
            pi_perturbed /= pi_perturbed.sum()
            fp2 = msm.fingerprint_relaxation(pi_perturbed, a, k=k)
            # first timescale is infinite
            assert (fp2[0][0] == np.inf)
            # next timescales are identical to timescales:
            assert (np.allclose(fp2[0][1:], msm.timescales(k-1)))
            # dynamical amplitudes should be significant because we are not in equilibrium
            assert (np.max(np.abs(fp2[1][1:])) > 0.1)
        else:  # raise ValueError, because fingerprints are not defined for nonreversible
            with self.assertRaises(ValueError):
                a = list(range(self.msm.nstates))
                msm.fingerprint_relaxation(msm.stationary_distribution, a, k=k)
            with self.assertRaises(ValueError):
                pi_perturbed = (msm.stationary_distribution ** 2)
                pi_perturbed /= pi_perturbed.sum()
                a = list(range(self.msm.nstates))
                msm.fingerprint_relaxation(pi_perturbed, a)

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
        self._active_state_indexes(self.msmrevpi)
        self._active_state_indexes(self.msm)
        self._active_state_indexes(self.msmrev_sparse)
        self._active_state_indexes(self.msmrevpi_sparse)
        self._active_state_indexes(self.msm_sparse)

    def _generate_traj(self, msm):
        T = 10
        gt = msm.generate_traj(T)
        # Test: should have the right dimension
        assert (np.all(gt.shape == (T, 2)))
        # itraj should be right
        assert (np.all(gt[:, 0] == 0))

    def test_generate_traj(self):
        self._generate_traj(self.msmrev)
        self._generate_traj(self.msmrevpi)
        self._generate_traj(self.msm)
        with warnings.catch_warnings(record=True) as w:
            self._generate_traj(self.msmrev_sparse)
        with warnings.catch_warnings(record=True) as w:
            self._generate_traj(self.msmrevpi_sparse)
        with warnings.catch_warnings(record=True) as w:
            self._generate_traj(self.msm_sparse)

    def _sample_by_state(self, msm):
        nsample = 100
        ss = msm.sample_by_state(nsample)
        # must have the right size
        assert (len(ss) == msm.nstates)
        # must be correctly assigned
        dtraj_active = msm.discrete_trajectories_active[0]
        for i, samples in enumerate(ss):
            # right shape
            assert (np.all(samples.shape == (nsample, 2)))
            for row in samples:
                assert (row[0] == 0)  # right trajectory
                assert (dtraj_active[row[1]] == i)

    def test_sample_by_state(self):
        self._sample_by_state(self.msmrev)
        self._sample_by_state(self.msmrevpi)
        self._sample_by_state(self.msm)
        self._sample_by_state(self.msmrev_sparse)
        self._sample_by_state(self.msmrevpi_sparse)
        self._sample_by_state(self.msm_sparse)

    def _trajectory_weights(self, msm):
        W = msm.trajectory_weights()
        # should sum to 1
        assert (np.abs(np.sum(W[0]) - 1.0) < 1e-6)

    def test_trajectory_weights(self):
        self._trajectory_weights(self.msmrev)
        self._trajectory_weights(self.msmrevpi)
        self._trajectory_weights(self.msm)
        self._trajectory_weights(self.msmrev_sparse)
        self._trajectory_weights(self.msmrevpi_sparse)
        self._trajectory_weights(self.msm_sparse)

    def test_simulate_MSM(self):
        msm = self.msm
        N=400
        start=1
        traj = msm.simulate(N=N, start=start)
        assert (len(traj) <= N)
        assert (len(np.unique(traj)) <= len(msm.transition_matrix))
        assert (start == traj[0])

    # ----------------------------------
    # MORE COMPLEX TESTS / SANITY CHECKS
    # ----------------------------------

    def _two_state_kinetics(self, msm, eps=0.001):
        if msm.is_sparse:
            k = 4
        else:
            k = msm.nstates
        # sanity check: k_forward + k_backward = 1.0/t2 for the two-state process
        l2 = msm.eigenvectors_left(k)[1, :]
        core1 = np.argmin(l2)
        core2 = np.argmax(l2)
        # transition time from left to right and vice versa
        t12 = msm.mfpt(core1, core2)
        t21 = msm.mfpt(core2, core1)
        # relaxation time
        t2 = msm.timescales(k)[0]
        # the following should hold roughly = k12 + k21 = k2.
        # sum of forward/backward rates can be a bit smaller because we are using small cores and
        # therefore underestimate rates
        ksum = 1.0 / t12 + 1.0 / t21
        k2 = 1.0 / t2
        assert (np.abs(k2 - ksum) < eps)

    def test_two_state_kinetics(self):
        self._two_state_kinetics(self.msmrev)
        self._two_state_kinetics(self.msmrevpi)
        self._two_state_kinetics(self.msm)
        self._two_state_kinetics(self.msmrev_sparse)
        self._two_state_kinetics(self.msmrevpi_sparse)
        self._two_state_kinetics(self.msm_sparse)


class TestMSMMinCountConnectivity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dtraj = np.array(
            [0, 3, 0, 1, 2, 3, 0, 0, 1, 0, 1, 0, 3, 1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 3, 0, 0, 3, 3, 0, 0, 1, 1, 3, 0,
             1, 0, 0, 1, 0, 0, 0, 0, 3, 0, 1, 0, 3, 2, 1, 0, 3, 1, 0, 1, 0, 1, 0, 3, 0, 0, 3, 0, 0, 0, 2, 0, 0, 3,
             0, 1, 0, 0, 0, 0, 3, 3, 3, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 3, 3, 1, 0, 0, 0, 2, 1, 3, 0, 0])
        assert (dtraj == 2).sum() == 5 # state 2 has only 5 counts,
        cls.dtraj = dtraj
        cls.mincount_connectivity = 6 # state 2 will be kicked out by this choice.
        cls.active_set_unrestricted = np.array([0, 1, 2, 3])
        cls.active_set_restricted = np.array([0, 1, 3])

    def _test_connectivity(self, msm, msm_mincount):
        np.testing.assert_equal(msm.active_set, self.active_set_unrestricted)
        np.testing.assert_equal(msm_mincount.active_set, self.active_set_restricted)

    def test_msm(self):
        msm_one_over_n = estimate_markov_model(self.dtraj, lag=1, mincount_connectivity='1/n')
        msm_restrict_connectivity = estimate_markov_model(self.dtraj, lag=1,
                                                          mincount_connectivity=self.mincount_connectivity)
        self._test_connectivity(msm_one_over_n, msm_restrict_connectivity)

    def test_bmsm(self):
        from pyemma.msm import bayesian_markov_model
        msm = bayesian_markov_model(self.dtraj, lag=1, mincount_connectivity='1/n')
        msm_restricted = bayesian_markov_model(self.dtraj, lag=1, mincount_connectivity=self.mincount_connectivity)
        self._test_connectivity(msm, msm_restricted)

    @unittest.skip("""
      File "/home/marscher/workspace/pyemma/pyemma/msm/estimators/_OOM_MSM.py", line 260, in oom_components
    omega = np.real(R[:, 0])
IndexError: index 0 is out of bounds for axis 1 with size 0
    """)
    def test_oom(self):
        from pyemma import msm
        msm_one_over_n = msm.estimate_markov_model(self.dtraj, lag=1, mincount_connectivity='1/n', weights='oom')

        # we now restrict the connectivity to have at least 6 counts, so we will loose state 2
        msm_restrict_connectivity = msm.estimate_markov_model(self.dtraj, lag=1, mincount_connectivity=6, weights='oom')
        self._test_connectivity(msm_one_over_n, msm_restrict_connectivity)

    def test_timescales(self):
        from pyemma.msm import timescales_msm
        its = timescales_msm(self.dtraj, lags=[1, 2], mincount_connectivity=0, errors=None)
        assert its.estimator.mincount_connectivity == 0


if __name__ == "__main__":
    unittest.main()
