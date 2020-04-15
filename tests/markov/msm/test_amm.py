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


r"""Unit test for the AMM module

.. moduleauthor:: S. Olsson <solsson AT zedat DOT fu DASH berlin DOT de>

"""

import unittest

import numpy as np
import warnings

from numpy.testing import *
from msmtools.util.birth_death_chain import BirthDeathChain

from sktime.markov import TransitionCountEstimator
from sktime.markov.msm import MarkovStateModel
from sktime.markov.msm.augmented_msm import AugmentedMSMEstimator


def test_amm_sanity(fixed_seed):
    # Meta-stable birth-death chain
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
    dtraj = MarkovStateModel(P).simulate(N=10000, start=0)
    tau = 1

    k = 3
    # Predictions and experimental data
    E = np.vstack((np.linspace(-0.1, 1., 7), np.linspace(1.5, -0.1, 7))).T
    m = np.array([0.0, 0.0])
    w = np.array([2.0, 2.5])
    sigmas = 1. / np.sqrt(2) / np.sqrt(w)

    """ Feature trajectory """
    ftraj = E[dtraj, :]

    amm_estimator = AugmentedMSMEstimator(E=E, m=m, w=w)
    counts = TransitionCountEstimator(lagtime=tau, count_mode="sliding").fit(dtraj).fetch_model()

    amm = amm_estimator.fit(counts).fetch_model()
    amm_convenience_estimator = AugmentedMSMEstimator.estimator_from_feature_trajectories(
        dtraj, ftraj, n_states=counts.n_states_full, m=m, sigmas=sigmas)
    amm_convenience = amm_convenience_estimator.fit(counts).fetch_model()
    assert_equal(tau, amm.lagtime)
    assert_array_almost_equal(E, amm_estimator.expectations_by_state)
    assert_array_almost_equal(E, amm_convenience_estimator.expectations_by_state, decimal=4)
    assert_array_almost_equal(m, amm_estimator.experimental_measurements)
    assert_array_almost_equal(m, amm_convenience_estimator.experimental_measurements)
    assert_array_almost_equal(w, amm_estimator.experimental_measurement_weights)
    assert_array_almost_equal(w, amm_convenience_estimator.experimental_measurement_weights)
    assert_array_almost_equal(amm.transition_matrix, amm_convenience.transition_matrix, decimal=5)
    assert_array_almost_equal(amm.stationary_distribution, amm_convenience.stationary_distribution)
    assert_array_almost_equal(amm.optimizer_state.lagrange, amm_convenience.optimizer_state.lagrange)


class TestAMMSimple(unittest.TestCase):
    def setUp(self):
        """Store state of the rng"""
        self.state = np.random.mtrand.get_state()

        """Reseed the rng to enforce 'deterministic' behavior"""
        np.random.mtrand.seed(0xDEADBEEF)

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

        self.k = 3
        """ Predictions and experimental data """
        self.E = np.vstack((np.linspace(-0.1, 1., 7), np.linspace(1.5, -0.1, 7))).T
        self.m = np.array([0.0, 0.0])
        self.w = np.array([2.0, 2.5])
        self.sigmas = 1. / np.sqrt(2) / np.sqrt(self.w)

        """ Feature trajectory """
        self.ftraj = self.E[self.dtraj, :]

        self.AMM = AugmentedMarkovModel(E=self.E, m=self.m, w=self.w)
        self.AMM.estimate([self.dtraj])

    def tearDown(self):
        """Revert the state of the rng"""
        np.random.mtrand.set_state(self.state)

    def test_AMM(self):
        """ self-consistency, explicit class instantiation/estimation and convienence function """
        amm = estimate_augmented_markov_model([self.dtraj], [self.ftraj], self.tau, self.m, self.sigmas)
        assert_allclose(self.dtraj, amm.discrete_trajectories_full[0])
        self.assertEqual(self.tau, amm.lagtime)
        self.assertTrue(np.allclose(self.E, amm.E))
        self.assertTrue(np.allclose(self.m, amm.m))
        self.assertTrue(np.allclose(self.w, amm.w))
        self.assertTrue(np.allclose(self.AMM.P, amm.P))
        self.assertTrue(np.allclose(self.AMM.pi, amm.pi))
        self.assertTrue(np.allclose(self.AMM.lagrange, amm.lagrange))


class TestAMMDoubleWell(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyemma.datasets
        cls.dtraj = pyemma.datasets.load_2well_discrete().dtraj_T100K_dt10
        cls.E_ = np.linspace(0.01, 2. * np.pi, 66).reshape(-1, 1) ** (0.5)
        cls.m = np.array([1.9])
        cls.w = np.array([2.0])
        cls.sigmas = 1. / np.sqrt(2) / np.sqrt(cls.w)
        _sd = list(set(cls.dtraj))

        cls.ftraj = cls.E_[[_sd.index(d) for d in cls.dtraj], :]
        cls.tau = 10
        cls.amm = estimate_augmented_markov_model([cls.dtraj], [cls.ftraj], cls.tau, cls.m, cls.sigmas)
        cls.msm = cls.amm

    def test_reversible(cls):
        pass

    def test_score_cv(cls):
        pass

    def test_score(cls):
        pass

    def test_sparse(cls):
        pass

    # ---------------------------------
    # BASIC PROPERTIES
    # ---------------------------------

    def test_lagtime(self):
        self._lagtime(self.amm)

    def test_active_set(self):
        self._active_set(self.amm)

    def test_largest_connected_set(self):
        self._largest_connected_set(self.amm)

    def test_nstates(self):
        self._nstates(self.amm)

    def test_connected_sets(self):
        self._connected_sets(self.amm)

    def test_connectivity(self):
        self._connectivity(self.amm)

    def test_count_matrix_active(self):
        self._count_matrix_active(self.amm)

    def test_count_matrix_full(self):
        self._count_matrix_full(self.amm)

    def test_discrete_trajectories_full(self):
        self._discrete_trajectories_full(self.amm)

    def test_discrete_trajectories_active(self):
        self._discrete_trajectories_active(self.amm)

    def test_timestep(self):
        self._timestep(self.amm)

    def test_transition_matrix(self):
        self._transition_matrix(self.amm)

    def test_dt_model(self):
        self._dt_model(self.amm)

    # ---------------------------------
    # SIMPLE STATISTICS
    # ---------------------------------

    def test_active_count_fraction(self):
        self._active_count_fraction(self.amm)

    def test_active_state_fraction(self):
        # should always be a fraction
        self._active_state_fraction(self.amm)

    def test_effective_count_matrix(self):
        self._effective_count_matrix(self.amm)

    # ---------------------------------
    # EIGENVALUES, EIGENVECTORS
    # ---------------------------------

    def test_statdist(self):
        self._statdist(self.amm)

    def test_eigenvalues(self):
        self._eigenvalues(self.amm)

    def test_eigenvectors_left(self):
        self._eigenvectors_left(self.amm)

    def test_eigenvectors_right(self):
        self._eigenvectors_right(self.amm)

    def test_eigenvectors_RDL(self):
        self._eigenvectors_RDL(self.amm)

    def _timescales(self, amm):
        if not amm.is_sparse:
            if not amm.is_reversible:
                with warnings.catch_warnings(record=True) as w:
                    ts = amm.timescales()
            else:
                ts = amm.timescales()
        else:
            k = 4
            if not amm.is_reversible:
                with warnings.catch_warnings(record=True) as w:
                    ts = amm.timescales(k)
            else:
                ts = amm.timescales(k)

        # should be all positive
        assert (np.all(ts > 0))
        # REVERSIBLE: should be all real
        if amm.is_reversible:
            ts_ref = np.array([270.83, 8.77, 5.21])
            assert (np.all(np.isreal(ts)))
            # HERE:
            np.testing.assert_almost_equal(ts[:3], ts_ref, decimal=2)
        else:
            ts_ref = np.array([270.83, 8.77, 5.21])
            # HERE:
            np.testing.assert_almost_equal(ts[:3], ts_ref, decimal=2)

    def test_timescales(self):
        self._timescales(self.amm)

    # ---------------------------------
    # FIRST PASSAGE PROBLEMS
    # ---------------------------------

    def test_committor(self):
        self._committor(self.amm)

    def _mfpt(self, amm):
        a = 16
        b = 48
        t = amm.mfpt(a, b)
        assert (t > 0)
        # HERE:
        np.testing.assert_allclose(t, 546.81, rtol=1e-3, atol=1e-6)

    def test_mfpt(self):
        self._mfpt(self.amm)

    # ---------------------------------
    # PCCA
    # ---------------------------------

    def test_pcca_assignment(self):
        self._pcca_assignment(self.amm)

    def test_pcca_distributions(self):
        self._pcca_distributions(self.amm)

    def test_pcca_memberships(self):
        self._pcca_memberships(self.amm)

    def test_pcca_sets(self):
        self._pcca_sets(self.amm)

    # ---------------------------------
    # EXPERIMENTAL STUFF
    # ---------------------------------

    def _expectation(self, amm):
        e = amm.expectation(list(range(amm.nstates)))
        # approximately equal for both
        assert (np.abs(e - 39.02) < 0.01)

    def test_expectation(self):
        self._expectation(self.amm)

    def test_correlation(self):
        self._correlation(self.amm)

    def _relaxation(self, amm):
        if amm.is_sparse:
            k = 4
        else:
            k = amm.nstates
        pi_perturbed = (amm.stationary_distribution ** 2)
        pi_perturbed /= pi_perturbed.sum()
        a = list(range(amm.nstates))[::-1]
        maxtime = 100000
        times, rel1 = amm.relaxation(amm.stationary_distribution, a, maxtime=maxtime, k=k)
        # should be constant because we are in equilibrium
        assert (np.allclose(rel1 - rel1[0], np.zeros((np.shape(rel1)[0]))))
        times, rel2 = amm.relaxation(pi_perturbed, a, maxtime=maxtime, k=k)
        # should relax
        assert (len(times) == maxtime / amm.lagtime)
        assert (len(rel2) == maxtime / amm.lagtime)
        assert (rel2[0] < rel2[-1])

    def test_relaxation(self):
        self._relaxation(self.amm)

    def test_fingerprint_correlation(self):
        self._fingerprint_correlation(self.amm)

    def test_fingerprint_relaxation(self):
        self._fingerprint_relaxation(self.amm)

    # ---------------------------------
    # STATISTICS, SAMPLING
    # ---------------------------------

    def test_active_state_indexes(self):
        self._active_state_indexes(self.amm)

    def test_generate_traj(self):
        self._generate_traj(self.amm)

    def test_sample_by_state(self):
        self._sample_by_state(self.amm)

    def test_trajectory_weights(self):
        self._trajectory_weights(self.amm)

    def test_simulate_MSM(self):
        amm = self.amm
        N = 400
        start = 1
        traj = amm.simulate(N=N, start=start)
        assert (len(traj) <= N)
        assert (len(np.unique(traj)) <= len(amm.transition_matrix))
        assert (start == traj[0])

    # ----------------------------------
    # MORE COMPLEX TESTS / SANITY CHECKS
    # ----------------------------------

    def test_two_state_kinetics(self):
        self._two_state_kinetics(self.amm, eps=0.01)

    def test_serialize(self):
        import tempfile
        import pyemma
        f = tempfile.mktemp()
        try:
            self.amm.save(f)
            restored = pyemma.load(f)

            # check estimation parameters
            np.testing.assert_equal(self.amm.lag, restored.lag)
            np.testing.assert_equal(self.amm.count_mode, restored.count_mode)
            np.testing.assert_equal(self.amm.connectivity, restored.connectivity)
            np.testing.assert_equal(self.amm.dt_traj, restored.dt_traj)
            np.testing.assert_equal(self.amm.E, restored.E)
            np.testing.assert_equal(self.amm.m, restored.m)
            np.testing.assert_equal(self.amm.w, restored.w)
            np.testing.assert_equal(self.amm.eps, restored.eps)
            np.testing.assert_equal(self.amm.support_ci, restored.support_ci)
            np.testing.assert_equal(self.amm.maxiter, restored.maxiter)
            np.testing.assert_equal(self.amm.max_cache, restored.max_cache)
            np.testing.assert_equal(self.amm.mincount_connectivity, restored.mincount_connectivity)

            # ensure we got the estimated quantities right
            np.testing.assert_equal(self.amm.E_active, restored.E_active)
            np.testing.assert_equal(self.amm.E_min, restored.E_min)
            np.testing.assert_equal(self.amm.E_max, restored.E_max)
            np.testing.assert_equal(self.amm.mhat, restored.mhat)
            np.testing.assert_equal(self.amm.lagrange, restored.lagrange)
            np.testing.assert_equal(self.amm.sigmas, restored.sigmas)
            np.testing.assert_equal(self.amm.count_inside, restored.count_inside)
            np.testing.assert_equal(self.amm.count_outside, restored.count_outside)
            # derived from msm_estimator
            np.testing.assert_equal(self.amm.P, restored.P)
            np.testing.assert_equal(self.amm.pi, restored.pi)
        finally:
            import os
            os.unlink(f)


# avoid parent class being executed as additional test case
class TestCoreAMMDoubleWell(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyemma.datasets
        cls.core_set = [34, 65]
        cls.dtraj = pyemma.datasets.load_2well_discrete().dtraj_T100K_dt10
        cls.E_ = np.linspace(0.01, 2. * np.pi, 66).reshape(-1, 1) ** (0.5)
        cls.m = np.array([1.9])
        cls.w = np.array([2.0])
        cls.sigmas = 1. / np.sqrt(2) / np.sqrt(cls.w)
        _sd = list(set(cls.dtraj))

        cls.ftraj = cls.E_[[_sd.index(d) for d in cls.dtraj], :]
        cls.tau = 10
        cls.amm = estimate_augmented_markov_model([cls.dtraj], [cls.ftraj], cls.tau, cls.m, cls.sigmas,
                                                  core_set=cls.core_set)
        cls.msm = cls.amm

    def test_reversible(cls):
        pass

    def test_score_cv(cls):
        pass

    def test_score(cls):
        pass

    def test_sparse(cls):
        pass

    # ---------------------------------
    # BASIC PROPERTIES
    # ---------------------------------

    def test_lagtime(self):
        self._lagtime(self.amm)

    def test_active_set(self):
        self._active_set(self.amm)

    def test_largest_connected_set(self):
        self._largest_connected_set(self.amm)

    def test_nstates(self):
        self._nstates(self.amm)

    def test_connected_sets(self):
        self._connected_sets(self.amm)

    def test_connectivity(self):
        self._connectivity(self.amm)

    def test_count_matrix_active(self):
        self._count_matrix_active(self.amm)

    def test_count_matrix_full(self):
        self._count_matrix_full(self.amm)

    def test_discrete_trajectories_full(self):
        self._discrete_trajectories_full(self.amm)

    def test_discrete_trajectories_active(self):
        self._discrete_trajectories_active(self.amm)

    def test_timestep(self):
        self._timestep(self.amm)

    def test_transition_matrix(self):
        self._transition_matrix(self.amm)

    def test_dt_model(self):
        self._dt_model(self.amm)

    # ---------------------------------
    # SIMPLE STATISTICS
    # ---------------------------------

    def test_active_count_fraction(self):
        self._active_count_fraction(self.amm)

    def test_active_state_fraction(self):
        # should always be a fraction
        self._active_state_fraction(self.amm)

    # ---------------------------------
    # EIGENVALUES, EIGENVECTORS
    # ---------------------------------

    def test_statdist(self):
        self._statdist(self.amm)

    def test_eigenvalues(self):
        self._eigenvalues(self.amm)

    def test_eigenvectors_left(self):
        self._eigenvectors_left(self.amm)

    def test_eigenvectors_right(self):
        self._eigenvectors_right(self.amm)

    def test_eigenvectors_RDL(self):
        self._eigenvectors_RDL(self.amm)

    def _timescales(self, amm):
        if not amm.is_sparse:
            if not amm.is_reversible:
                with warnings.catch_warnings(record=True) as w:
                    ts = amm.timescales()
            else:
                ts = amm.timescales()
        else:
            k = 4
            if not amm.is_reversible:
                with warnings.catch_warnings(record=True) as w:
                    ts = amm.timescales(k)
            else:
                ts = amm.timescales(k)

        # should be all positive
        assert (np.all(ts > 0))
        # reference?
        ts_ref = np.array([364.])
        assert (np.all(np.isreal(ts)))

        np.testing.assert_almost_equal(ts, ts_ref, decimal=0)

    def test_timescales(self):
        self._timescales(self.amm)

    # ---------------------------------
    # FIRST PASSAGE PROBLEMS
    # ---------------------------------

    def test_committor(self):
        self._committor(self.amm)

    def _mfpt(self, amm):
        a = 0
        b = 1
        t = amm.mfpt(a, b)
        assert (t > 0)
        # HERE:
        # reference!?
        np.testing.assert_allclose(t, 546.81, rtol=1e-3, atol=50)

    def test_mfpt(self):
        self._mfpt(self.amm)

    # ---------------------------------
    # EXPERIMENTAL STUFF
    # ---------------------------------

    def _expectation(self, amm):
        e = amm.expectation(list(range(amm.nstates)))
        # approximately equal for both
        # reference!?
        assert (np.abs(e - 0.73) < 0.01)

    def test_expectation(self):
        self._expectation(self.amm)

    def test_correlation(self):
        self._correlation(self.amm)

    def _relaxation(self, amm):
        if amm.is_sparse:
            k = 4
        else:
            k = amm.nstates
        pi_perturbed = (amm.stationary_distribution ** 2)
        pi_perturbed /= pi_perturbed.sum()
        a = list(range(amm.nstates))[::-1]
        maxtime = 100000
        times, rel1 = amm.relaxation(amm.stationary_distribution, a, maxtime=maxtime, k=k)
        # should be constant because we are in equilibrium
        assert (np.allclose(rel1 - rel1[0], np.zeros((np.shape(rel1)[0]))))
        times, rel2 = amm.relaxation(pi_perturbed, a, maxtime=maxtime, k=k)
        # should relax
        assert (len(times) == maxtime / amm.lagtime)
        assert (len(rel2) == maxtime / amm.lagtime)
        assert (rel2[0] < rel2[-1])

    def test_relaxation(self):
        self._relaxation(self.amm)

    def test_fingerprint_correlation(self):
        self._fingerprint_correlation(self.amm)

    def test_fingerprint_relaxation(self):
        self._fingerprint_relaxation(self.amm)

    # ---------------------------------
    # STATISTICS, SAMPLING
    # ---------------------------------

    def test_active_state_indexes(self):
        self._active_state_indexes(self.amm)

    def test_generate_traj(self):
        self._generate_traj(self.amm)

    def test_sample_by_state(self):
        self._sample_by_state(self.amm)

    def test_trajectory_weights(self):
        self._trajectory_weights(self.amm)

    def test_simulate_MSM(self):
        amm = self.amm
        N = 400
        start = 1
        traj = amm.simulate(N=N, start=start)
        assert (len(traj) <= N)
        assert (len(np.unique(traj)) <= len(amm.transition_matrix))
        assert (start == traj[0])

    # ----------------------------------
    # MORE COMPLEX TESTS / SANITY CHECKS
    # ----------------------------------

    def test_two_state_kinetics(self):
        self._two_state_kinetics(self.amm, eps=0.01)

    def test_serialize(self):
        import tempfile
        import pyemma
        f = tempfile.mktemp()
        try:
            self.amm.save(f)
            restored = pyemma.load(f)

            # check estimation parameters
            np.testing.assert_equal(self.amm.lag, restored.lag)
            np.testing.assert_equal(self.amm.count_mode, restored.count_mode)
            np.testing.assert_equal(self.amm.connectivity, restored.connectivity)
            np.testing.assert_equal(self.amm.dt_traj, restored.dt_traj)
            np.testing.assert_equal(self.amm.E, restored.E)
            np.testing.assert_equal(self.amm.m, restored.m)
            np.testing.assert_equal(self.amm.w, restored.w)
            np.testing.assert_equal(self.amm.eps, restored.eps)
            np.testing.assert_equal(self.amm.support_ci, restored.support_ci)
            np.testing.assert_equal(self.amm.maxiter, restored.maxiter)
            np.testing.assert_equal(self.amm.max_cache, restored.max_cache)
            np.testing.assert_equal(self.amm.mincount_connectivity, restored.mincount_connectivity)

            # ensure we got the estimated quantities right
            np.testing.assert_equal(self.amm.E_active, restored.E_active)
            np.testing.assert_equal(self.amm.E_min, restored.E_min)
            np.testing.assert_equal(self.amm.E_max, restored.E_max)
            np.testing.assert_equal(self.amm.mhat, restored.mhat)
            np.testing.assert_equal(self.amm.lagrange, restored.lagrange)
            np.testing.assert_equal(self.amm.sigmas, restored.sigmas)
            np.testing.assert_equal(self.amm.count_inside, restored.count_inside)
            np.testing.assert_equal(self.amm.count_outside, restored.count_outside)
            # derived from msm_estimator
            np.testing.assert_equal(self.amm.P, restored.P)
            np.testing.assert_equal(self.amm.pi, restored.pi)
        finally:
            import os
            os.unlink(f)


if __name__ == "__main__":
    unittest.main()
