"""Unit test for the reaction pathway decomposition

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import warnings
import unittest
import numpy as np
from scipy.sparse import csr_matrix

from tests.markov.tools.numeric import assert_allclose
from deeptime.markov.tools.flux import pathways


class TestPathways(unittest.TestCase):

    def setUp(self):
        """Small flux-network"""
        F = np.zeros((8, 8))
        F[0, 2] = 10.0
        F[2, 6] = 10.0
        F[1, 3] = 100.0
        F[3, 4] = 30.0
        F[3, 5] = 70.0
        F[4, 6] = 5.0
        F[4, 7] = 25.0
        F[5, 6] = 30.0
        F[5, 7] = 40.0
        """Reactant and product states"""
        A = [0, 1]
        B = [6, 7]

        self.F = F
        self.F_sparse = csr_matrix(F)
        self.A = A
        self.B = B
        self.paths = []
        self.capacities = []
        p1 = np.array([1, 3, 5, 7])
        c1 = 40.0
        self.paths.append(p1)
        self.capacities.append(c1)
        p2 = np.array([1, 3, 5, 6])
        c2 = 30.0
        self.paths.append(p2)
        self.capacities.append(c2)
        p3 = np.array([1, 3, 4, 7])
        c3 = 25.0
        self.paths.append(p3)
        self.capacities.append(c3)
        p4 = np.array([0, 2, 6])
        c4 = 10.0
        self.paths.append(p4)
        self.capacities.append(c4)
        p5 = np.array([1, 3, 4, 6])
        c5 = 5.0
        self.paths.append(p5)
        self.capacities.append(c5)

    def test_pathways_dense(self):
        paths, capacities = pathways(self.F, self.A, self.B)
        self.assertTrue(len(paths) == len(self.paths))
        self.assertTrue(len(capacities) == len(self.capacities))

        for i in range(len(paths)):
            assert_allclose(paths[i], self.paths[i])
            assert_allclose(capacities[i], self.capacities[i])

    def test_pathways_dense_incomplete(self):
        paths, capacities = pathways(self.F, self.A, self.B, fraction=0.5)
        self.assertTrue(len(paths) == len(self.paths[0:2]))
        self.assertTrue(len(capacities) == len(self.capacities[0:2]))

        for i in range(len(paths)):
            assert_allclose(paths[i], self.paths[i])
            assert_allclose(capacities[i], self.capacities[i])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore')
            warnings.simplefilter('always', category=RuntimeWarning)
            paths, capacities = pathways(self.F, self.A, self.B, fraction=1.0, maxiter=1)
            for i in range(len(paths)):
                assert_allclose(paths[i], self.paths[i])
                assert_allclose(capacities[i], self.capacities[i])
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)

    def test_pathways_sparse(self):
        paths, capacities = pathways(self.F_sparse, self.A, self.B)
        self.assertTrue(len(paths) == len(self.paths))
        self.assertTrue(len(capacities) == len(self.capacities))

        for i in range(len(paths)):
            assert_allclose(paths[i], self.paths[i])
            assert_allclose(capacities[i], self.capacities[i])

    def test_pathways_sparse_incomplete(self):
        paths, capacities = pathways(self.F_sparse, self.A, self.B, fraction=0.5)
        self.assertTrue(len(paths) == len(self.paths[0:2]))
        self.assertTrue(len(capacities) == len(self.capacities[0:2]))

        for i in range(len(paths)):
            assert_allclose(paths[i], self.paths[i])
            assert_allclose(capacities[i], self.capacities[i])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            paths, capacities = pathways(self.F, self.A, self.B, fraction=1.0, maxiter=1)
            for i in range(len(paths)):
                assert_allclose(paths[i], self.paths[i])
                assert_allclose(capacities[i], self.capacities[i])
            assert issubclass(w[-1].category, RuntimeWarning)

    def test_with_almost_converged_stat_dist(self):
        """ test for #106 """
        from deeptime.markov.tools.analysis import committor, is_reversible
        from deeptime.markov.tools.flux import flux_matrix, to_netflux
        from deeptime.markov import reactive_flux, ReactiveFlux

        T = np.array([[0.2576419223095193, 0.2254214623509954, 0.248270708174756,
                       0.2686659071647294],
                      [0.2233847186210225, 0.2130434781715344, 0.2793477268264001,
                       0.284224076381043],
                      [0.2118717275169231, 0.2405661227681972, 0.2943396213976011,
                       0.2532225283172787],
                      [0.2328617711043517, 0.2485926610067547, 0.2571819311236834,
                       0.2613636367652102]])
        mu = np.array([0.2306979668517676, 0.2328013892993006, 0.2703312416016573,
                       0.2661694022472743])
        assert is_reversible(T)
        np.testing.assert_allclose(mu.dot(T), mu)
        np.testing.assert_equal(mu.dot(T), T.T.dot(mu))
        A = [0]
        B = [1]

        # forward committor
        qplus = committor(T, A, B, forward=True, mu=mu)
        # backward committor
        if is_reversible(T, mu=mu):
            qminus = 1.0 - qplus
        else:
            qminus = committor(T, A, B, forward=False, mu=mu)

        tpt_obj = reactive_flux(T, A, B)
        tpt_obj.major_flux(1.0)
        # gross flux
        grossflux = flux_matrix(T, mu, qminus, qplus, netflux=False)
        # net flux
        netflux = to_netflux(grossflux)

        F = ReactiveFlux(A, B, netflux, stationary_distribution=mu, qminus=qminus, qplus=qplus, gross_flux=grossflux)
        F.pathways(1.0)
