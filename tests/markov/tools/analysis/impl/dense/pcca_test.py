"""
Created on 06.12.2013

@author: jan-hendrikprinz

This module provides the unittest for the pcca module

"""
import unittest

import numpy as np

from deeptime.markov.tools.analysis import stationary_distribution
from deeptime.markov.tools.analysis.dense._pcca import pcca, coarsegrain, PCCA
from deeptime.markov.tools.estimation import connected_sets
from tests.markov.tools.numeric import assert_allclose


class TestPCCA(unittest.TestCase):
    def test_pcca_no_transition_matrix(self):
        P = np.array([[1.0, 1.0],
                      [0.1, 0.9]])

        with self.assertRaises(ValueError):
            pcca(P, 2)

    def test_pcca_no_detailed_balance(self):
        P = np.array([[0.8, 0.1, 0.1],
                      [0.3, 0.2, 0.5],
                      [0.6, 0.3, 0.1]])
        with self.assertRaises(ValueError):
            pcca(P, 2)

    def test_pcca_1(self):
        P = np.array([[1, 0],
                      [0, 1]])
        chi = pcca(P, 2)
        sol = np.array([[1., 0.],
                        [0., 1.]])
        assert_allclose(chi, sol)

    def test_pcca_2(self):
        P = np.array([[0.0, 1.0, 0.0],
                      [0.0, 0.999, 0.001],
                      [0.0, 0.001, 0.999]])
        chi = pcca(P, 2)
        sol = np.array([[1., 0.],
                        [1., 0.],
                        [0., 1.]])
        assert_allclose(chi, sol)

    def test_pcca_3(self):
        P = np.array([[0.9, 0.1, 0.0, 0.0],
                      [0.1, 0.9, 0.0, 0.0],
                      [0.0, 0.0, 0.8, 0.2],
                      [0.0, 0.0, 0.2, 0.8]])
        # n=2
        chi = pcca(P, 2)
        sol = np.array([[1., 0.],
                        [1., 0.],
                        [0., 1.],
                        [0., 1.]])
        assert_allclose(chi, sol)
        # n=3
        chi = pcca(P, 3)
        sol = np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.],
                        [0., 0., 1.]])
        assert_allclose(chi, sol)
        # n=4
        chi = pcca(P, 4)
        sol = np.array([[1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]])
        assert_allclose(chi, sol)

    def test_pcca_4(self):
        P = np.array([[0.9, 0.1, 0.0, 0.0],
                      [0.1, 0.8, 0.1, 0.0],
                      [0.0, 0.0, 0.8, 0.2],
                      [0.0, 0.0, 0.2, 0.8]])
        chi = pcca(P, 2)
        sol = np.array([[1., 0.],
                        [1., 0.],
                        [1., 0.],
                        [0., 1.]])
        assert_allclose(chi, sol)

    def test_pcca_5(self):
        P = np.array([[0.9, 0.1, 0.0, 0.0, 0.0],
                      [0.1, 0.9, 0.0, 0.0, 0.0],
                      [0.0, 0.1, 0.8, 0.1, 0.0],
                      [0.0, 0.0, 0.0, 0.8, 0.2],
                      [0.0, 0.0, 0.0, 0.2, 0.8]])
        # n=2
        chi = pcca(P, 2)
        sol = np.array([[1., 0.],
                        [1., 0.],
                        [0.5, 0.5],
                        [0., 1.],
                        [0., 1.]])
        assert_allclose(chi, sol)
        # n=4
        chi = pcca(P, 4)
        sol = np.array([[1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0.5, 0.5, 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]])
        assert_allclose(chi, sol)

    def test_pcca_large(self):
        import os

        P = np.loadtxt(os.path.split(__file__)[0] + '/../../P_rev_251x251.dat')
        # n=2
        chi = pcca(P, 2)
        assert (np.alltrue(chi >= 0))
        assert (np.alltrue(chi <= 1))
        # n=3
        chi = pcca(P, 3)
        assert (np.alltrue(chi >= 0))
        assert (np.alltrue(chi <= 1))
        # n=4
        chi = pcca(P, 4)
        assert (np.alltrue(chi >= 0))
        assert (np.alltrue(chi <= 1))

    def test_pcca_coarsegrain(self):
        # fine-grained transition matrix
        P = np.array([[0.9,  0.1,  0.0,  0.0,  0.0],
                      [0.1,  0.89, 0.01, 0.0,  0.0],
                      [0.0,  0.1,  0.8,  0.1,  0.0],
                      [0.0,  0.0,  0.01, 0.79, 0.2],
                      [0.0,  0.0,  0.0,  0.2,  0.8]])
        from deeptime.markov.tools.analysis import stationary_distribution
        pi = stationary_distribution(P)
        Pi = np.diag(pi)
        m = 3
        # Susanna+Marcus' expression ------------
        M = pcca(P, m)
        pi_c = np.dot(M.T, pi)
        Pi_c_inv = np.diag(1.0/pi_c)
        # restriction and interpolation operators
        R = M.T
        I = np.dot(np.dot(Pi, M), Pi_c_inv)
        # result
        ms1 = np.linalg.inv(np.dot(R,I)).T
        ms2 = np.dot(np.dot(I.T, P), R.T)
        Pc_ref = np.dot(ms1,ms2)
        # ---------------------------------------

        Pc = coarsegrain(P, 3)
        # test against Marcus+Susanna's expression
        assert np.max(np.abs(Pc - Pc_ref)) < 1e-10
        # test mass conservation
        assert np.allclose(Pc.sum(axis=1), np.ones(m))

        p = PCCA(P, m)
        # test against Marcus+Susanna's expression
        assert np.max(np.abs(p.coarse_grained_transition_matrix - Pc_ref)) < 1e-10
        # test against the present coarse-grained stationary dist
        assert np.max(np.abs(p.coarse_grained_stationary_probability - pi_c)) < 1e-10
        # test mass conservation
        assert np.allclose(p.coarse_grained_transition_matrix.sum(axis=1), np.ones(m))

    def test_multiple_components(self):
        L = np.array([[0, 1, 1, 0, 0, 0, 0],
                      [1, 0, 1, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0]])

        transition_matrix = L / np.sum(L, 1).reshape(-1, 1)
        pi = np.zeros((transition_matrix.shape[0],))
        for cs in connected_sets(transition_matrix):
            P_sub = transition_matrix[cs, :][:, cs]
            pi[cs] = stationary_distribution(P_sub)
        chi = pcca(transition_matrix, 2, pi=pi)
        expected = np.array([[0., 1.],
                             [0., 1.],
                             [0., 1.],
                             [1., 0.],
                             [1., 0.],
                             [1., 0.],
                             [1., 0.]])
        np.testing.assert_equal(chi, expected)
