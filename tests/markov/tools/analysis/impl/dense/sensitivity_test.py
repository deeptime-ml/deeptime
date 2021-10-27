"""
Created on 06.12.2013

@author: Jan-Hendrik Prinz

This module provides unit tests for the sensitivity module

Most tests consist of the comparison of some (randomly selected) sensitivity matrices
against numerical differentiation results.
"""

import unittest
import numpy as np
from tests.markov.tools.numeric import assert_allclose

from deeptime.markov.tools.analysis.dense._sensitivity import timescale_sensitivity, eigenvalue_sensitivity, \
    mfpt_sensitivity, forward_committor_sensitivity, backward_committor_sensitivity, eigenvector_sensitivity, \
    stationary_distribution_sensitivity, expectation_sensitivity


class TestExpectations(unittest.TestCase):
    def setUp(self):
        self.T = np.array([[0.8, 0.2], [0.05, 0.95]])

        self.S0 = np.array([[0.2, 0.2], [0.8, 0.8]])
        self.S1 = np.array([[0.8, -0.2], [-0.8, 0.2]])

        self.TS1 = np.array([[12.8885223, -3.2221306], [-12.8885223, 3.2221306]])

        self.T4 = np.array([[0.9, 0.04, 0.03, 0.03],
                            [0.02, 0.94, 0.02, 0.02],
                            [0.01, 0.01, 0.94, 0.04],
                            [0.01, 0.01, 0.08, 0.9]])

        self.qS42 = np.array([[0., 0., 0., 0.],
                              [0., 1.7301, 2.24913, 2.94118],
                              [0., 10.3806, 13.4948, 17.6471],
                              [0., 0., 0., 0.]])

        self.qS41 = np.array([[0., 0., 0., 0.],
                              [0., 10.3806, 13.4948, 17.6471],
                              [0., 3.46021, 4.49826, 5.88235],
                              [0., 0., 0., 0.]])

        self.qSI41 = np.array(
            [[-0.8370915, -4.4385316, 0.7235325, 2.4042048],
             [-1.4649101, 1.8717024, 2.8727056, 4.2073585],
             [-3.4978465, 15.8788050, 8.7609111, 10.0461416],
             [-1.9432481, 13.1056296, 5.5811895, 5.5811898]]
        )

        self.S2zero = np.zeros((2, 2))
        self.S4zero = np.zeros((4, 4))

        self.mS01 = np.array(
            [[0., 0., 0., 0.],
             [0., 1875., 2187.5, 2187.5],
             [0., 2410.71, 2812.5, 2812.5],
             [0., 1339.29, 1562.5, 1562.5]]
        )

        self.mS02 = np.array(
            [[0., 0., 0., 0.],
             [0., 937.5, 1093.75, 1093.75],
             [0., 3883.93, 4531.25, 4531.25],
             [0., 1741.07, 2031.25, 2031.25]]
        )

        self.mS32 = np.array(
            [[102.959, 114.793, 87.574, 0.],
             [180.178, 200.888, 153.254, 0.],
             [669.231, 746.154, 569.231, 0.],
             [0., 0., 0., 0.]]
        )

        self.mV11 = np.array(
            [[-3.4819290, -6.6712389, 2.3317857, 2.3317857],
             [1.4582191, 2.7938918, -0.9765401, -0.9765414],
             [-0.7824563, -1.4991658, 0.5239938, 0.5239950],
             [-0.2449557, -0.4693191, 0.1640369, 0.1640476]]
        )

        self.mV22 = np.array(
            [[0.0796750, -0.0241440, -0.0057555, -0.0057555],
             [-2.2829491, 0.6918640, 0.1649531, 0.1649531],
             [-5.8183459, 1.7632923, 0.4203993, 0.4203985],
             [16.4965144, -4.9993827, -1.1919380, -1.1919347]]
        )

        self.mV03 = np.array(
            [[1.3513524, 1.3513531, 1.3513533, 1.3513533],
             [2.3648662, 2.3648656, 2.3648655, 2.3648656],
             [-0.6032816, -0.6032783, -0.6032800, -0.6032799],
             [-3.1129331, -3.1129331, -3.1129321, -3.1129312]]
        )

        self.mV01left = np.array(
            [[0.4473028, 2.5148236, -0.8052692, -0.6389904],
             [0.7827807, 4.4009367, -1.4092215, -1.1182336],
             [1.8690916, 10.5083832, -3.3648744, -2.6700682],
             [1.0383831, 5.8379865, -1.8693753, -1.4833698]]
        )

        self.pS1 = np.array(
            [[0.0868655, 1.2556020, -0.3514107, -0.3514107],
             [0.1520148, 2.1973013, -0.6149689, -0.6149689],
             [0.3629750, 5.2466298, -1.4683944, -1.4683955],
             [0.2016525, 2.9147921, -0.8157750, -0.8157744]]
        )

        pass

    def tearDown(self):
        pass

    def test_eigenvalue_sensitivity(self):
        assert_allclose(eigenvalue_sensitivity(self.T, 0), self.S0)
        assert_allclose(eigenvalue_sensitivity(self.T, 1), self.S1)

    def test_timescale_sensitivity(self):
        assert_allclose(timescale_sensitivity(self.T, 1), self.TS1)

    def test_forward_committor_sensitivity(self):
        assert_allclose(forward_committor_sensitivity(self.T4, [0], [3], 0), self.S4zero)
        assert_allclose(forward_committor_sensitivity(self.T4, [0], [3], 1), self.qS41)
        assert_allclose(forward_committor_sensitivity(self.T4, [0], [3], 2), self.qS42)
        assert_allclose(forward_committor_sensitivity(self.T4, [0], [3], 3), self.S4zero)

    def test_backward_committor_sensitivity(self):
        assert_allclose(backward_committor_sensitivity(self.T4, [0], [3], 1), self.qSI41)

    def test_mfpt_sensitivity(self):
        assert_allclose(mfpt_sensitivity(self.T4, 0, 0), self.S4zero)
        assert_allclose(mfpt_sensitivity(self.T4, 0, 1), self.mS01)
        assert_allclose(mfpt_sensitivity(self.T4, 0, 2), self.mS02)
        assert_allclose(mfpt_sensitivity(self.T4, 3, 2), self.mS32)

    def test_eigenvector_sensitivity(self):
        assert_allclose(eigenvector_sensitivity(self.T4, 1, 1), self.mV11, atol=1e-5)
        assert_allclose(eigenvector_sensitivity(self.T4, 2, 2), self.mV22, atol=1e-5)
        assert_allclose(eigenvector_sensitivity(self.T4, 0, 3), self.mV03, atol=1e-5)

        assert_allclose(eigenvector_sensitivity(self.T4, 0, 1, right=False), self.mV01left, atol=1e-5)

    def test_stationary_sensitivity(self):
        assert_allclose(stationary_distribution_sensitivity(self.T4, 1), self.pS1, atol=1e-5)

    def test_expectation_sensitivity(self):
        a = np.array([0.0, 3.0, 0.0, 0.0])
        S = 3.0 * self.pS1
        Sn = expectation_sensitivity(self.T4, a)
        assert_allclose(Sn, S)
