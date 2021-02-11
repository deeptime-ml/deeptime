r"""Unit tests for the mean first passage time module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>
.. moduleauthor:: C.Wehmeyer <christoph DOT wehmeyer AT fu-berlin DOT de>

"""
import unittest

import numpy as np
from tests.markov.tools.numeric import assert_allclose

from deeptime.markov.tools.analysis.dense._mean_first_passage_time import mfpt, mfpt_between_sets


class TestMfpt(unittest.TestCase):
    def setUp(self):
        p1 = 0.7
        p2 = 0.5
        q2 = 0.4
        q3 = 0.6
        """3x3 birth and death transition matrix"""
        self.P = np.array([[1.0 - p1, p1, 0.0],
                           [q2, 1.0 - q2 - p2, p2],
                           [0.0, q3, 1.0 - q3]])
        """Vector of mean first passage times to target state t=0"""
        self.m0 = np.array([0.0, (p2 + q3) / (q2 * q3), (p2 + q2 + q3) / (q2 * q3)])
        """Vector of mean first passage times to target state t=1"""
        self.m1 = np.array([1.0 / p1, 0.0, 1.0 / q3])
        """Vector of mean first passage times to target state t=2"""
        self.m2 = np.array([(p2 + p1 + q2) / (p1 * p2), (p1 + q2) / (p1 * p2), 0.0])
        """Vector of mean first passage times to target states t=(0,1)"""
        self.m01 = np.array([0.0, 0.0, 1.0 / q3])
        """Vector of mean first passage times to target states t=(1,2)"""
        self.m12 = np.array([1.0 / p1, 0.0, 0.0])
        """Vector of stationary weights"""
        self.mu = np.array([1.0, p1 / q2, p1 * p2 / q2 / q3])
        """Mean first passage times from (0) to (1,2)"""
        self.o0t12 = 1.0 / p1
        """Mean first passage times from (2) to (0,1)"""
        self.o2t01 = 1.0 / q3
        """Mean first passage times from (0,1) to (2)"""
        self.o01t2 = (self.m2[0] * self.mu[0] + self.m2[1] * self.mu[1]) / (self.mu[0] + self.mu[1])
        """Mean first passage times from (1,2) to (0)"""
        self.o12t0 = (self.m0[1] * self.mu[1] + self.m0[2] * self.mu[2]) / (self.mu[1] + self.mu[2])

    def tearDown(self):
        pass

    def test_mfpt(self):
        x = mfpt(self.P, 0)
        assert_allclose(x, self.m0)

        x = mfpt(self.P, 1)
        assert_allclose(x, self.m1)

        x = mfpt(self.P, 2)
        assert_allclose(x, self.m2)

        x = mfpt(self.P, [0, 1])
        assert_allclose(x, self.m01)

        x = mfpt(self.P, [1, 2])
        assert_allclose(x, self.m12)

    def test_mfpt_between_sets(self):
        x = mfpt_between_sets(self.P, [1, 2], 0)
        assert_allclose(x, self.o0t12)

        x = mfpt_between_sets(self.P, [0, 1], 2)
        assert_allclose(x, self.o2t01)

        x = mfpt_between_sets(self.P, 2, [0, 1])
        assert_allclose(x, self.o01t2)

        x = mfpt_between_sets(self.P, 0, [1, 2])
        assert_allclose(x, self.o12t0)
