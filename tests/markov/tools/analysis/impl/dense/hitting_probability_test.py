r"""Unit test, dense implementation of hitting probabilities

.. moduleauthor:: F.Noe <frank DOT noe AT fu-berlin DOT de>

"""
import unittest

import numpy as np
from tests.markov.tools.numeric import assert_allclose

from deeptime.markov.tools.analysis.dense._hitting_probability import hitting_probability


class TestHitting(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_hitting1(self):
        P = np.array([[0., 1., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.]])
        sol = np.array([1, 0, 0])
        assert_allclose(hitting_probability(P, 1), sol)
        assert_allclose(hitting_probability(P, [1, 2]), sol)

    def test_hitting2(self):
        P = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.1, 0.8, 0.1, 0.0],
                      [0.0, 0.0, 0.8, 0.2],
                      [0.0, 0.0, 0.2, 0.8]])
        sol = np.array([0., 0.5, 1., 1.])
        assert_allclose(hitting_probability(P, [2, 3]), sol)

    def test_hitting3(self):
        P = np.array([[0.9, 0.1, 0.0, 0.0, 0.0],
                      [0.1, 0.9, 0.0, 0.0, 0.0],
                      [0.0, 0.1, 0.4, 0.5, 0.0],
                      [0.0, 0.0, 0.0, 0.8, 0.2],
                      [0.0, 0.0, 0.0, 0.2, 0.8]])
        sol = np.array([0.0, 0.0, 8.33333333e-01, 1.0, 1.0])
        assert_allclose(hitting_probability(P, 3), sol)
        assert_allclose(hitting_probability(P, [3, 4]), sol)
