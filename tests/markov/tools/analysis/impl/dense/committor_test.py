r"""Unit tests for the committor module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest
import numpy as np

from sktime.data import birth_death_chain

from tests.markov.tools.numeric import assert_allclose

from sktime.markov.tools.analysis.dense import committor


class TestCommittor(unittest.TestCase):
    def setUp(self):
        p = np.zeros(10)
        q = np.zeros(10)
        p[0:-1] = 0.5
        q[1:] = 0.5
        p[4] = 0.01
        q[6] = 0.1

        self.bdc = birth_death_chain(q, p)

    def test_forward_comittor(self):
        P = self.bdc.transition_matrix
        un = committor.forward_committor(P, [0, 1], [8, 9])
        u = self.bdc.committor_forward(1, 8)
        assert_allclose(un, u)

    def test_backward_comittor(self):
        P = self.bdc.transition_matrix
        un = committor.backward_committor(P, [0, 1], [8, 9])
        u = self.bdc.committor_backward(1, 8)
        assert_allclose(un, u)
