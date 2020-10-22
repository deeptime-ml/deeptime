r"""Unit tests for the committor module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import unittest
import numpy as np

from deeptime.data import birth_death_chain
from tests.markov.tools.numeric import assert_allclose

from deeptime.markov.tools.analysis.sparse import committor


class TestCommittor(unittest.TestCase):
    def setUp(self):
        p = np.zeros(100)
        q = np.zeros(100)
        p[0:-1] = 0.5
        q[1:] = 0.5
        p[49] = 0.01
        q[51] = 0.1

        self.bdc = birth_death_chain(q, p)

    def tearDown(self):
        pass

    def test_forward_comittor(self):
        P = self.bdc.transition_matrix_sparse
        un = committor.forward_committor(P, list(range(10)), list(range(90, 100)))
        u = self.bdc.committor_forward(9, 90)
        assert_allclose(un, u)

    def test_backward_comittor(self):
        P = self.bdc.transition_matrix_sparse
        un = committor.backward_committor(P, list(range(10)), list(range(90, 100)))
        u = self.bdc.committor_backward(9, 90)
        assert_allclose(un, u)
