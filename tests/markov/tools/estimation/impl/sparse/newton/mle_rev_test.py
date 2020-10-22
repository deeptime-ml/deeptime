r"""Unit test for the reversible mle newton module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest

from os.path import abspath, join
from os import pardir

import numpy as np
from scipy.sparse import csr_matrix

from tests.markov.tools.numeric import assert_allclose
from deeptime.markov.tools.estimation.sparse.mle.newton.mle_rev import solve_mle_rev

testpath = abspath(join(abspath(__file__), pardir)) + '/testfiles/'


class TestReversibleEstimatorNewton(unittest.TestCase):
    def setUp(self):
        """Count matrix for bith-death chain with 100 states"""
        self.C = np.loadtxt(testpath + 'C.dat')
        self.P_ref = self.C/self.C.sum(axis=1)[:,np.newaxis]

    def test_estimator(self):
        P, pi = solve_mle_rev(csr_matrix(self.C))
        assert_allclose(P.toarray(), self.P_ref)
