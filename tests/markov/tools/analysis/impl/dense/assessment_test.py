"""
Created on 07.10.2013

@author: marscher
"""
import unittest
import numpy as np

from deeptime.data import birth_death_chain
from deeptime.markov.tools.analysis.dense import assessment


class RateMatrixTest(unittest.TestCase):

    def setUp(self):
        self.A = np.array(
            [[-3, 3, 0, 0],
             [3, -5, 2, 0],
             [0, 3, -5, 2],
             [0, 0, 3, -3]]
        )

    def testIsRateMatrix(self):
        self.assertTrue(assessment.is_rate_matrix(self.A), 'A should be a rate matrix')

        # manipulate matrix so it isn't a rate matrix any more
        self.A[0][0] = 3
        self.assertFalse(assessment.is_rate_matrix(self.A), 'matrix is not a rate matrix')


class ReversibleTest(unittest.TestCase):

    def setUp(self):
        p = np.zeros(10)
        q = np.zeros(10)
        p[0:-1] = 0.5
        q[1:] = 0.5
        p[4] = 0.01
        q[6] = 0.1

        self.bdc = birth_death_chain(q, p)

    def testIsReversible(self):
        # create a reversible matrix
        self.assertTrue(assessment.is_reversible(self.bdc.transition_matrix, self.bdc.stationary_distribution),
                        "T should be reversible")
