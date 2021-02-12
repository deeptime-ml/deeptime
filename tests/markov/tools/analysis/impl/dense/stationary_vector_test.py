import unittest
import numpy as np

from deeptime.data import birth_death_chain
from tests.markov.tools.numeric import assert_allclose

from deeptime.markov.tools.analysis.dense._stationary_vector import stationary_distribution_from_eigenvector
from deeptime.markov.tools.analysis.dense._stationary_vector import stationary_distribution_from_backward_iteration


class TestStationaryVector(unittest.TestCase):
    def setUp(self):
        self.dim = 100
        self.k = 10

        """Set up meta-stable birth-death chain"""
        p = np.zeros(self.dim)
        p[0:-1] = 0.5

        q = np.zeros(self.dim)
        q[1:] = 0.5

        p[self.dim // 2 - 1] = 0.001
        q[self.dim // 2 + 1] = 0.001

        self.bdc = birth_death_chain(q, p)

    def test_statdist_decomposition(self):
        P = self.bdc.transition_matrix
        mu = self.bdc.stationary_distribution
        mun = stationary_distribution_from_eigenvector(P)
        assert_allclose(mu, mun)

    def test_statdist_iteration(self):
        P = self.bdc.transition_matrix
        mu = self.bdc.stationary_distribution
        mun = stationary_distribution_from_backward_iteration(P)
        assert_allclose(mu, mun)
