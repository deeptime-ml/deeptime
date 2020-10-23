r"""Unit test for the TPT-module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest
import numpy as np

from deeptime.data import birth_death_chain
from tests.markov.tools.numeric import assert_allclose

from deeptime.markov.tools.flux.dense import tpt


class TestTPT(unittest.TestCase):
    def setUp(self):
        p = np.zeros(10)
        q = np.zeros(10)
        p[0:-1] = 0.5
        q[1:] = 0.5
        p[4] = 0.01
        q[6] = 0.1

        self.A = [0, 1]
        self.B = [8, 9]
        self.a = 1
        self.b = 8

        self.bdc = birth_death_chain(q, p)
        self.T = self.bdc.transition_matrix

        """Use precomputed mu, qminus, qplus"""
        self.mu = self.bdc.stationary_distribution
        self.qplus = self.bdc.committor_forward(self.a, self.b)
        self.qminus = self.bdc.committor_backward(self.a, self.b)
        # self.qminus = committor.backward_committor(self.T, self.A, self.B, mu=self.mu)
        # self.qplus = committor.forward_committor(self.T, self.A, self.B)
        self.fluxn = tpt.flux_matrix(self.T, self.mu, self.qminus, self.qplus, netflux=False)
        self.netfluxn = tpt.to_netflux(self.fluxn)
        self.Fn = tpt.total_flux(self.fluxn, self.A)
        self.kn = tpt.rate(self.Fn, self.mu, self.qminus)

    def test_flux(self):
        flux = self.bdc.flux(self.a, self.b)
        assert_allclose(self.fluxn, flux)

    def test_netflux(self):
        netflux = self.bdc.netflux(self.a, self.b)
        assert_allclose(self.netfluxn, netflux)

    def test_totalflux(self):
        F = self.bdc.totalflux(self.a, self.b)
        assert_allclose(self.Fn, F)

    def test_rate(self):
        k = self.bdc.rate(self.a, self.b)
        assert_allclose(self.kn, k)
