
# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group
#
# MSMTools is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

r"""Unit test for the TPT-functions of the analysis API

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest
import numpy as np
from msmtools.util.birth_death_chain import BirthDeathChain
from tests.numeric import assert_allclose

from scipy.sparse import csr_matrix

import msmtools.flux as flux

################################################################################
# Dense
################################################################################


class TestTPTDense(unittest.TestCase):
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

        self.bdc = BirthDeathChain(q, p)
        self.T = self.bdc.transition_matrix()

        """Compute mu, qminus, qplus in constructor"""
        self.tpt = flux.tpt(self.T, self.A, self.B)

        """Use precomputed mu, qminus, qplus"""
        self.mu = self.bdc.stationary_distribution()
        self.qminus = self.bdc.committor_backward(self.a, self.b)
        self.qplus = self.bdc.committor_forward(self.a, self.b)
        self.tpt_fast = flux.tpt(self.T, self.A, self.B, mu=self.mu, qminus=self.qminus, qplus=self.qplus)

    def test_grossflux(self):
        flux = self.bdc.flux(self.a, self.b)

        fluxn = self.tpt.gross_flux
        assert_allclose(fluxn, flux)

        fluxn = self.tpt_fast.gross_flux
        assert_allclose(fluxn, flux)

    def test_netflux(self):
        netflux = self.bdc.netflux(self.a, self.b)

        netfluxn = self.tpt.net_flux
        assert_allclose(netfluxn, netflux)

        netfluxn = self.tpt_fast.net_flux
        assert_allclose(netfluxn, netflux)

    def test_totalflux(self):
        F = self.bdc.totalflux(self.a, self.b)

        Fn = self.tpt.total_flux
        assert_allclose(Fn, F)

        Fn = self.tpt_fast.total_flux
        assert_allclose(Fn, F)

    def test_rate(self):
        k = self.bdc.rate(self.a, self.b)

        kn = self.tpt.rate
        assert_allclose(kn, k)

        kn = self.tpt_fast.rate
        assert_allclose(kn, k)

    def test_backward_committor(self):
        qminus = self.qminus

        qminusn = self.tpt.backward_committor
        assert_allclose(qminusn, qminus)

        qminusn = self.tpt_fast.backward_committor
        assert_allclose(qminusn, qminus)

    def test_forward_committor(self):
        qplus = self.qplus

        qplusn = self.tpt.forward_committor
        assert_allclose(qplusn, qplus)

        qplusn = self.tpt_fast.forward_committor
        assert_allclose(qplusn, qplus)

    def test_stationary_distribution(self):
        mu = self.mu

        mun = self.tpt.stationary_distribution
        assert_allclose(mun, mu)

        mun = self.tpt_fast.stationary_distribution
        assert_allclose(mun, mu)


class TestTptFunctionsDense(unittest.TestCase):
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

        self.bdc = BirthDeathChain(q, p)
        self.T = self.bdc.transition_matrix()

        self.mu = self.bdc.stationary_distribution()
        self.qminus = self.bdc.committor_backward(self.a, self.b)
        self.qplus = self.bdc.committor_forward(self.a, self.b)

        # present results
        self.fluxn = flux.flux_matrix(self.T, self.mu, self.qminus, self.qplus, netflux=False)
        self.netfluxn = flux.flux_matrix(self.T, self.mu, self.qminus, self.qplus, netflux=True)
        self.totalfluxn = flux.total_flux(self.netfluxn, self.A)
        self.raten = flux.rate(self.totalfluxn, self.mu, self.qminus)

    def test_tpt_flux(self):
        flux = self.bdc.flux(self.a, self.b)
        assert_allclose(self.fluxn, flux)

    def test_tpt_netflux(self):
        netflux = self.bdc.netflux(self.a, self.b)
        assert_allclose(self.netfluxn, netflux)

    def test_tpt_totalflux(self):
        totalflux = self.bdc.totalflux(self.a, self.b)
        assert_allclose(self.totalfluxn, totalflux)

    def test_tpt_rate(self):
        rate = self.bdc.rate(self.a, self.b)
        assert_allclose(self.raten, rate)


################################################################################
# Sparse
################################################################################

class TestTPTSparse(unittest.TestCase):
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

        self.bdc = BirthDeathChain(q, p)
        T_dense = self.bdc.transition_matrix()
        T_sparse = csr_matrix(T_dense)
        self.T = T_sparse

        """Compute mu, qminus, qplus in constructor"""
        self.tpt = flux.tpt(self.T, self.A, self.B)

        """Use precomputed mu, qminus, qplus"""
        self.mu = self.bdc.stationary_distribution()
        self.qminus = self.bdc.committor_backward(self.a, self.b)
        self.qplus = self.bdc.committor_forward(self.a, self.b)
        self.tpt_fast = flux.tpt(self.T, self.A, self.B, mu=self.mu, qminus=self.qminus, qplus=self.qplus)

    def test_flux(self):
        flux = self.bdc.flux(self.a, self.b)

        fluxn = self.tpt.gross_flux
        assert_allclose(fluxn.toarray(), flux)

        fluxn = self.tpt_fast.gross_flux
        assert_allclose(fluxn.toarray(), flux)

    def test_netflux(self):
        netflux = self.bdc.netflux(self.a, self.b)

        netfluxn = self.tpt.net_flux
        assert_allclose(netfluxn.toarray(), netflux)

        netfluxn = self.tpt_fast.net_flux
        assert_allclose(netfluxn.toarray(), netflux)

    def test_totalflux(self):
        F = self.bdc.totalflux(self.a, self.b)

        Fn = self.tpt.total_flux
        assert_allclose(Fn, F)

        Fn = self.tpt_fast.total_flux
        assert_allclose(Fn, F)

    def test_rate(self):
        k = self.bdc.rate(self.a, self.b)

        kn = self.tpt.rate
        assert_allclose(kn, k)

        kn = self.tpt_fast.rate
        assert_allclose(kn, k)

    def test_backward_committor(self):
        qminus = self.qminus

        qminusn = self.tpt.backward_committor
        assert_allclose(qminusn, qminus)

        qminusn = self.tpt_fast.backward_committor
        assert_allclose(qminusn, qminus)

    def test_forward_committor(self):
        qplus = self.qplus

        qplusn = self.tpt.forward_committor
        assert_allclose(qplusn, qplus)

        qplusn = self.tpt_fast.forward_committor
        assert_allclose(qplusn, qplus)

    def test_stationary_distribution(self):
        mu = self.mu

        mun = self.tpt.stationary_distribution
        assert_allclose(mun, mu)

        mun = self.tpt_fast.stationary_distribution
        assert_allclose(mun, mu)


class TestTptFunctionsSparse(unittest.TestCase):
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

        self.bdc = BirthDeathChain(q, p)
        T_dense = self.bdc.transition_matrix()
        T_sparse = csr_matrix(T_dense)
        self.T = T_sparse

        self.mu = self.bdc.stationary_distribution()
        self.qminus = self.bdc.committor_backward(self.a, self.b)
        self.qplus = self.bdc.committor_forward(self.a, self.b)

        # present results
        self.fluxn = flux.flux_matrix(self.T, self.mu, self.qminus, self.qplus, netflux=False)
        self.netfluxn = flux.flux_matrix(self.T, self.mu, self.qminus, self.qplus, netflux=True)
        self.totalfluxn = flux.total_flux(self.netfluxn, self.A)
        self.raten = flux.rate(self.totalfluxn, self.mu, self.qminus)

    def test_tpt_flux(self):
        flux = self.bdc.flux(self.a, self.b)
        assert_allclose(self.fluxn.toarray(), flux)

    def test_tpt_netflux(self):
        netflux = self.bdc.netflux(self.a, self.b)
        assert_allclose(self.netfluxn.toarray(), netflux)

    def test_tpt_totalflux(self):
        totalflux = self.bdc.totalflux(self.a, self.b)
        assert_allclose(self.totalfluxn, totalflux)

    def test_tpt_rate(self):
        rate = self.bdc.rate(self.a, self.b)
        assert_allclose(self.raten, rate)


if __name__ == "__main__":
    unittest.main()
