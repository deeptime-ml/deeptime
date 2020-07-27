
# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

r"""Test package for the decomposition module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import unittest

import numpy as np
from msmtools.util.birth_death_chain import BirthDeathChain
from tests.numeric import assert_allclose
from scipy.linalg import eig, eigvals

from msmtools.analysis.sparse.decomposition import eigenvalues, eigenvectors, rdl_decomposition
from msmtools.analysis.sparse.decomposition import timescales


class TestDecomposition(unittest.TestCase):
    def setUp(self):
        self.dim = 100
        self.k = 10
        self.ncv = 40

        """Set up meta-stable birth-death chain"""
        p = np.zeros(self.dim)
        p[0:-1] = 0.5

        q = np.zeros(self.dim)
        q[1:] = 0.5

        p[self.dim // 2 - 1] = 0.001
        q[self.dim // 2 + 1] = 0.001

        self.bdc = BirthDeathChain(q, p)

    def test_eigenvalues(self):
        P = self.bdc.transition_matrix_sparse()
        P_dense = self.bdc.transition_matrix()
        ev = eigvals(P_dense)
        """Sort with decreasing magnitude"""
        ev = ev[np.argsort(np.abs(ev))[::-1]]

        """k=None"""
        with self.assertRaises(ValueError):
            evn = eigenvalues(P)

        """k is not None"""
        evn = eigenvalues(P, k=self.k)
        assert_allclose(ev[0:self.k], evn)

        """k is not None and ncv is not None"""
        evn = eigenvalues(P, k=self.k, ncv=self.ncv)
        assert_allclose(ev[0:self.k], evn)

    def test_eigenvalues_rev(self):
        P = self.bdc.transition_matrix_sparse()
        P_dense = self.bdc.transition_matrix()
        ev = eigvals(P_dense)
        """Sort with decreasing magnitude"""
        ev = ev[np.argsort(np.abs(ev))[::-1]]

        """k=None"""
        with self.assertRaises(ValueError):
            evn = eigenvalues(P, reversible=True)

        """k is not None"""
        evn = eigenvalues(P, k=self.k, reversible=True)
        assert_allclose(ev[0:self.k], evn)

        """k is not None and ncv is not None"""
        evn = eigenvalues(P, k=self.k, ncv=self.ncv, reversible=True)
        assert_allclose(ev[0:self.k], evn)

        """mu is not None"""
        mu = self.bdc.stationary_distribution()

        """k=None"""
        with self.assertRaises(ValueError):
            evn = eigenvalues(P, reversible=True, mu=mu)

        """k is not None"""
        evn = eigenvalues(P, k=self.k, reversible=True, mu=mu)
        assert_allclose(ev[0:self.k], evn)

        """k is not None and ncv is not None"""
        evn = eigenvalues(P, k=self.k, ncv=self.ncv, reversible=True, mu=mu)
        assert_allclose(ev[0:self.k], evn)

    def test_eigenvectors(self):
        P_dense = self.bdc.transition_matrix()
        P = self.bdc.transition_matrix_sparse()
        ev, L, R = eig(P_dense, left=True, right=True)
        ind = np.argsort(np.abs(ev))[::-1]
        ev = ev[ind]
        R = R[:, ind]
        L = L[:, ind]
        vals = ev[0:self.k]

        """k=None"""
        with self.assertRaises(ValueError):
            Rn = eigenvectors(P)

        with self.assertRaises(ValueError):
            Ln = eigenvectors(P, right=False)

        """k is not None"""
        Rn = eigenvectors(P, k=self.k)
        assert_allclose(vals[np.newaxis, :] * Rn, P.dot(Rn))

        Ln = eigenvectors(P, right=False, k=self.k)
        assert_allclose(P.transpose().dot(Ln), vals[np.newaxis, :] * Ln)

        """k is not None and ncv is not None"""
        Rn = eigenvectors(P, k=self.k, ncv=self.ncv)
        assert_allclose(vals[np.newaxis, :] * Rn, P.dot(Rn))

        Ln = eigenvectors(P, right=False, k=self.k, ncv=self.ncv)
        assert_allclose(P.transpose().dot(Ln), vals[np.newaxis, :] * Ln)

    def test_eigenvectors_rev(self):
        P_dense = self.bdc.transition_matrix()
        P = self.bdc.transition_matrix_sparse()
        ev, L, R = eig(P_dense, left=True, right=True)
        ind = np.argsort(np.abs(ev))[::-1]
        ev = ev[ind]
        R = R[:, ind]
        L = L[:, ind]
        vals = ev[0:self.k]

        """k=None"""
        with self.assertRaises(ValueError):
            Rn = eigenvectors(P, reversible=True)

        with self.assertRaises(ValueError):
            Ln = eigenvectors(P, right=False, reversible=True)

        """k is not None"""
        Rn = eigenvectors(P, k=self.k, reversible=True)
        assert_allclose(vals[np.newaxis, :] * Rn, P.dot(Rn))

        Ln = eigenvectors(P, right=False, k=self.k, reversible=True)
        assert_allclose(P.transpose().dot(Ln), vals[np.newaxis, :] * Ln)

        """k is not None and ncv is not None"""
        Rn = eigenvectors(P, k=self.k, ncv=self.ncv, reversible=True)
        assert_allclose(vals[np.newaxis, :] * Rn, P.dot(Rn))

        Ln = eigenvectors(P, right=False, k=self.k, ncv=self.ncv, reversible=True)
        assert_allclose(P.transpose().dot(Ln), vals[np.newaxis, :] * Ln)

        """mu is not None"""
        mu = self.bdc.stationary_distribution()

        """k=None"""
        with self.assertRaises(ValueError):
            Rn = eigenvectors(P, reversible=True, mu=mu)

        with self.assertRaises(ValueError):
            Ln = eigenvectors(P, right=False, reversible=True, mu=mu)

        """k is not None"""
        Rn = eigenvectors(P, k=self.k, reversible=True, mu=mu)
        assert_allclose(vals[np.newaxis, :] * Rn, P.dot(Rn))

        Ln = eigenvectors(P, right=False, k=self.k, reversible=True, mu=mu)
        assert_allclose(P.transpose().dot(Ln), vals[np.newaxis, :] * Ln)

        """k is not None and ncv is not None"""
        Rn = eigenvectors(P, k=self.k, ncv=self.ncv, reversible=True, mu=mu)
        assert_allclose(vals[np.newaxis, :] * Rn, P.dot(Rn))

        Ln = eigenvectors(P, right=False, k=self.k, ncv=self.ncv, reversible=True, mu=mu)
        assert_allclose(P.transpose().dot(Ln), vals[np.newaxis, :] * Ln)

    def test_rdl_decomposition(self):
        P = self.bdc.transition_matrix_sparse()
        mu = self.bdc.stationary_distribution()

        """Non-reversible"""

        """k=None"""
        with self.assertRaises(ValueError):
            Rn, Dn, Ln = rdl_decomposition(P)

        """k is not None"""
        Rn, Dn, Ln = rdl_decomposition(P, k=self.k)
        Xn = np.dot(Ln, Rn)
        """Right-eigenvectors"""
        assert_allclose(P.dot(Rn), np.dot(Rn, Dn))
        """Left-eigenvectors"""
        assert_allclose(P.transpose().dot(Ln.transpose()).transpose(), np.dot(Dn, Ln))
        """Orthonormality"""
        assert_allclose(Xn, np.eye(self.k))
        """Probability vector"""
        assert_allclose(np.sum(Ln[0, :]), 1.0)

        """k is not None, ncv is not None"""
        Rn, Dn, Ln = rdl_decomposition(P, k=self.k, ncv=self.ncv)
        Xn = np.dot(Ln, Rn)
        """Right-eigenvectors"""
        assert_allclose(P.dot(Rn), np.dot(Rn, Dn))
        """Left-eigenvectors"""
        assert_allclose(P.transpose().dot(Ln.transpose()).transpose(), np.dot(Dn, Ln))
        """Orthonormality"""
        assert_allclose(Xn, np.eye(self.k))
        """Probability vector"""
        assert_allclose(np.sum(Ln[0, :]), 1.0)

        """Reversible"""

        """k=None"""
        with self.assertRaises(ValueError):
            Rn, Dn, Ln = rdl_decomposition(P, norm='reversible')

        """k is not None"""
        Rn, Dn, Ln = rdl_decomposition(P, k=self.k, norm='reversible')
        Xn = np.dot(Ln, Rn)
        """Right-eigenvectors"""
        assert_allclose(P.dot(Rn), np.dot(Rn, Dn))
        """Left-eigenvectors"""
        assert_allclose(P.transpose().dot(Ln.transpose()).transpose(), np.dot(Dn, Ln))
        """Orthonormality"""
        assert_allclose(Xn, np.eye(self.k))
        """Probability vector"""
        assert_allclose(np.sum(Ln[0, :]), 1.0)
        """Reversibility"""
        assert_allclose(Ln.transpose(), mu[:, np.newaxis] * Rn)

        """k is not None ncv is not None"""
        Rn, Dn, Ln = rdl_decomposition(P, k=self.k, norm='reversible', ncv=self.ncv)
        Xn = np.dot(Ln, Rn)
        """Right-eigenvectors"""
        assert_allclose(P.dot(Rn), np.dot(Rn, Dn))
        """Left-eigenvectors"""
        assert_allclose(P.transpose().dot(Ln.transpose()).transpose(), np.dot(Dn, Ln))
        """Orthonormality"""
        assert_allclose(Xn, np.eye(self.k))
        """Probability vector"""
        assert_allclose(np.sum(Ln[0, :]), 1.0)
        """Reversibility"""
        assert_allclose(Ln.transpose(), mu[:, np.newaxis] * Rn)

    def test_rdl_decomposition_rev(self):
        P = self.bdc.transition_matrix_sparse()
        mu = self.bdc.stationary_distribution()

        """Non-reversible"""

        """k=None"""
        with self.assertRaises(ValueError):
            Rn, Dn, Ln = rdl_decomposition(P, reversible=True)

        """norm='standard'"""
        Rn, Dn, Ln = rdl_decomposition(P, k=self.k, reversible=True, norm='standard')
        Xn = np.dot(Ln, Rn)
        """Right-eigenvectors"""
        assert_allclose(P.dot(Rn), np.dot(Rn, Dn))
        """Left-eigenvectors"""
        assert_allclose(P.transpose().dot(Ln.transpose()).transpose(), np.dot(Dn, Ln))
        """Orthonormality"""
        assert_allclose(Xn, np.eye(self.k))
        """Probability vector"""
        assert_allclose(np.sum(Ln[0, :]), 1.0)
        """Standard l2-normalization of right eigenvectors except dominant one"""
        Yn = np.dot(Rn.T, Rn)
        assert_allclose(np.diag(Yn)[1:], 1.0)

        """ncv is not None"""
        Rn, Dn, Ln = rdl_decomposition(P, k=self.k, reversible=True,
                                       norm='standard', ncv=self.ncv)
        Xn = np.dot(Ln, Rn)
        """Right-eigenvectors"""
        assert_allclose(P.dot(Rn), np.dot(Rn, Dn))
        """Left-eigenvectors"""
        assert_allclose(P.transpose().dot(Ln.transpose()).transpose(), np.dot(Dn, Ln))
        """Orthonormality"""
        assert_allclose(Xn, np.eye(self.k))
        """Probability vector"""
        assert_allclose(np.sum(Ln[0, :]), 1.0)
        """Standard l2-normalization of right eigenvectors except dominant one"""
        Yn = np.dot(Rn.T, Rn)
        assert_allclose(np.diag(Yn)[1:], 1.0)

        """norm='reversible'"""
        Rn, Dn, Ln = rdl_decomposition(P, reversible=True, norm='reversible', k=self.k)
        Xn = np.dot(Ln, Rn)
        """Right-eigenvectors"""
        assert_allclose(P.dot(Rn), np.dot(Rn, Dn))
        """Left-eigenvectors"""
        assert_allclose(P.transpose().dot(Ln.transpose()).transpose(), np.dot(Dn, Ln))
        """Orthonormality"""
        assert_allclose(Xn, np.eye(self.k))
        """Probability vector"""
        assert_allclose(np.sum(Ln[0, :]), 1.0)
        """Reversibility"""
        assert_allclose(Ln.transpose(), mu[:, np.newaxis] * Rn)

        Rn, Dn, Ln = rdl_decomposition(P, reversible=True, norm='reversible',
                                       k=self.k, ncv=self.ncv)
        Xn = np.dot(Ln, Rn)
        """Right-eigenvectors"""
        assert_allclose(P.dot(Rn), np.dot(Rn, Dn))
        """Left-eigenvectors"""
        assert_allclose(P.transpose().dot(Ln.transpose()).transpose(), np.dot(Dn, Ln))
        """Orthonormality"""
        assert_allclose(Xn, np.eye(self.k))
        """Probability vector"""
        assert_allclose(np.sum(Ln[0, :]), 1.0)
        """Reversibility"""
        assert_allclose(Ln.transpose(), mu[:, np.newaxis] * Rn)

        """mu is not None"""

        """norm='standard'"""
        Rn, Dn, Ln = rdl_decomposition(P, k=self.k, reversible=True, norm='standard', mu=mu)
        Xn = np.dot(Ln, Rn)
        """Right-eigenvectors"""
        assert_allclose(P.dot(Rn), np.dot(Rn, Dn))
        """Left-eigenvectors"""
        assert_allclose(P.transpose().dot(Ln.transpose()).transpose(), np.dot(Dn, Ln))
        """Orthonormality"""
        assert_allclose(Xn, np.eye(self.k))
        """Probability vector"""
        assert_allclose(np.sum(Ln[0, :]), 1.0)
        """Standard l2-normalization of right eigenvectors except dominant one"""
        Yn = np.dot(Rn.T, Rn)
        assert_allclose(np.diag(Yn)[1:], 1.0)

        """ncv is not None"""
        Rn, Dn, Ln = rdl_decomposition(P, k=self.k, reversible=True,
                                       norm='standard', ncv=self.ncv, mu=mu)
        Xn = np.dot(Ln, Rn)
        """Right-eigenvectors"""
        assert_allclose(P.dot(Rn), np.dot(Rn, Dn))
        """Left-eigenvectors"""
        assert_allclose(P.transpose().dot(Ln.transpose()).transpose(), np.dot(Dn, Ln))
        """Orthonormality"""
        assert_allclose(Xn, np.eye(self.k))
        """Probability vector"""
        assert_allclose(np.sum(Ln[0, :]), 1.0)
        """Standard l2-normalization of right eigenvectors except dominant one"""
        Yn = np.dot(Rn.T, Rn)
        assert_allclose(np.diag(Yn)[1:], 1.0)

        """norm='reversible'"""
        Rn, Dn, Ln = rdl_decomposition(P, reversible=True, norm='reversible', k=self.k, mu=mu)
        Xn = np.dot(Ln, Rn)
        """Right-eigenvectors"""
        assert_allclose(P.dot(Rn), np.dot(Rn, Dn))
        """Left-eigenvectors"""
        assert_allclose(P.transpose().dot(Ln.transpose()).transpose(), np.dot(Dn, Ln))
        """Orthonormality"""
        assert_allclose(Xn, np.eye(self.k))
        """Probability vector"""
        assert_allclose(np.sum(Ln[0, :]), 1.0)
        """Reversibility"""
        assert_allclose(Ln.transpose(), mu[:, np.newaxis] * Rn)

        Rn, Dn, Ln = rdl_decomposition(P, reversible=True, norm='reversible',
                                       k=self.k, ncv=self.ncv, mu=mu)
        Xn = np.dot(Ln, Rn)
        """Right-eigenvectors"""
        assert_allclose(P.dot(Rn), np.dot(Rn, Dn))
        """Left-eigenvectors"""
        assert_allclose(P.transpose().dot(Ln.transpose()).transpose(), np.dot(Dn, Ln))
        """Orthonormality"""
        assert_allclose(Xn, np.eye(self.k))
        """Probability vector"""
        assert_allclose(np.sum(Ln[0, :]), 1.0)
        """Reversibility"""
        assert_allclose(Ln.transpose(), mu[:, np.newaxis] * Rn)

    def test_timescales(self):
        P_dense = self.bdc.transition_matrix()
        P = self.bdc.transition_matrix_sparse()
        ev = eigvals(P_dense)
        """Sort with decreasing magnitude"""
        ev = ev[np.argsort(np.abs(ev))[::-1]]
        ts = -1.0 / np.log(np.abs(ev))

        """k=None"""
        with self.assertRaises(ValueError):
            tsn = timescales(P)

        """k is not None"""
        tsn = timescales(P, k=self.k)
        assert_allclose(ts[1:self.k], tsn[1:])

        """k is not None, ncv is not None"""
        tsn = timescales(P, k=self.k, ncv=self.ncv)
        assert_allclose(ts[1:self.k], tsn[1:])

        """tau=7"""

        """k is not None"""
        tsn = timescales(P, k=self.k, tau=7)
        assert_allclose(7 * ts[1:self.k], tsn[1:])

    def test_timescales_rev(self):
        P_dense = self.bdc.transition_matrix()
        P = self.bdc.transition_matrix_sparse()
        mu = self.bdc.stationary_distribution()
        ev = eigvals(P_dense)
        """Sort with decreasing magnitude"""
        ev = ev[np.argsort(np.abs(ev))[::-1]]
        ts = -1.0 / np.log(np.abs(ev))

        """k=None"""
        with self.assertRaises(ValueError):
            tsn = timescales(P, reversible=True)

        """k is not None"""
        tsn = timescales(P, k=self.k, reversible=True)
        assert_allclose(ts[1:self.k], tsn[1:])

        """k is not None, ncv is not None"""
        tsn = timescales(P, k=self.k, ncv=self.ncv, reversible=True)
        assert_allclose(ts[1:self.k], tsn[1:])

        """k is not None, mu is not None"""
        tsn = timescales(P, k=self.k, reversible=True, mu=mu)
        assert_allclose(ts[1:self.k], tsn[1:])

        """k is not None, mu is not None, ncv is not None"""
        tsn = timescales(P, k=self.k, ncv=self.ncv, reversible=True, mu=mu)
        assert_allclose(ts[1:self.k], tsn[1:])

        """tau=7"""

        """k is not None"""
        tsn = timescales(P, k=self.k, tau=7, reversible=True)
        assert_allclose(7 * ts[1:self.k], tsn[1:])


if __name__ == "__main__":
    unittest.main()
