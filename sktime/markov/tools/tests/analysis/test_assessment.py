
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

"""This module provides unit tests for the assessment functions of the analysis API

.. moduleauthor:: Martin Scherer <m DOT scherer AT fu-berlin DOT de>
.. moduleauthor:: Benjamin Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest
import numpy as np

import scipy.sparse
from scipy.sparse.dia import dia_matrix

from msmtools.util.birth_death_chain import BirthDeathChain

from msmtools.analysis import is_rate_matrix, is_reversible, is_transition_matrix, is_connected

################################################################################
# Dense
################################################################################


def create_rate_matrix():
    a = [[-3, 3, 0, 0],
         [3, -5, 2, 0],
         [0, 3, -5, 2],
         [0, 0, 3, -3]]

    return np.array(a)


class TestAssessmentDense(unittest.TestCase):
    def setUp(self):
        p = np.zeros(10)
        q = np.zeros(10)
        p[0:-1] = 0.5
        q[1:] = 0.5
        p[4] = 0.01
        q[6] = 0.1

        self.bdc = BirthDeathChain(q, p)
        self.T = self.bdc.transition_matrix()
        self.mu = self.bdc.stationary_distribution()

        self.A = create_rate_matrix()

    def test_IsRateMatrix(self):
        self.assertTrue(is_rate_matrix(self.A), 'A should be a rate matrix')

        # manipulate matrix so it isn't a rate matrix any more
        self.A[0][0] = 3
        self.assertFalse(is_rate_matrix(self.A), 'matrix is not a rate matrix')

    def test_IsReversible(self):
        # create a reversible matrix
        self.assertTrue(is_reversible(self.T, self.mu),
                        "T should be reversible")

    def test_is_transition_matrix(self):
        self.assertTrue(is_transition_matrix(self.T))

        """Larger test-case to prevent too restrictive tolerance settings"""
        X = np.random.random((2000, 2000))
        Tlarge = X / X.sum(axis=1)[:, np.newaxis]
        self.assertTrue(is_transition_matrix(Tlarge))

    def test_is_connected(self):
        self.assertTrue(is_connected(self.T))
        self.assertTrue(is_connected(self.T, directed=False))


################################################################################
# Sparse
################################################################################

def normalize_rows(A):
    """Normalize rows of sparse marix"""
    A = A.tocsr()
    values = A.data
    indptr = A.indptr  # Row index pointers
    indices = A.indices  # Column indices

    dim = A.shape[0]
    normed_values = np.zeros(len(values))

    for i in range(dim):
        thisrow = values[indptr[i]:indptr[i + 1]]
        rowsum = np.sum(thisrow)
        normed_values[indptr[i]:indptr[i + 1]] = thisrow / rowsum

    return scipy.sparse.csr_matrix((normed_values, indices, indptr))


def random_nonempty_rows(M, N, density=0.01):
    """Generate a random sparse matrix with nonempty rows"""
    N_el = int(density * M * N)  # total number of non-zero elements
    if N_el < M:
        raise ValueError("Density too small to obtain nonempty rows")
    else:
        rows = np.zeros(N_el, dtype=int)
        rows[0:M] = np.arange(M)
        rows[M:N_el] = np.random.randint(0, M, size=(N_el - M,))
        cols = np.random.randint(0, N, size=(N_el,))
        values = np.random.rand(N_el)
        return scipy.sparse.coo_matrix((values, (rows, cols)))


class TestTransitionMatrixSparse(unittest.TestCase):
    def setUp(self):
        self.dim = 10000
        self.density = 0.001
        self.tol = 1e-15
        A = random_nonempty_rows(self.dim, self.dim, density=self.density)
        self.T = normalize_rows(A)

    def tearDown(self):
        pass

    def test_is_transition_matrix(self):
        self.assertTrue(is_transition_matrix(self.T, tol=self.tol))


class TestRateMatrixSparse(unittest.TestCase):
    def create_sparse_rate_matrix(self):
        """
        constructs the following rate matrix for a M/M/1 queue
        TODO: fix math string
        :math: `
        Q = \begin{pmatrix}
        -\lambda & \lambda \\
        \mu & -(\mu+\lambda) & \lambda \\
        &\mu & -(\mu+\lambda) & \lambda \\
        &&\mu & -(\mu+\lambda) & \lambda &\\
        &&&&\ddots
        \end{pmatrix}`
        taken from: https://en.wikipedia.org/wiki/Transition_rate_matrix
        """
        lambda_ = 5
        mu = 3
        dim = self.dim

        diag = np.empty((3, dim))
        # main diagonal
        diag[0, 0] = (-lambda_)
        diag[0, 1:dim - 1] = -(mu + lambda_)
        diag[0, dim - 1] = lambda_

        # lower diag
        diag[1, :] = mu
        diag[1, -2:] = -mu
        diag[1, -2:] = lambda_
        diag[0, dim - 1] = -lambda_
        # upper diag
        diag[2, :] = lambda_

        offsets = [0, -1, 1]

        return dia_matrix((diag, offsets), shape=(dim, dim))

    def setUp(self):
        self.dim = 10
        self.K = self.create_sparse_rate_matrix()
        self.tol = 1e-15

    def test_is_rate_matrix(self):
        K_copy = self.K.copy()
        self.assertTrue(is_rate_matrix(self.K, self.tol), "K should be evaluated as rate matrix.")

        self.assertTrue(np.allclose(self.K.data, K_copy.data) and np.allclose(self.K.offsets, K_copy.offsets),
                        "object modified!")


class TestReversibleSparse(unittest.TestCase):
    def create_rev_t(self):
        dim = self.dim

        diag = np.zeros((3, dim))

        # forward_p = 4 / 5.
        forward_p = 0.6
        backward_p = 1 - forward_p
        # main diagonal
        diag[0, 0] = backward_p
        diag[0, -1] = backward_p

        # lower diag
        diag[1, :] = backward_p
        diag[1, 1] = forward_p

        # upper diag
        diag[2, :] = forward_p

        return dia_matrix((diag, [0, 1, -1]), shape=(dim, dim))

    def setUp(self):
        self.dim = 100
        self.tol = 1e-15
        self.T = self.create_rev_t()

    def test_is_reversible(self):
        self.assertTrue(is_reversible(self.T, tol=self.tol), 'matrix should be reversible')


class TestIsConnectedSparse(unittest.TestCase):
    def setUp(self):
        C1 = 1.0 * np.array([[1, 4, 3], [3, 2, 4], [4, 5, 1]])
        C2 = 1.0 * np.array([[0, 1], [1, 0]])
        C3 = 1.0 * np.array([[7]])

        C = scipy.sparse.block_diag((C1, C2, C3))

        C = C.toarray()
        """Forward transition block 1 -> block 2"""
        C[2, 3] = 1
        """Forward transition block 2 -> block 3"""
        C[4, 5] = 1

        self.T_connected = scipy.sparse.csr_matrix(C1 / C1.sum(axis=1)[:, np.newaxis])
        self.T_not_connected = scipy.sparse.csr_matrix(C / C.sum(axis=1)[:, np.newaxis])

    def tearDown(self):
        pass

    def test_connected_count_matrix(self):
        """Directed"""
        is_c = is_connected(self.T_not_connected)
        self.assertFalse(is_c)

        is_c = is_connected(self.T_connected)
        self.assertTrue(is_c)

        """Undirected"""
        is_c = is_connected(self.T_not_connected, directed=False)
        self.assertTrue(is_c)


if __name__ == "__main__":
    unittest.main()
