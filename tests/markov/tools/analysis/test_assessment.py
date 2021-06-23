"""
Created on 07.10.2013

@author: marscher and clonker
"""
import pytest
import numpy as np
import scipy
from numpy.testing import assert_, assert_allclose, assert_equal
from scipy.sparse import dia_matrix, csr_matrix, issparse, coo_matrix

from deeptime.data import birth_death_chain
from deeptime.markov.tools.analysis import is_rate_matrix, is_transition_matrix, is_reversible, is_connected


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

    return csr_matrix((normed_values, indices, indptr))


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
        return coo_matrix((values, (rows, cols)))


@pytest.fixture
def mm1_queue_rate_matrix(request):
    r""" constructs the following rate matrix for a M/M/1 queue
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
    dim = request.param[0]

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

    spmat = dia_matrix((diag, offsets), shape=(dim, dim))
    if not request.param[1]:
        spmat = spmat.toarray()
    return spmat


@pytest.fixture
def rate_matrix(request):
    A = np.array(
        [[-3, 3, 0, 0],
         [3, -5, 2, 0],
         [0, 3, -5, 2],
         [0, 0, 3, -3]]
    )
    if request.param:
        A = csr_matrix(A)
    return A


@pytest.mark.parametrize("rate_matrix", [False, True], indirect=True, ids=lambda x: f"sparse={x}")
def test_is_rate_matrix(rate_matrix):
    assert_(is_rate_matrix(rate_matrix))
    if issparse(rate_matrix):
        rate_matrix = rate_matrix.toarray()
        rate_matrix[0][0] = 3
        rate_matrix = csr_matrix(rate_matrix)
    else:
        rate_matrix[0][0] = 3
    assert_(not is_rate_matrix(rate_matrix))


@pytest.mark.parametrize("mm1_queue_rate_matrix", [(10, False), (10, True)], indirect=True,
                         ids=lambda x: f"sparse={x[1]}")
def test_is_rate_matrix_mm1_queue(mm1_queue_rate_matrix):
    K_copy = mm1_queue_rate_matrix.copy()
    assert_(is_rate_matrix(mm1_queue_rate_matrix, tol=1e-15))
    if issparse(mm1_queue_rate_matrix):
        assert_allclose(mm1_queue_rate_matrix.data, K_copy.data)
        assert_allclose(mm1_queue_rate_matrix.offsets, K_copy.offsets)
    else:
        assert_allclose(mm1_queue_rate_matrix, K_copy)


def test_random_transition_matrix(sparse_mode):
    if sparse_mode:
        dim = 10000
        density = 0.001
    else:
        dim = 25
        density = .5
    tol = 1e-15
    A = random_nonempty_rows(dim, dim, density=density)
    T = normalize_rows(A)
    if not sparse_mode:
        T = T.toarray()
    assert_(is_transition_matrix(T, tol=tol))


def test_is_reversible(sparse_mode):
    p = np.zeros(10)
    q = np.zeros(10)
    p[0:-1] = 0.5
    q[1:] = 0.5
    p[4] = 0.01
    q[6] = 0.1
    bdc = birth_death_chain(q, p, sparse=sparse_mode)
    assert_equal(sparse_mode, bdc.sparse)
    assert_equal(issparse(bdc.transition_matrix), sparse_mode)
    assert_(is_reversible(bdc.transition_matrix, bdc.stationary_distribution))


def test_is_connected(sparse_mode):
    C1 = 1.0 * np.array([[1, 4, 3], [3, 2, 4], [4, 5, 1]])
    C2 = 1.0 * np.array([[0, 1], [1, 0]])
    C3 = 1.0 * np.array([[7]])

    C = scipy.sparse.block_diag((C1, C2, C3))

    C = C.toarray()
    """Forward transition block 1 -> block 2"""
    C[2, 3] = 1
    """Forward transition block 2 -> block 3"""
    C[4, 5] = 1

    T_connected = scipy.sparse.csr_matrix(C1 / C1.sum(axis=1)[:, np.newaxis])
    T_not_connected = scipy.sparse.csr_matrix(C / C.sum(axis=1)[:, np.newaxis])

    if not sparse_mode:
        T_connected = T_connected.toarray()
        T_not_connected = T_not_connected.toarray()
    """Directed"""
    assert_(not is_connected(T_not_connected))
    assert_(is_connected(T_connected))

    """Undirected"""
    assert_(is_connected(T_not_connected, directed=False))
