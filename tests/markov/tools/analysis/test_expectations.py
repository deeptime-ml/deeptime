r"""This module provides unit tests for the expectations function in API

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>
.. moduleauthor:: clonker

"""

import numpy as np
import pytest
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from numpy.random import choice
from scipy.linalg import eig
from scipy.sparse import diags

from deeptime.markov.tools.analysis import expected_counts, expected_counts_stationary
from tests.markov.tools.numeric import assert_allclose


################################################################################
# Dense
################################################################################


@pytest.fixture
def setting():
    dim = 20
    C = np.random.randint(0, 100, size=(dim, dim))
    C = C + np.transpose(C)  # Symmetric count matrix for real eigenvalues
    T = 1.0 * C / np.sum(C, axis=1)[:, np.newaxis]
    """Eigenvalues and left eigenvectors, sorted"""
    v, L, R = scipy.linalg.eig(T, left=True, right=True)
    v, L = eig(np.transpose(T))
    ind = np.argsort(np.abs(v))[::-1]
    v = v[ind]
    L = L[:, ind]
    """Compute stationary distribution"""
    mu = L[:, 0] / np.sum(L[:, 0])

    yield T, mu, v, L, R


def random_orthonormal_sparse_vectors(d, k):
    r"""Generate a random set of k orthonormal sparse vectors

    The algorithm draws random indices, {i_1,...,i_k}, from the set
    of all possible indices, {0,...,d-1}, without replacement.
    Random sparse vectors v are given by

    v[i]=k^{-1/2} for i in {i_1,...,i_k} and zero elsewhere.

    """
    indices = choice(d, replace=False, size=(k * k))
    indptr = np.arange(0, k * (k + 1), k)
    values = 1.0 / np.sqrt(k) * np.ones(k * k)
    return scipy.sparse.csc_matrix((values, indices, indptr))


def to_sparse_setting(setting, output_dim):
    T, mu, v, L, R = setting
    k = T.shape[0]
    """
    Generate k random sparse
    orthorgonal vectors of dimension d
    """
    Q = random_orthonormal_sparse_vectors(output_dim, k)
    """Push forward dense decomposition to sparse one via Q"""
    L_sparse = Q.dot(scipy.sparse.csr_matrix(L))
    R_sparse = Q.dot(scipy.sparse.csr_matrix(R))
    v_sparse = v  # Eigenvalues are invariant

    """Push forward transition matrix and stationary distribution"""
    T_sparse = Q.dot(scipy.sparse.csr_matrix(T)).dot(Q.transpose())
    mu_sparse = Q.dot(mu) / np.sqrt(k)
    return T_sparse, mu_sparse, v_sparse, L_sparse, R_sparse


@pytest.mark.parametrize("N", [20, 50, 0], ids=lambda x: f"nsteps={x}")
def test_expected_counts(setting, sparse_mode, N):
    if sparse_mode:
        setting = to_sparse_setting(setting, 10000)
    T, mu, v, L, R = setting
    EC_n = expected_counts(T, mu, N)
    D_mu = diags(mu, 0)
    if N == 0:
        EC_true = np.zeros(T.shape)
    else:
        # If p0 is the stationary vector the computation can be carried out by a simple multiplication
        EC_true = N * D_mu.dot(T)
    assert_allclose(EC_true, EC_n)


@pytest.mark.parametrize("statdist", [False, True], ids=lambda x: f"statdist={x}")
def test_expected_counts_stationary(setting, sparse_mode, statdist):
    if sparse_mode:
        setting = to_sparse_setting(setting, 10000)
    T, mu, v, L, R = setting
    N = 20
    D_mu = diags(mu, 0)
    EC_n = expected_counts_stationary(T, N, mu=mu if statdist else None)
    EC_true = N * D_mu.dot(T)
    assert_allclose(EC_true, EC_n)
