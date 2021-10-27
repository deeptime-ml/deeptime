r"""Unit test for decomposition functions in api.py

.. moduleauthor:: Benjamin Trendelkamp-Schroer<benjamin DOT trendelkamp-schorer AT fu-berlin DOT de>
.. moduleauthor:: clonker (the refactors)
"""
import warnings

import numpy as np
import pytest
from numpy.testing import assert_raises
from scipy.linalg import eigvals as _eigvals
from scipy.sparse import issparse

from tests.testing_utilities import nullcontext

from deeptime.data import birth_death_chain
from deeptime.markov.tools.analysis import rdl_decomposition, timescales
from deeptime.markov.tools.analysis import stationary_distribution, eigenvalues, eigenvectors
from deeptime.util.exceptions import SpectralWarning, ImaginaryEigenValueWarning
from tests.markov.tools.numeric import assert_allclose


def eigvals(mat):
    if issparse(mat):
        mat = mat.toarray()
    return _eigvals(mat)


@pytest.fixture(params=[None, 40], ids=lambda x: f"ncv={x}")
def ncv_values(request):
    yield request.param


@pytest.fixture(params=[None, 10], ids=lambda x: f"k={x}")
def scenario(request, sparse_mode):
    dim = 100

    """Set up meta-stable birth-death chain"""
    p = np.zeros(dim)
    p[0:-1] = 0.5

    q = np.zeros(dim)
    q[1:] = 0.5

    p[dim // 2 - 1] = 0.001
    q[dim // 2 + 1] = 0.001

    bdc = birth_death_chain(q, p, sparse=sparse_mode)
    yield request.param, bdc


def test_statdist(scenario):
    _, bdc = scenario
    P = bdc.transition_matrix
    mu = bdc.stationary_distribution
    mun = stationary_distribution(P)
    assert_allclose(mu, mun)


def test_eigenvalues(scenario, ncv_values):
    k, bdc = scenario
    P = bdc.transition_matrix
    ev = eigvals(P)
    """Sort with decreasing magnitude"""
    ev = ev[np.argsort(np.abs(ev))[::-1]]

    """k=None"""
    with assert_raises(ValueError) if bdc.sparse and k is None else nullcontext():
        evn = eigenvalues(P, ncv=ncv_values, k=k)
        assert_allclose(ev[:k], evn)


def test_eigenvectors(scenario, ncv_values):
    k, bdc = scenario
    P = bdc.transition_matrix
    ev = eigvals(P)
    ev = ev[np.argsort(np.abs(ev))[::-1]]

    Dn = np.diag(ev)
    Dnk = Dn[:, :k][:k, :]
    with assert_raises(ValueError) if bdc.sparse and k is None else nullcontext():
        # right eigenvectors
        Rn = eigenvectors(P, k=k, ncv=ncv_values)
        assert_allclose(P @ Rn, Rn @ Dnk)
        # left eigenvectors
        Ln = eigenvectors(P, right=False, k=k, ncv=ncv_values).T
        assert_allclose(Ln.T @ P, Dnk @ Ln.T)
        # orthogonality
        Xn = Ln.T @ Rn
        di = np.diag_indices(Xn.shape[0] if k is None else k)
        Xn[di] = 0.0
        assert_allclose(Xn, 0)


def test_eigenvalues_reversible(scenario, ncv_values):
    k, bdc = scenario
    P = bdc.transition_matrix
    ev = eigvals(P)
    """Sort with decreasing magnitude"""
    ev = ev[np.argsort(np.abs(ev))[::-1]]

    with assert_raises(ValueError) if bdc.sparse and k is None else nullcontext():
        """reversible without given mu"""
        evn = eigenvalues(P, reversible=True, k=k, ncv=ncv_values)
        assert_allclose(ev[:k], evn)

        """reversible with given mu"""
        evn = eigenvalues(P, reversible=True, mu=bdc.stationary_distribution, k=k, ncv=ncv_values)
        assert_allclose(ev[:k], evn)


def test_eigenvectors_reversible(scenario, ncv_values):
    k, bdc = scenario
    P = bdc.transition_matrix

    ev = eigvals(P)
    ev = ev[np.argsort(np.abs(ev))[::-1]]
    Dn = np.diag(ev)
    Dnk = Dn[:, :k][:k, :]
    with assert_raises(ValueError) if bdc.sparse and k is None else nullcontext():
        # right eigenvectors
        Rn = eigenvectors(P, k=k, reversible=True, ncv=ncv_values)
        assert_allclose(P @ Rn, Rn @ Dnk)
        # left eigenvectors
        Ln = eigenvectors(P, right=False, k=k, reversible=True, ncv=ncv_values).T
        assert_allclose(Ln.T @ P, Dnk @ Ln.T)
        # orthogonality
        Xn = Ln.T @ Rn
        di = np.diag_indices(Xn.shape[0] if k is None else k)
        Xn[di] = 0.0
        assert_allclose(Xn, 0)

        Rn = eigenvectors(P, k=k, ncv=ncv_values, reversible=True, mu=bdc.stationary_distribution)
        assert_allclose(ev[:k][np.newaxis, :] * Rn, P.dot(Rn))

        Ln = eigenvectors(P, right=False, k=k, ncv=ncv_values, reversible=True, mu=bdc.stationary_distribution).T
        assert_allclose(P.transpose().dot(Ln), ev[:k][np.newaxis, :] * Ln)


@pytest.mark.parametrize("norm", ['auto', 'standard', 'reversible'], ids=lambda x: f"norm={x}")
@pytest.mark.parametrize("statdist", [False, True], ids=lambda x: f"statdist={x}")
@pytest.mark.parametrize("reversible", [False, True], ids=lambda x: f"reversible={x}")
def test_rdl_decomposition(scenario, ncv_values, norm, statdist, reversible):
    k, bdc = scenario
    dim = bdc.q.shape[0]
    P = bdc.transition_matrix
    mu = bdc.stationary_distribution

    with assert_raises(ValueError) if bdc.sparse and k is None else nullcontext():
        Rn, Dn, Ln = rdl_decomposition(P, reversible=reversible, norm=norm,
                                       ncv=ncv_values, k=k, mu=mu if statdist else None)
        Xn = Ln @ Rn
        """Right-eigenvectors"""
        assert_allclose(P @ Rn, Rn @ Dn)
        """Left-eigenvectors"""
        assert_allclose(Ln @ P, Dn @ Ln)
        """Orthonormality"""
        assert_allclose(Xn, np.eye(dim if k is None else k))
        """Probability vector"""
        assert_allclose(np.sum(Ln[0, :]), 1.0)
        if norm == 'standard' and reversible:
            """Standard l2-normalization of right eigenvectors except dominant one"""
            Yn = Rn.T @ Rn
            assert_allclose(np.diag(Yn)[1:], 1.0)
        if norm == 'reversible':
            """Reversibility"""
            assert_allclose(Ln.transpose(), mu[:, np.newaxis] * Rn)


@pytest.mark.parametrize("statdist", [False, True], ids=lambda x: f"{x}")
@pytest.mark.parametrize("reversible", [False, True], ids=lambda x: f"{x}")
@pytest.mark.parametrize("tau", [1, 7], ids=lambda x: f"{x}")
def test_timescales(scenario, ncv_values, tau, statdist, reversible):
    k, bdc = scenario
    P = bdc.transition_matrix
    mu = bdc.stationary_distribution
    ev = eigvals(P)
    """Sort with decreasing magnitude"""
    ev = ev[np.argsort(np.abs(ev))[::-1]]
    ts = -1.0 / np.log(np.abs(ev))

    with assert_raises(ValueError) if bdc.sparse and k is None else nullcontext():
        tsn = timescales(P, k=k, tau=tau, mu=mu if statdist else None, reversible=reversible)
        assert_allclose(tau * ts[1:k], tsn[1:k])


def test_timescales_inf():
    """Multiple eigenvalues of magnitude one, eigenvalues with non-zero imaginary part"""
    W = np.array([[0, 1], [1, 0]])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('ignore')
        warnings.simplefilter('always', category=SpectralWarning)
        tsn = timescales(W)
        assert_allclose(tsn, np.array([np.inf, np.inf]))
        assert issubclass(w[-1].category, SpectralWarning)


def test_timescales_inf2():
    """Eigenvalues with non-zero imaginary part"""
    ts = np.array([np.inf, 0.971044, 0.971044])

    T = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    P = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('ignore')
        warnings.simplefilter('always', category=ImaginaryEigenValueWarning)
        tsn = timescales(0.5 * T + 0.5 * P)
        assert_allclose(tsn, ts)
        assert issubclass(w[-1].category, ImaginaryEigenValueWarning)
