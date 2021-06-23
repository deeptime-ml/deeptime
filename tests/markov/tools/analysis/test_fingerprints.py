r"""Unit test for the fingerprint API-functions

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import pytest
import numpy as np

from deeptime.data import birth_death_chain
from tests.markov.tools.numeric import assert_allclose

from deeptime.markov.tools.analysis import rdl_decomposition, timescales
from deeptime.markov.tools.analysis import fingerprint_correlation, fingerprint_relaxation
from deeptime.markov.tools.analysis import expectation, correlation, relaxation


@pytest.fixture
def fingerprints_data(sparse_mode):
    p = np.zeros(10)
    q = np.zeros(10)
    p[0:-1] = 0.5
    q[1:] = 0.5
    p[4] = 0.01
    q[6] = 0.1
    return birth_death_chain(q, p, sparse=sparse_mode)


@pytest.fixture
def observables():
    obs1 = np.zeros(10)
    obs1[0] = 1
    obs1[1] = 1
    obs2 = np.zeros(10)
    obs2[8] = 1
    obs2[9] = 1
    return obs1, obs2


def test_fingerprint_correlation(fingerprints_data, observables):
    obs1, obs2 = observables
    T = fingerprints_data.transition_matrix
    ts = timescales(T, k=4 if fingerprints_data.sparse else None)
    R, D, L = rdl_decomposition(T, k=4 if fingerprints_data.sparse else None)
    mu = fingerprints_data.stationary_distribution
    tau = 7.5
    if not fingerprints_data.sparse:
        """k=None, tau=1"""
        acorr_amp = np.dot(mu * obs1, R) * np.dot(L, obs1)
        tsn, acorr_ampn = fingerprint_correlation(T, obs1)
        assert_allclose(tsn, ts)
        assert_allclose(acorr_ampn, acorr_amp)

        """k=None, tau=7.5"""
        tau = tau
        tsn, acorr_ampn = fingerprint_correlation(T, obs1, tau=tau)
        assert_allclose(tsn, tau * ts)
        assert_allclose(acorr_ampn, acorr_amp)

    """k=4, tau=1"""
    k = 4
    acorr_amp = np.dot(mu * obs1, R[:, 0:k]) * np.dot(L[0:k, :], obs1)
    tsn, acorr_ampn = fingerprint_correlation(T, obs1, k=k)
    assert_allclose(tsn, ts[0:k])
    assert_allclose(acorr_ampn, acorr_amp)

    """k=4, tau=7.5"""
    tau = tau
    tsn, acorr_ampn = fingerprint_correlation(T, obs1, k=k, tau=tau)
    assert_allclose(tsn, tau * ts[0:k])
    assert_allclose(acorr_ampn, acorr_amp)

    """Cross-correlation"""

    if not fingerprints_data.sparse:
        """k=None, tau=1"""
        corr_amp = np.dot(mu * obs1, R) * np.dot(L, obs2)
        tsn, corr_ampn = fingerprint_correlation(T, obs1, obs2=obs2)
        assert_allclose(tsn, ts)
        assert_allclose(corr_ampn, corr_amp)

        """k=None, tau=7.5"""
        tau = tau
        tsn, corr_ampn = fingerprint_correlation(T, obs1, obs2=obs2, tau=tau)
        assert_allclose(tsn, tau * ts)
        assert_allclose(corr_ampn, corr_amp)

    """k=4, tau=1"""
    corr_amp = np.dot(mu * obs1, R[:, 0:k]) * np.dot(L[0:k, :], obs2)
    tsn, corr_ampn = fingerprint_correlation(T, obs1, obs2=obs2, k=k)
    assert_allclose(tsn, ts[0:k])
    assert_allclose(corr_ampn, corr_amp)

    """k=4, tau=7.5"""
    tsn, corr_ampn = fingerprint_correlation(T, obs1, obs2=obs2, k=k, tau=tau)
    assert_allclose(tsn, tau * ts[0:k])
    assert_allclose(corr_ampn, corr_amp)


def test_fingerprint_relaxation(fingerprints_data, observables):
    obs1, obs2 = observables
    T = fingerprints_data.transition_matrix
    ts = timescales(T, k=4 if fingerprints_data.sparse else None)
    R, D, L = rdl_decomposition(T, k=4 if fingerprints_data.sparse else None)
    """Initial vector for relaxation"""
    p0 = np.zeros(10)
    p0[0:4] = 0.25
    if not fingerprints_data.sparse:
        """k=None"""
        relax_amp = np.dot(p0, R) * np.dot(L, obs1)
        tsn, relax_ampn = fingerprint_relaxation(T, p0, obs1)
        assert_allclose(tsn, ts)
        assert_allclose(relax_ampn, relax_amp)

    """k=4"""
    k = 4
    relax_amp = np.dot(p0, R[:, 0:k]) * np.dot(L[0:k, :], obs1)
    tsn, relax_ampn = fingerprint_relaxation(T, p0, obs1, k=k)
    assert_allclose(tsn, ts[0:k])
    assert_allclose(relax_ampn, relax_amp)


def test_expectation(fingerprints_data):
    obs1 = np.zeros(10)
    obs1[0] = 1
    obs1[1] = 1
    exp = np.dot(fingerprints_data.stationary_distribution, obs1)
    expn = expectation(fingerprints_data.transition_matrix, obs1)
    assert_allclose(exp, expn)


def test_correlation(fingerprints_data, observables):
    obs1, obs2 = observables
    T = fingerprints_data.transition_matrix
    R, D, L = rdl_decomposition(T, k=4 if fingerprints_data.sparse else None)
    mu = fingerprints_data.stationary_distribution
    times = np.array([1, 5, 10, 20, 100])

    ev = np.diagonal(D)
    ev_t = ev[np.newaxis, :] ** times[:, np.newaxis]

    if not fingerprints_data.sparse:
        """k=None"""
        acorr_amp = np.dot(mu * obs1, R) * np.dot(L, obs1)
        acorr = np.dot(ev_t, acorr_amp)
        acorrn = correlation(T, obs1, times=times)
        assert_allclose(acorrn, acorr)

    """k=4"""
    k = 4
    acorr_amp = np.dot(mu * obs1, R[:, 0:k]) * np.dot(L[0:k, :], obs1)
    acorr = np.dot(ev_t[:, 0:k], acorr_amp)
    acorrn = correlation(T, obs1, times=times, k=k)
    assert_allclose(acorrn, acorr)

    """Cross-correlation"""

    if not fingerprints_data.sparse:
        """k=None"""
        corr_amp = np.dot(mu * obs1, R) * np.dot(L, obs2)
        corr = np.dot(ev_t, corr_amp)
        corrn = correlation(T, obs1, obs2=obs2, times=times)
        assert_allclose(corrn, corr)

    """k=4"""
    k = k
    corr_amp = np.dot(mu * obs1, R[:, 0:k]) * np.dot(L[0:k, :], obs2)
    corr = np.dot(ev_t[:, 0:k], corr_amp)
    corrn = correlation(T, obs1, obs2=obs2, times=times, k=k)
    assert_allclose(corrn, corr)


def test_relaxation(fingerprints_data):
    T = fingerprints_data.transition_matrix
    R, D, L = rdl_decomposition(T, k=4 if fingerprints_data.sparse else None)

    times = np.array([1, 5, 10, 20, 100])
    ev = np.diagonal(D)
    ev_t = ev[np.newaxis, :] ** times[:, np.newaxis]

    obs = np.zeros(10)
    obs[0] = 1
    obs[1] = 1

    p0 = np.zeros(10)
    p0[:4] = 0.25

    if not fingerprints_data.sparse:
        """k=None"""
        relax_amp = np.dot(p0, R) * np.dot(L, obs)
        relax = np.dot(ev_t, relax_amp)
        relaxn = relaxation(T, p0, obs, times=times)
        assert_allclose(relaxn, relax)

    """k=4"""
    k = 4
    relax_amp = np.dot(p0, R[:, 0:k]) * np.dot(L[0:k, :], obs)
    relax = np.dot(ev_t[:, 0:k], relax_amp)
    relaxn = relaxation(T, p0, obs, k=k, times=times)
    assert_allclose(relaxn, relax)
