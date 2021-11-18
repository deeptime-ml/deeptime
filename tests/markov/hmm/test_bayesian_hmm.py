import unittest

import numpy as np
import pytest
from deeptime.markov.hmm._hmm_bindings.util import count_matrix

import deeptime
from deeptime.data import double_well_discrete
from deeptime.markov.hmm import MaximumLikelihoodHMM, BayesianHMM, BayesianHMMPosterior
from deeptime.util.stats import confidence_interval
from deeptime.util.types import ensure_dtraj_list
from numpy.testing import assert_, assert_equal


class BHMMScenario:

    def __init__(self, init_dist_prior, tmat_prior):
        self.obs = double_well_discrete().dtraj
        self.n_states = 2
        self.n_samples = 100
        self.lag = 10
        self.est = BayesianHMM.default(
            dtrajs=self.obs, n_hidden_states=self.n_states, lagtime=self.lag, reversible=True, n_samples=self.n_samples,
            initial_distribution_prior=init_dist_prior,
            transition_matrix_prior=tmat_prior
        )
        self.bhmm = self.est.fit(self.obs).fetch_model()


@pytest.fixture(scope="module", params=['mixed', 'uniform', 'sparse', np.arange(2)],
                ids=lambda x: f"init_prior={x if isinstance(x, str) else 'array'}")
def initial_distribution_priors(request):
    return request.param


@pytest.fixture(scope="module", params=['mixed', 'uniform', 'sparse', np.arange(4).reshape((2, 2))],
                ids=lambda x: f"tmat_prior={x if isinstance(x, str) else 'array'}")
def transition_matrix_priors(request):
    return request.param


@pytest.fixture(scope="module")
def bhmm_fixture(initial_distribution_priors, transition_matrix_priors):
    scenario = BHMMScenario(initial_distribution_priors, transition_matrix_priors)
    return scenario


def test_model_type(bhmm_fixture):
    assert_(isinstance(bhmm_fixture.bhmm, BayesianHMMPosterior))


def test_counting(bhmm_fixture):
    dtrajs = [np.array([0, 1, 0, 0, 0, 1, 2, 3, 0], dtype=np.int32)]
    C = count_matrix(dtrajs, lag=2, n_states=4)
    np.testing.assert_array_equal(C, np.array(
        [[2, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0]]
    ))


def test_reversible(bhmm_fixture):
    assert bhmm_fixture.bhmm.prior.transition_model.reversible
    assert all(s.transition_model.reversible for s in bhmm_fixture.bhmm)


def test_lag(bhmm_fixture):
    assert bhmm_fixture.bhmm.prior.lagtime == bhmm_fixture.lag
    lags = [sample.lagtime for sample in bhmm_fixture.bhmm]
    np.testing.assert_allclose(bhmm_fixture.lag, lags)


def test_n_states(bhmm_fixture):
    assert bhmm_fixture.bhmm.prior.n_hidden_states == bhmm_fixture.n_states
    assert all(s.n_hidden_states == bhmm_fixture.n_states for s in bhmm_fixture.bhmm)


def test_transition_matrix_samples(bhmm_fixture):
    Psamples = np.array([m.transition_model.transition_matrix for m in bhmm_fixture.bhmm])
    # shape
    assert np.array_equal(np.shape(Psamples), (bhmm_fixture.n_samples, bhmm_fixture.n_states, bhmm_fixture.n_states))
    # consistency
    import deeptime.markov.tools.analysis as msmana
    for P in Psamples:
        assert msmana.is_transition_matrix(P)
        assert msmana.is_reversible(P)


def test_transition_matrix_stats(bhmm_fixture):
    stats = bhmm_fixture.bhmm.gather_stats('transition_model/transition_matrix')
    import deeptime.markov.tools.analysis as msmana
    # mean
    Pmean = stats.mean
    # test shape and consistency
    assert np.array_equal(Pmean.shape, (bhmm_fixture.n_states, bhmm_fixture.n_states))
    assert msmana.is_transition_matrix(Pmean)
    # std
    Pstd = stats.std
    # test shape
    assert np.array_equal(Pstd.shape, (bhmm_fixture.n_states, bhmm_fixture.n_states))
    # conf
    L, R = stats.L, stats.R
    # test shape
    assert np.array_equal(L.shape, (bhmm_fixture.n_states, bhmm_fixture.n_states))
    assert np.array_equal(R.shape, (bhmm_fixture.n_states, bhmm_fixture.n_states))
    # test consistency
    assert np.all(L <= Pmean)
    assert np.all(R >= Pmean)


def test_eigenvalues_samples(bhmm_fixture):
    samples = np.array([m.transition_model.eigenvalues() for m in bhmm_fixture.bhmm])
    # shape
    assert_equal(np.shape(samples), (bhmm_fixture.n_samples, bhmm_fixture.n_states))
    # consistency
    for ev in samples:
        np.testing.assert_allclose(ev[0], 1.)
    for ev in samples:
        assert np.isclose(ev[0], 1)
        assert np.all(ev[1:] < 1.0)


def test_eigenvalues_stats(bhmm_fixture):
    stats = bhmm_fixture.bhmm.gather_stats('transition_model/eigenvalues', k=None)
    # mean
    mean = stats.mean
    # test shape and consistency
    assert np.array_equal(mean.shape, (bhmm_fixture.n_states,))
    assert np.isclose(mean[0], 1)
    assert np.all(mean[1:] < 1.0)
    # std
    std = stats.std
    # test shape
    assert np.array_equal(std.shape, (bhmm_fixture.n_states,))
    # conf
    L, R = stats.L, stats.R
    # test shape
    assert np.array_equal(L.shape, (bhmm_fixture.n_states,))
    assert np.array_equal(R.shape, (bhmm_fixture.n_states,))
    # test consistency
    tol = 1e-12
    assert np.all(L - tol <= mean)
    assert np.all(R + tol >= mean)


def test_eigenvectors_left_samples(bhmm_fixture):
    samples = np.array([m.transition_model.eigenvectors_left() for m in bhmm_fixture.bhmm])
    # shape
    np.testing.assert_equal(np.shape(samples), (bhmm_fixture.n_samples, bhmm_fixture.n_states, bhmm_fixture.n_states))
    # consistency
    for evec in samples:
        assert np.sign(evec[0, 0]) == np.sign(evec[0, 1])
        assert np.sign(evec[1, 0]) != np.sign(evec[1, 1])


def test_eigenvectors_left_stats(bhmm_fixture):
    samples = np.array([m.transition_model.eigenvectors_left() for m in bhmm_fixture.bhmm])
    # mean
    mean = samples.mean(axis=0)
    # test shape and consistency
    assert np.array_equal(mean.shape, (bhmm_fixture.n_states, bhmm_fixture.n_states))
    assert np.sign(mean[0, 0]) == np.sign(mean[0, 1])
    assert np.sign(mean[1, 0]) != np.sign(mean[1, 1])
    # std
    std = samples.std(axis=0)
    # test shape
    assert np.array_equal(std.shape, (bhmm_fixture.n_states, bhmm_fixture.n_states))
    # conf
    L, R = confidence_interval(samples)
    # test shape
    assert np.array_equal(L.shape, (bhmm_fixture.n_states, bhmm_fixture.n_states))
    assert np.array_equal(R.shape, (bhmm_fixture.n_states, bhmm_fixture.n_states))
    # test consistency
    tol = 1e-12
    assert np.all(L - tol <= mean)
    assert np.all(R + tol >= mean)


def test_eigenvectors_right_samples(bhmm_fixture):
    stats = bhmm_fixture.bhmm.gather_stats('transition_model/eigenvectors_right', store_samples=True)
    # shape
    np.testing.assert_equal(np.shape(stats.samples),
                            (bhmm_fixture.n_samples, bhmm_fixture.n_states, bhmm_fixture.n_states))
    # consistency
    for evec in stats.samples:
        assert np.sign(evec[0, 0]) == np.sign(evec[1, 0])
        assert np.sign(evec[0, 1]) != np.sign(evec[1, 1])

    # mean
    mean = stats.mean
    # test shape and consistency
    np.testing.assert_equal(mean.shape, (bhmm_fixture.n_states, bhmm_fixture.n_states))
    assert np.sign(mean[0, 0]) == np.sign(mean[1, 0])
    assert np.sign(mean[0, 1]) != np.sign(mean[1, 1])
    # std
    std = stats.std
    # test shape
    assert np.array_equal(std.shape, (bhmm_fixture.n_states, bhmm_fixture.n_states))
    # conf
    L, R = stats.L, stats.R
    # test shape
    assert np.array_equal(L.shape, (bhmm_fixture.n_states, bhmm_fixture.n_states))
    assert np.array_equal(R.shape, (bhmm_fixture.n_states, bhmm_fixture.n_states))
    # test consistency
    tol = 1e-12
    assert np.all(L - tol <= mean)
    assert np.all(R + tol >= mean)


def test_stationary_distribution_samples(bhmm_fixture):
    samples = np.array([m.transition_model.stationary_distribution for m in bhmm_fixture.bhmm])
    # shape
    assert np.array_equal(np.shape(samples), (bhmm_fixture.n_samples, bhmm_fixture.n_states))
    # consistency
    for mu in samples:
        assert np.isclose(mu.sum(), 1.0)
        assert np.all(mu > 0.0)


def test_stationary_distribution_stats(bhmm_fixture):
    samples = np.array([m.transition_model.stationary_distribution for m in bhmm_fixture.bhmm])
    tol = 1e-12
    # mean
    mean = samples.mean(axis=0)
    # test shape and consistency
    assert np.array_equal(mean.shape, (bhmm_fixture.n_states,))
    assert np.isclose(mean.sum(), 1.0)
    assert np.all(mean > 0.0)
    assert np.max(np.abs(mean[0] - mean[1])) < 0.05
    # std
    std = samples.std(axis=0)
    # test shape
    assert np.array_equal(std.shape, (bhmm_fixture.n_states,))
    # conf
    L, R = confidence_interval(samples)
    # test shape
    assert np.array_equal(L.shape, (bhmm_fixture.n_states,))
    assert np.array_equal(R.shape, (bhmm_fixture.n_states,))
    # test consistency
    assert np.all(L - tol <= mean)
    assert np.all(R + tol >= mean)


def test_timescales_samples(bhmm_fixture):
    samples = np.array([m.transition_model.timescales() for m in bhmm_fixture.bhmm])
    # shape
    np.testing.assert_equal(np.shape(samples), (bhmm_fixture.n_samples, bhmm_fixture.n_states - 1))
    # consistency
    for l in samples:
        assert np.all(l > 0.0)


def test_timescales_stats(bhmm_fixture):
    stats = bhmm_fixture.bhmm.gather_stats('transition_model/timescales')
    mean = stats.mean
    # test shape and consistency
    assert np.array_equal(mean.shape, (bhmm_fixture.n_states - 1,))
    assert np.all(mean > 0.0)
    # std
    std = stats.std
    # test shape
    assert np.array_equal(std.shape, (bhmm_fixture.n_states - 1,))
    # conf
    L, R = stats.L, stats.R
    # test shape
    assert np.array_equal(L.shape, (bhmm_fixture.n_states - 1,))
    assert np.array_equal(R.shape, (bhmm_fixture.n_states - 1,))
    # test consistency
    assert np.all(L <= mean)
    assert np.all(R >= mean)


def test_lifetimes_samples(bhmm_fixture):
    samples = np.array([m.lifetimes for m in bhmm_fixture.bhmm])
    # shape
    assert np.array_equal(np.shape(samples), (bhmm_fixture.n_samples, bhmm_fixture.n_states))
    # consistency
    for l in samples:
        assert np.all(l > 0.0)


def test_lifetimes_stats(bhmm_fixture):
    stats = bhmm_fixture.bhmm.gather_stats('lifetimes')
    # mean
    mean = stats.mean
    # test shape and consistency
    assert np.array_equal(mean.shape, (bhmm_fixture.n_states,))
    assert np.all(mean > 0.0)
    # std
    std = stats.std
    # test shape
    assert np.array_equal(std.shape, (bhmm_fixture.n_states,))
    # conf
    L, R = stats.L, stats.R
    # test shape
    assert np.array_equal(L.shape, (bhmm_fixture.n_states,))
    assert np.array_equal(R.shape, (bhmm_fixture.n_states,))
    # test consistency
    assert np.all(L <= mean)
    assert np.all(R >= mean)


def test_submodel_simple(bhmm_fixture):
    # sanity check for submodel;
    dtrj = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0,
                     0, 2, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0,
                     1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 2, 0, 0, 1, 1, 2, 0, 1, 1, 1,
                     0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0])

    h = BayesianHMM.default(dtrj, n_hidden_states=3, lagtime=2).fit(dtrj, n_burn_in=5).fetch_model()
    hs = h.submodel_largest(directed=True, connectivity_threshold=5, observe_nonempty=True, dtrajs=dtrj)
    hss = h.submodel_populous(directed=True, connectivity_threshold=5, observe_nonempty=True, dtrajs=dtrj)

    models_to_check = [hs.prior] + hs.samples + [hss.prior] + hss.samples
    for i, m in enumerate(models_to_check):
        assert_equal(m.transition_model.timescales().shape[0], 1)
        assert_equal(m.transition_model.stationary_distribution.shape[0], 2)
        assert_equal(m.transition_model.transition_matrix.shape, (2, 2))


class TestBHMMPathological(unittest.TestCase):

    def test_2state_rev_step(self):
        obs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
        dtrajs = ensure_dtraj_list(obs)
        init_hmm = deeptime.markov.hmm.init.discrete.metastable_from_data(dtrajs, 2, 1, regularize=False)
        hmm = MaximumLikelihoodHMM(init_hmm, lagtime=1).fit(dtrajs).fetch_model()
        # this will generate disconnected count matrices and should fail:
        with self.assertRaises(NotImplementedError):
            BayesianHMM(hmm).fit(obs)

    def test_2state_nonrev_step(self):
        obs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
        init_hmm = deeptime.markov.hmm.init.discrete.metastable_from_data(obs, n_hidden_states=2, lagtime=1,
                                                                          regularize=False)
        mle = MaximumLikelihoodHMM(init_hmm, lagtime=1).fit(obs).fetch_model()
        bhmm = BayesianHMM(mle, reversible=False, n_samples=2000).fit(obs).fetch_model()
        tmatrix_samples = np.array([s.transition_model.transition_matrix for s in bhmm])
        std = tmatrix_samples.std(axis=0)
        assert np.all(std[0] > 0)
        assert np.max(np.abs(std[1])) < 1e-3

    def test_2state_rev_2step(self):
        obs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0], dtype=int)
        init_hmm = deeptime.markov.hmm.init.discrete.metastable_from_data(obs, n_hidden_states=2, lagtime=1,
                                                                          regularize=False)
        mle = MaximumLikelihoodHMM(init_hmm, lagtime=1).fit(obs).fetch_model()
        bhmm = BayesianHMM(mle, reversible=False, n_samples=100).fit(obs).fetch_model()
        tmatrix_samples = np.array([s.transition_model.transition_matrix for s in bhmm])
        std = tmatrix_samples.std(axis=0)
        assert np.all(std > 0)


class TestBHMMSpecialCases(unittest.TestCase):

    def test_separate_states(self):
        dtrajs = [np.array([0, 1, 1, 1, 1, 1, 0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1]),
                  np.array([2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2]), ]
        hmm_bayes = BayesianHMM.default(dtrajs, n_hidden_states=3, store_hidden=True,
                                        lagtime=1, separate=[0], n_samples=100) \
            .fit(dtrajs).fetch_model()
        # we expect zeros in all samples at the following indexes:
        pobs_zeros = ((0, 1, 2, 2, 2), (0, 0, 1, 2, 3))
        for i, s in enumerate(hmm_bayes.samples):
            np.testing.assert_allclose(s.output_probabilities[pobs_zeros], 0, err_msg=f"{i}")
        for sample in hmm_bayes.samples:
            strajs = sample.hidden_state_trajectories
            assert strajs[0][0] == 2
            assert strajs[0][6] == 2
