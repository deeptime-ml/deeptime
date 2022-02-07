import matplotlib.pyplot as plt
import pytest
import numpy as np
from numpy.testing import assert_raises, assert_equal, assert_array_equal, assert_

from deeptime.data import double_well_discrete, double_well_2d
from deeptime.decomposition import TICA
from deeptime.markov.hmm import HiddenMarkovModel, GaussianOutputModel, MaximumLikelihoodHMM, init, BayesianHMM
from deeptime.markov.msm import MarkovStateModel, MaximumLikelihoodMSM, BayesianMSM
from deeptime.markov import BayesianMSMPosterior
from deeptime.plots import plot_implied_timescales
from deeptime.util.validation import implied_timescales


@pytest.fixture
def figure():
    f, ax = plt.subplots(1, 1)
    yield f, ax


def test_to_its_data_wrong_args():
    with assert_raises(ValueError):
        implied_timescales([])

    with assert_raises(AssertionError):
        implied_timescales([object])


@pytest.mark.parametrize("model", [MarkovStateModel([[.9, .1], [.1, .9]]),
                                   HiddenMarkovModel(MarkovStateModel([[.9, .1], [.1, .9]]),
                                                     GaussianOutputModel(2, [0, 1], [1, 1]))],
                         ids=lambda x: x.__class__.__name__)
def test_to_its_data(model):
    data = implied_timescales([model, model])
    assert_equal(data.lagtimes, [1, 1])
    assert_equal(data.n_lagtimes, 2)
    assert_equal(data.max_n_samples, 0)
    assert_equal(data.max_n_processes, 1)
    timescales = data.timescales_for_process(0)
    assert_equal(len(timescales), 2)
    assert_array_equal(timescales[0], timescales[1])
    with assert_raises(AssertionError):
        data.timescales_for_process(1)


def doublewell_mlmsm(lagtime, n_samples, count_mode='sliding'):
    return MaximumLikelihoodMSM(lagtime=lagtime).fit_fetch(double_well_discrete().dtraj_n6good, count_mode=count_mode)


def doublewell_bmsm(lagtime, n_samples):
    return BayesianMSM(n_samples=n_samples)\
        .fit_fetch(doublewell_mlmsm(lagtime, 0, count_mode='effective'))


def doublewell_hmm(lagtime, n_samples):
    return MaximumLikelihoodHMM(
        init.discrete.metastable_from_data(double_well_discrete().dtraj_n6good, n_hidden_states=4, lagtime=lagtime),
        lagtime=lagtime, maxit=10, maxit_reversible=100
    ).fit_fetch(double_well_discrete().dtraj_n6good)


def doublewell_bhmm(lagtime, n_samples):
    return BayesianHMM.default(double_well_discrete().dtraj_n6good, n_hidden_states=4, lagtime=lagtime,
                               n_samples=n_samples) \
        .fit_fetch(double_well_discrete().dtraj_n6good)


def doublewell_tica(lagtime, n_samples):
    return TICA(lagtime=lagtime).fit_fetch(double_well_2d().trajectory([[0, 0]], length=200))


doublewell_models = [
    doublewell_mlmsm,
    doublewell_bmsm,
    doublewell_bhmm,
    doublewell_hmm,
    doublewell_tica
]


@pytest.mark.parametrize("dw_model", doublewell_models, ids=lambda x: x.__name__)
def test_plot_its(figure, dw_model):
    f, ax = figure
    lagtimes = [1, 2, 5, 10, 15, 100]
    n_samples = [5, 10, 2, 30, 100, 20]

    models = []
    for lagtime, n in zip(lagtimes, n_samples):
        models.append(dw_model(lagtime, n))
    data = implied_timescales(models)
    assert_(data.max_n_processes >= 2)
    assert_equal(data.has_samples, isinstance(models[0], BayesianMSMPosterior))
    assert_equal(data.max_n_samples, 100 if data.has_samples else 0)
    if data.has_samples:
        assert_equal(data.n_samples(0, 0), n_samples[0])
        assert_equal(data.n_samples(1, 0), n_samples[1])
        assert_equal(data.n_samples(2, 0), n_samples[2])
        assert_equal(data.n_samples(3, 0), n_samples[3])
        assert_equal(data.n_samples(4, 0), n_samples[4])
        assert_equal(data.n_samples(5, 0), n_samples[5])
    plot_implied_timescales(data, ax=ax)


def test_its_mixed_est_sort(figure):
    f, ax = figure

    mlmsm5 = MaximumLikelihoodMSM(lagtime=5).fit_fetch(double_well_discrete().dtraj_n6good, count_mode='effective')
    mlmsm6 = MaximumLikelihoodMSM(lagtime=6).fit_fetch(double_well_discrete().dtraj_n6good, count_mode='effective')
    mlmsm7 = MaximumLikelihoodMSM(lagtime=7).fit_fetch(double_well_discrete().dtraj_n6good, count_mode='effective')
    mlmsm8 = MaximumLikelihoodMSM(lagtime=8).fit_fetch(double_well_discrete().dtraj_n6good, count_mode='effective')
    mlmsm9 = MaximumLikelihoodMSM(lagtime=9).fit_fetch(double_well_discrete().dtraj_n6good, count_mode='effective')
    mlmsm10 = MaximumLikelihoodMSM(lagtime=10).fit_fetch(double_well_discrete().dtraj_n6good, count_mode='effective')
    models = [
        BayesianMSM(n_samples=25).fit_fetch(mlmsm8),
        mlmsm5,
        BayesianMSM(n_samples=15).fit_fetch(mlmsm6),
        mlmsm10,
        mlmsm7,
        BayesianMSM(n_samples=13).fit_fetch(mlmsm9)
    ]
    data = implied_timescales(models)
    assert_equal(data.n_lagtimes, len(models))
    assert_equal(data.lagtimes, [5, 6, 7, 8, 9, 10])
    assert_equal(data.max_n_samples, 25)
    assert_equal(data.n_samples(0, 0), 0)
    assert_equal(data.n_samples(1, 0), 15)
    assert_equal(data.n_samples(2, 0), 0)
    assert_equal(data.n_samples(3, 0), 25)
    assert_equal(data.n_samples(4, 0), 13)
    assert_equal(data.n_samples(5, 0), 0)
    data.plot(ax=ax)


@pytest.mark.parametrize("bayesian", [False, True])
def test_decayed_process(bayesian):
    dtraj = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 2, 1])  # state "2" is only visible with lagtime 1
    msm1 = MaximumLikelihoodMSM(lagtime=1).fit_fetch(dtraj)
    msm2 = MaximumLikelihoodMSM(lagtime=2).fit_fetch(dtraj)
    msm3 = MaximumLikelihoodMSM(lagtime=3).fit_fetch(dtraj)
    models = [msm1, msm2, msm3]
    if bayesian:
        models = [BayesianMSM(n_samples=15).fit_fetch(msm, ignore_counting_mode=True) for msm in models]
    data = implied_timescales(models)
    assert_equal(data.max_n_processes, 2)
    assert_equal(data.max_n_samples, 0 if not bayesian else 15)
    assert_equal(data.n_lagtimes, 3)
    assert_equal(np.count_nonzero(np.isnan(data.timescales_for_process(0))), 0)  # 0 nans
    assert_equal(np.count_nonzero(np.isnan(data.timescales_for_process(1))), 2)  # 2 nans
    if bayesian:
        assert_equal(np.count_nonzero(np.isnan(data.samples_for_process(0))), 0)  # 0 nans
        assert_equal(np.count_nonzero(np.isnan(data.samples_for_process(1))), 2 * 15)  # 0 nans
