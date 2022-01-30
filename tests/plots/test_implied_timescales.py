import matplotlib.pyplot as plt
import pytest
from numpy.testing import assert_raises, assert_equal, assert_array_equal

from deeptime.data import double_well_discrete, double_well_2d
from deeptime.decomposition import TICA
from deeptime.markov.hmm import HiddenMarkovModel, GaussianOutputModel, MaximumLikelihoodHMM, init, BayesianHMM
from deeptime.markov.msm import MarkovStateModel, MaximumLikelihoodMSM, BayesianMSM
from deeptime.plots import plot_implied_timescales, ImpliedTimescalesData


@pytest.fixture
def figure():
    f, ax = plt.subplots(1, 1)
    yield f, ax


def test_to_its_data_wrong_args():
    with assert_raises(ValueError):
        ImpliedTimescalesData.from_models([])

    with assert_raises(AssertionError):
        ImpliedTimescalesData.from_models([object])


@pytest.mark.parametrize("model", [MarkovStateModel([[.9, .1], [.1, .9]]),
                                   HiddenMarkovModel(MarkovStateModel([[.9, .1], [.1, .9]]),
                                                     GaussianOutputModel(2, [0, 1], [1, 1]))],
                         ids=lambda x: x.__class__.__name__)
def test_to_its_data(model):
    data = ImpliedTimescalesData.from_models([model, model])
    assert_equal(data.lagtimes, [1, 1])
    assert_equal(data.n_lagtimes, 2)
    assert_equal(data.n_samples, 0)
    assert_equal(data.n_processes, 1)
    assert_array_equal(data.its[0], data.its[1])


def doublewell_mlmsm(lagtime, count_mode='sliding'):
    return MaximumLikelihoodMSM(lagtime=lagtime).fit_fetch(double_well_discrete().dtraj_n6good, count_mode=count_mode)


def doublewell_bmsm(lagtime):
    return BayesianMSM().fit_fetch(doublewell_mlmsm(lagtime, count_mode='effective'))


def doublewell_hmm(lagtime):
    return MaximumLikelihoodHMM(
        init.discrete.metastable_from_data(double_well_discrete().dtraj_n6good, n_hidden_states=4, lagtime=lagtime),
        lagtime=lagtime, maxit=10, maxit_reversible=100
    ).fit_fetch(double_well_discrete().dtraj_n6good)


def doublewell_bhmm(lagtime):
    return BayesianHMM.default(double_well_discrete().dtraj_n6good, n_hidden_states=4, lagtime=lagtime, n_samples=10) \
        .fit_fetch(double_well_discrete().dtraj_n6good)


def doublewell_tica(lagtime):
    return TICA(lagtime=lagtime).fit_fetch(double_well_2d().trajectory([[0, 0]], length=200))


models = [doublewell_mlmsm, doublewell_bmsm, doublewell_bhmm, doublewell_hmm, doublewell_tica]


@pytest.mark.parametrize("dw_model", models, ids=lambda x: x.__name__)
def test_plot_its(figure, dw_model):
    f, ax = figure
    lagtimes = [1, 2, 5, 10, 15, 100]

    models = []
    for lagtime in lagtimes:
        models.append(dw_model(lagtime))

    plot_implied_timescales(ax, ImpliedTimescalesData.from_models(models))
