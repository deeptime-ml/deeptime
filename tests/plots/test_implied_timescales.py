import matplotlib.pyplot as plt
import pytest
from numpy.testing import assert_raises, assert_equal, assert_array_equal

from deeptime.data import double_well_discrete
from deeptime.markov.hmm import HiddenMarkovModel, GaussianOutputModel, MaximumLikelihoodHMM, init, BayesianHMM
from deeptime.markov.msm import MarkovStateModel, MaximumLikelihoodMSM, BayesianMSM
from deeptime.plots import plot_implied_timescales
from deeptime.plots.implied_timescales import to_its_data


@pytest.fixture
def figure():
    f, ax = plt.subplots(1, 1)
    yield f, ax


def test_to_its_data_wrong_args():
    with assert_raises(ValueError):
        to_its_data([])

    with assert_raises(AssertionError):
        to_its_data([object])


@pytest.mark.parametrize("model", [MarkovStateModel([[.9, .1], [.1, .9]]),
                                   HiddenMarkovModel(MarkovStateModel([[.9, .1], [.1, .9]]),
                                                     GaussianOutputModel(2, [0, 1], [1, 1]))],
                         ids=lambda x: x.__class__.__name__)
def test_to_its_data(model):
    data = to_its_data([model, model])
    assert_equal(data.lagtimes, [1, 1])
    assert_equal(data.n_lagtimes, 2)
    assert_equal(data.n_samples, 0)
    assert_equal(data.n_processes, 1)
    assert_array_equal(data.its[0], data.its[1])


def _generate_mlmsm(data):
    return lambda lagtime: MaximumLikelihoodMSM(lagtime=lagtime).fit_fetch(data)


def _generate_bmsm(data):
    return lambda lagtime: BayesianMSM().fit_fetch(_generate_mlmsm(data)(lagtime))


def _generate_hmm(data):
    return lambda lagtime: MaximumLikelihoodHMM(
        init.discrete.metastable_from_data(data, n_hidden_states=4, lagtime=lagtime),
        lagtime=lagtime, maxit=10, maxit_reversible=100
    ).fit_fetch(data)


def _generate_bhmm(data):
    return lambda lagtime: BayesianHMM.default(data, n_hidden_states=4, lagtime=lagtime, n_samples=10).fit_fetch(data)


@pytest.mark.parametrize("dw_model", [_generate_mlmsm, _generate_bmsm, _generate_hmm, _generate_bhmm])
def test_plot_its(figure, dw_model):
    f, ax = figure
    data = double_well_discrete().dtraj_n6good
    lagtimes = [1, 2, 5, 10, 15, 100]

    models = []
    for lagtime in lagtimes:
        models.append(dw_model(data)(lagtime))

    plot_implied_timescales(ax, models)
