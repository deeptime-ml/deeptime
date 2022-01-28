import matplotlib.pyplot as plt
import pytest
from numpy.testing import assert_raises, assert_equal, assert_array_equal

from deeptime.markov.hmm import HiddenMarkovModel, GaussianOutputModel
from deeptime.markov.msm import MarkovStateModel
from deeptime.markov.plot import implied_timescales
from deeptime.markov.plot.implied_timescales import to_its_data


@pytest.fixture
def axis():
    f, ax = plt.subplots(1, 1)
    yield ax


def test_to_its_data_wrong_args():
    with assert_raises(ValueError):
        to_its_data(axis, [])

    with assert_raises(ValueError):
        to_its_data(axis, [object])

    msm = MarkovStateModel([[.9, .1], [.1, .9]])
    hmm = HiddenMarkovModel(msm, GaussianOutputModel(2, [0, 1], [1, 1]))

    with assert_raises(ValueError):
        to_its_data(axis, [msm, hmm])


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
