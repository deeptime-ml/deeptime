import matplotlib.pyplot as plt
import pytest
from numpy.testing import assert_raises

from deeptime.markov.plot import implied_timescales


@pytest.fixture
def axis():
    f, ax = plt.subplots(1, 1)
    yield ax
    f.close()


def test_wrong_models(axis):
    with assert_raises(ValueError):
        implied_timescales(axis, [], None)

    implied_timescales(axis, [object], None)
