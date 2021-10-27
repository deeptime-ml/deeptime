import numpy as np
import pytest

from deeptime.util.platform import module_available
from deeptime.util.stats import QuantityStatistics, confidence_interval


def test_module_available():
    np.testing.assert_(module_available('numpy'))
    np.testing.assert_(not module_available('thismodulecertainlydoesnotexist'))


def test_gather():
    class MyObject(object):

        def __init__(self):
            self._randn = np.random.normal(loc=5., scale=0.1)

        def yield_self(self):
            return self

        @property
        def rnd(self):
            return self._randn

    samples = [MyObject() for _ in range(10000)]
    stats = QuantityStatistics.gather(samples, 'yield_self.rnd', store_samples=False, delimiter='.')
    np.testing.assert_almost_equal(stats.mean, 5., decimal=2)
    np.testing.assert_almost_equal(stats.std, .1, decimal=2)
    print(stats.L, stats.mean, stats.R)
    np.testing.assert_almost_equal(stats.R, 5. + 0.2, decimal=2)
    np.testing.assert_almost_equal(stats.L, 5. - 0.2, decimal=2)


@pytest.mark.parametrize('conf', [.5, .55, .91])
def test_confidence_interval(conf):
    g = np.random.default_rng(seed=42)
    data = g.normal(size=(300000,))
    l, r = confidence_interval(data, conf=conf)
    data_inside = data[(data >= l) & (data <= r)]
    np.testing.assert_almost_equal(data_inside.size / data.size, conf)
