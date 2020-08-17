import numpy as np

from sktime.util import QuantityStatistics, module_available


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

    samples = [MyObject() for _ in range(100000)]
    stats = QuantityStatistics.gather(samples, 'yield_self.rnd', store_samples=False, delimiter='.')
    np.testing.assert_almost_equal(stats.mean, 5., decimal=2)
    np.testing.assert_almost_equal(stats.std, .1, decimal=2)
    np.testing.assert_almost_equal(stats.L, 5. - 0.2, decimal=2)
    np.testing.assert_almost_equal(stats.R, 5. + 0.2, decimal=2)
