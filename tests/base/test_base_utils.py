import numpy as np
import unittest

from sktime.markov import Q_
from sktime.util import QuantityStatistics


class BaseUtilsTest(unittest.TestCase):

    def test_gather(self):
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

    def test_with_units(self):
        samples = [5. * Q_('ms') for _ in range(1000)]
        stats = QuantityStatistics.gather(samples)
        np.testing.assert_equal(stats.mean, 5. * Q_('ms'))
        np.testing.assert_equal(stats.std, 0. * Q_('ms'))
        np.testing.assert_equal(stats.L, 5. * Q_('ms'))
        np.testing.assert_equal(stats.R, 5. * Q_('ms'))
