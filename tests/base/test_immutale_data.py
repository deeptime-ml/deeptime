import unittest
import numpy as np

from deeptime.base import Estimator, Model, InputFormatError


class EvilEstimator(Estimator):

    # fit accidentally writes to input array!
    def fit(self, x, y=None):
        if isinstance(x, (list, tuple)):
            x[0][0] = -1
        else:
            x[0] = -1
        return self


class MutableInputDataEstimator(Estimator):
    _MUTABLE_INPUT_DATA = True

    def fit(self, x):
        x[0] = 5
        return self


class WellBehavingEstimator(Estimator):

    def fit(self, x):
        self.y = x + 1
        return self


class TestImmutableData(unittest.TestCase):
    def setUp(self) -> None:
        self.est = EvilEstimator()

    def test_modifying_raises(self):
        data = np.arange(0, 10)

        with self.assertRaises(ValueError) as cm:
            self.est.fit(data)
        if not 'read-only' in cm.exception.args[0]:
            raise AssertionError('no read-only related value error')

    def test_input_data(self):
        data = [np.arange(0, 10)]

        with self.assertRaises(ValueError):
            self.est.fit(data)

    def test_illegal_input_data_format(self):
        """ should raise InputFormatError"""
        data = 'illegal'
        with self.assertRaises(InputFormatError):
            self.est.fit(data)

    def test_illegal_input_element(self):
        data = [np.ones(1), np.ones(1), (np.arange(2), )]
        with self.assertRaises(InputFormatError):
            self.est.fit(data)

    def test_flag_remains(self):
        x = np.empty(3)
        old_flag = x.flags.writeable
        try:
            self.est.fit(x)
        except ValueError:
            pass
        assert x.flags.writeable == old_flag

    def test_result_of_fit(self):
        """ fit should return estimator instance itself """
        result = WellBehavingEstimator().fit(np.empty(0))
        self.assertIsInstance(result, WellBehavingEstimator)

    def test_kw_data_passing(self):
        """ Estimator.fit(data, **kwargs) should allow for fit(data=foobar) calls """
        WellBehavingEstimator().fit(x=np.empty(0))

    def test_kw_data_passing_y_arg(self):
        class Supervised(WellBehavingEstimator):
            def fit(self, x, y=None, foobar=None):
                return super(Supervised, self).fit(x)
