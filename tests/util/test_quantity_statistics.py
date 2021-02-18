import pytest
from numpy.testing import assert_equal, assert_

from deeptime.util import QuantityStatistics
from deeptime.util.stats import evaluate_samples


class A:

    def method(self, input=1):
        return input * 2


class B:

    def __init__(self, a: A):
        self._a = a
        self.aattr = self._a

    @property
    def aprop(self):
        return self._a

    def ameth(self):
        return self._a


@pytest.fixture
def lots_of_b():
    return [B(A()) for _ in range(10000)]


@pytest.mark.parametrize("mode", ["attr", "prop", "meth"])
def test_quantity_statistics(lots_of_b, mode):
    qs = QuantityStatistics.gather(lots_of_b, f"a{mode}.method", delimiter='.', input=4)
    assert_equal(qs.mean, 8)
    assert_equal(qs.std, 0)
    assert_equal(qs.L, 8)
    assert_equal(qs.R, 8)

    a_list_of_a = evaluate_samples(lots_of_b, f"a{mode}")
    for a, b in zip(a_list_of_a, lots_of_b):
        assert_(a is b.aprop)
