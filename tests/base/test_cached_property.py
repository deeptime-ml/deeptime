import unittest
from deeptime.util.decorators import cached_property
import numpy as np

class MyClass(object):

    def __init__(self):
        self.computed_property = False

    @cached_property
    def expensive_property(self):
        self.computed_property = True
        self._value = 5
        return self._value

    @expensive_property.setter
    def expensive_property(self, value):
        self._value = value

    def invalidate(self):
        self.computed_property = False
        for member in self.__class__.__dict__.values():
            if isinstance(member, cached_property):
                member.invalidate()


class TestCachedProperty(unittest.TestCase):

    def test_property_cached(self):
        x = MyClass()
        np.testing.assert_equal(x.expensive_property, 5)
        np.testing.assert_(x.computed_property)
        x.computed_property = False
        np.testing.assert_equal(x.expensive_property, 5)
        np.testing.assert_(not x.computed_property)

    def test_property_override(self):
        x = MyClass()
        np.testing.assert_equal(x.expensive_property, 5)
        np.testing.assert_(x.computed_property)
        x.expensive_property = 100
        x.computed_property = False
        np.testing.assert_equal(x.expensive_property, 100)
        np.testing.assert_(not x.computed_property)

    def test_property_invalidate(self):
        x = MyClass()
        x.expensive_property = 1000
        np.testing.assert_equal(x.expensive_property, 1000)
        np.testing.assert_(not x.computed_property)
        del x.expensive_property
        np.testing.assert_equal(x.expensive_property, 5)
        np.testing.assert_(x.computed_property)

    def test_property_invalidate_method(self):
        x = MyClass()
        np.testing.assert_equal(x.expensive_property, 5)
        np.testing.assert_equal(x.computed_property, True)
        x.expensive_property = 1000
        np.testing.assert_equal(x.expensive_property, 1000)
        x.invalidate()
        np.testing.assert_equal(x.computed_property, False)
        np.testing.assert_equal(x.expensive_property, 5)
        np.testing.assert_equal(x.computed_property, True)
