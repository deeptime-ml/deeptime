import functools
from weakref import WeakKeyDictionary

from deeptime.util.platform import module_available


class cached_property(property):
    r"""
    Property that gets cached, obeys property api and can also be invalidated and overridden. Inspired from
    https://github.com/pydanny/cached-property/ and  https://stackoverflow.com/a/17330273.
    """
    _default_cache_entry = object()

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        super(cached_property, self).__init__(fget, fset, fdel, doc)
        self.cache = WeakKeyDictionary()

    def __get__(self, instance, owner):
        if instance is None:
            return self
        value = self.cache.get(instance, self._default_cache_entry)
        if value is self._default_cache_entry:
            value = self.fget(instance)
            self.cache[instance] = value
        return value

    def __set__(self, instance, value):
        self.cache[instance] = value

    def __delete__(self, instance):
        del self.cache[instance]

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)

    def invalidate(self):
        self.cache.clear()


def plotting_function(fn):  # pragma: no cover
    r""" Decorator marking a function that is a plotting utility. This will exclude it from coverage and test
    whether dependencies are installed. """

    @functools.wraps(fn)
    def wrapper(*args, **kw):
        if not module_available("matplotlib") or not module_available("networkx"):
            raise RuntimeError("Plotting functions require matplotlib and networkx to be installed.")
        return fn(*args, **kw)

    return wrapper
