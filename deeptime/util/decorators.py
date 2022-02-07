import functools
import typing
import warnings
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


def plotting_function(requires_networkx=False):
    r""" Decorator marking a function that is a plotting utility. This will test whether dependencies are installed. """

    def factory(fn: typing.Callable) -> typing.Callable:
        @functools.wraps(fn)
        def call(*args, **kw):
            if not module_available("matplotlib") or (requires_networkx and not module_available("networkx")):
                raise RuntimeError(f"Plotting function requires matplotlib {'and networkx ' if requires_networkx else ''}"
                                   f"to be installed.")
            return fn(*args, **kw)
        return call
    return factory


def deprecated_method(msg):
    def factory(fn: typing.Callable) -> typing.Callable:
        @functools.wraps(fn)
        def call(*args, **kw):
            warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
            return fn(*args, **kw)
        return call
    return factory


def handle_deprecated_args(argument_name, replaced_by, msg, **kw):
    r""" See :meth:`deprecated_argument` decorator. """
    if kw.get(argument_name, None) is not None and kw.get(replaced_by, None) is not None:
        raise ValueError(f"The argument {argument_name} is deprecated and replaced by {replaced_by}. Please "
                         f"only use {replaced_by}.")
    if kw.get(argument_name, None) is not None and kw.get(replaced_by, None) is None:
        warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
        deprecated_arg = kw.pop(argument_name)
        kw[replaced_by] = deprecated_arg
    return kw.get(replaced_by, None)


def deprecated_argument(argument_name, replaced_by, msg):
    r""" Marks an argument of a function as deprecated. Only works for keyword arguments.

    Parameters
    ----------
    argument_name : str
        The deprecated argument.
    replaced_by : str
        The replacement.
    msg : str
        Warning message.

    Returns
    -------
    factory : callable
        decorator factory parametrized by arguments
    """
    def factory(fn: typing.Callable) -> typing.Callable:
        @functools.wraps(fn)
        def call(*args, **kw):
            handle_deprecated_args(argument_name, replaced_by, msg, **kw)
            return fn(*args, **kw)
        return call
    return factory
