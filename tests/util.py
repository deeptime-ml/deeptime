import sys
import time

import numpy as np


class timing(object):
    """A timing context manager

    Examples
    --------
    >>> long_function = lambda : None
    >>> with timing('long_function'):  # doctest: +SKIP
    ...     long_function()  # doctest: +SKIP
    long_function: 0.000 seconds
    """

    def __init__(self, name='block'):
        self.name = name
        self.time = 0
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, ty, val, tb):
        self.end = time.time()
        self.time = self.end - self.start
        print("%s: %0.3f seconds" % (self.name, self.time))
        return False


class GenerateTestMatrix(type):
    """
    Metaclass definition for parameterized testing. Usage as follows:

    >>> import unittest
    >>> class TestSomething(unittest.TestCase, metaclass=GenerateTestMatrix):
    ...     # set up parameters
    ...     params = {
    ...         '_test_something': [dict(my_arg=val) for val in [5, 10, 15, 20]]  # generates 4 tests, parameters as kw
    ...     }
    ...     # this test is instantiated four times with the four different arguments
    ...     def _test_something(self, my_arg):
    ...         assert my_arg % 5 == 0
    >>> if __name__ == '__main__':
    ...     unittest.main()
    """

    def __new__(mcs, name, bases, attr):
        from functools import partial

        # needed for python2
        class partialmethod(partial):
            def __get__(self, instance, owner):
                if instance is None:
                    return self
                return partial(self.func, instance,
                               *(self.args or ()), **(self.keywords or {}))

        new_test_methods = {}

        test_templates = {k: v for k, v in attr.items() if k.startswith('_test')}
        if 'params' in attr:
            test_parameters = attr['params']
            for test, params in test_templates.items():
                if test in test_parameters:
                    test_param = test_parameters[test]
                else:
                    test_param = dict()

                for ix, param_set in enumerate(test_param):
                    func = partialmethod(attr[test], **param_set)
                    # only 'primitive' types should be used as part of test name.
                    vals_str = ''
                    for v in param_set.values():
                        if len(vals_str) > 0:
                            vals_str += '_'
                        if not isinstance(v, np.ndarray):
                            vals_str += str(v)
                        else:
                            vals_str += 'array{}'.format(ix)
                    assert '[' not in vals_str, 'this char makes pytest think it has to ' \
                                                'extract parameters out of the testname. (in {})'.format(vals_str)
                    out_name = '{}_{}'.format(test[1:], vals_str)
                    func.__qualname__ = 'TestReaders.{}'.format(out_name)
                    new_test_methods[out_name] = func

        attr.update(new_test_methods)
        return type.__new__(mcs, name, bases, attr)


def assert_array_not_equal(arr1, arr2, err_msg='', verbose=True):
    with np.testing.assert_raises(AssertionError, msg=err_msg):
        np.testing.assert_array_equal(arr1, arr2, verbose=verbose)


if sys.version_info[0] == 3 and sys.version_info[1] > 6:
    from contextlib import nullcontext
else:
    class nullcontext:

        def __enter__(self, *args, **kw):
            pass

        def __exit__(self, *args, **kw):
            pass
