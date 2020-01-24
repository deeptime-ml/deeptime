import numpy as np

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
        new_test_methods = {}

        test_templates = {k: v for k, v in attr.items() if k.startswith('_test')}
        test_parameters = attr['params']
        for test, params in test_templates.items():
            if test in test_parameters:
                test_param = test_parameters[test]
            else:
                test_param = dict()

            for ix, param_set in enumerate(test_param):
                func = lambda *args: attr[test](*args, **param_set)
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
