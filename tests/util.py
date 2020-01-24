import numpy as np


class GenerateTestMatrix(type):
    def __new__(mcs, name, bases, attr):
        new_test_methods = {}

        test_templates = {k: v for k, v in attr.items() if k.startswith('_test')}
        test_parameters = attr['params']
        for test, params in test_templates.items():
            if test in test_parameters:
                test_param = test_parameters[test]
            else:
                test_param = dict()
            for param_set in test_param:
                # partialmethod(attr[test], **param_set)
                func = lambda *args: attr[test](*args, **param_set)
                # only 'primitive' types should be used as part of test name.
                vals_str = '_'.join((str(v) if not isinstance(v, np.ndarray) else 'array' for v in param_set.values()))
                assert '[' not in vals_str, 'this char makes pytest think it has to ' \
                                            'extract parameters out of the testname.'
                out_name = '{}_{}'.format(test[1:], vals_str)
                func.__qualname__ = 'TestReaders.{}'.format(out_name)
                new_test_methods[out_name] = func

        attr.update(new_test_methods)
        return type.__new__(mcs, name, bases, attr)
