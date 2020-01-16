import abc
from inspect import signature

from sklearn.base import _pprint as pprint_sklearn


class _base_methods_mixin(object, metaclass=abc.ABCMeta):
    """ Defines common methods used by both Estimator and Model classes. These are mostly static and low-level
    checking of conformity with respect to scikit-time conventions.
    """

    def __repr__(self):
        name = '{cls}-{id}:'.format(id=id(self), cls=self.__class__.__name__)
        return '{name}{params}]'.format(name=name,
                                        params=pprint_sklearn(self.get_params(), offset=len(name), )
        )

    def get_params(self):
        r"""Get parameters of this kernel.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        params = dict()

        # introspect the constructor arguments to find the model parameters
        # to represent
        cls = self.__class__
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        init_sign = signature(init)
        args, varargs = [], []
        for parameter in init_sign.parameters.values():
            if (parameter.kind != parameter.VAR_KEYWORD and
                    parameter.name != 'self'):
                args.append(parameter.name)
            if parameter.kind == parameter.VAR_POSITIONAL:
                varargs.append(parameter.name)

        if len(varargs) != 0:
            raise RuntimeError("scikit-learn kernels should always "
                               "specify their parameters in the signature"
                               " of their __init__ (no varargs)."
                               " %s doesn't follow this convention."
                               % (cls, ))
        for arg in args:
            params[arg] = getattr(self, arg, None)
        return params

    def __getstate__(self):
        try:
            state = super().__getstate__()
        except AttributeError:
            state = self.__dict__

        if type(self).__module__.startswith('sktime.'):
            from sktime import __version__
            return dict(state.items(), _sktime_version=__version__)
        else:
            return state

    def __setstate__(self, state):
        from sktime import __version__
        if type(self).__module__.startswith('sktime.'):
            pickle_version = state.pop("_sktime_version", None)
            if pickle_version != __version__:
                import warnings
                warnings.warn(
                    "Trying to unpickle estimator {0} from version {1} when "
                    "using version {2}. This might lead to breaking code or "
                    "invalid results. Use at your own risk.".format(
                        self.__class__.__name__, pickle_version, __version__),
                    UserWarning)
        try:
            super().__setstate__(state)
        except AttributeError:
            self.__dict__.update(state)


class Model(_base_methods_mixin):

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def _update_params(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                setattr(self, k, v)


class Estimator(_base_methods_mixin):
    """ Base class of all estimators """

    """ class wide flag to control whether input of fit or partial_fit should be checked for modifications """
    _MUTABLE_INPUT_DATA = False

    def __init__(self, model=None):
        self._model = model

    @abc.abstractmethod
    def fit(self, data, **kwargs):
        """ performs a fit of this estimator with data. Creates a new model instance by default.
        :param data:
        :return: self
        """
        pass

    def fetch_model(self) -> Model:
        return self._model

    def __getattribute__(self, item):
        if (item == 'fit' or item == 'partial_fit') and not self._MUTABLE_INPUT_DATA:
            fit = super(Estimator, self).__getattribute__(item)
            return _ImmutableInputData(fit)

        return super(_base_methods_mixin, self).__getattribute__(item)


class Transformer(object):

    @abc.abstractmethod
    def transform(self, data):
        pass


class _ImmutableInputData(object):
    """A function decorator for Estimator.fit() to make input data immutable """
    def __init__(self, fit_method):
        self.fit_method = fit_method
        self._data = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value_):
        import numpy as np
        args, kwargs = value_
        # store data as a list of ndarrays
        # handle optional y for supervised learning
        y = kwargs.get('y', None)

        self._data = [] if y is None else [y]

        # first argument is x
        if len(args) == 0:
            if 'data' in kwargs:
                args = [kwargs['data']]
            elif len(kwargs) == 1:
                args = [kwargs[k] for k in kwargs.keys()]
            else:
                raise InputFormatError(f'No input at all for fit(). Input was {args}, kw={kwargs}')
        value = args[0]
        if isinstance(value, np.ndarray):
            self._data.append(value)
        elif isinstance(value, (list, tuple)):
            for i, x in enumerate(value):
                if isinstance(x, np.ndarray):
                    self._data.append(x)
                else:
                    raise InputFormatError(f'Invalid input element in position {i}, only numpy.ndarrays allowed.')
        elif isinstance(value, Model):
            self._data.append(value)
        else:
            raise InputFormatError(f'Only model, ndarray or list/tuple of ndarray allowed. '
                                   f'But was of type {type(value)}: {value}.')

    def __enter__(self):
        import numpy as np
        self.old_writable_flags = []
        for d in self.data:
            if isinstance(d, np.ndarray):
                self.old_writable_flags.append(d.flags.writeable)
                # set ndarray writabe flags to false
                d.flags.writeable = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        # restore ndarray writable flags to old state
        import numpy as np
        for d, writable in zip(self.data, self.old_writable_flags):
            if isinstance(d, np.ndarray):
                d.flags.writeable = writable

    def __call__(self, *args, **kwargs):
        # extract input data from args, **kwargs (namely x and y)
        self.data = args, kwargs

        # here we invoke the immutable setting context manager.
        with self:
            return self.fit_method(*args, **kwargs)


class InputFormatError(ValueError):
    """Input data for Estimator is not allowed."""
