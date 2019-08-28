import abc
from inspect import signature

from sklearn.base import _pprint as pprint_sklearn


class _base_methods_mixin(object, metaclass=abc.ABCMeta):

    def __repr__(self):
        name = '{cls}-{id}:'.format(id=id(self), cls=self.__class__.__name__)
        return '{name}{params}]'.format(name=name,
            params=pprint_sklearn(self.get_params(), offset=len(name), )
        )

    def get_params(self, deep=True):
        """Get parameters of this kernel.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
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


class Estimator(_base_methods_mixin):

    def __init__(self, model=None):
        self._model = model if model is not None else self._create_model()

    @abc.abstractmethod
    def fit(self, data):
        pass

    def fetch_model(self, copy=False) -> Model:
        return self._model if not copy else self._model.copy()

    @abc.abstractmethod
    def _create_model(self):
        pass


class Transformer(object):

    @abc.abstractmethod
    def transform(self, data):
        pass
