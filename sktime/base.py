import abc
from inspect import signature

from sklearn.base import _pprint as pprint_sklearn


class _base_methods_mixin(object, metaclass=abc.ABCMeta):
    """ defines common methods used by both Estimator and Model classes.
    """

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

    def _update_params(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                setattr(self, k, v)


class Estimator(_base_methods_mixin):

    def __init__(self, model=None):
        # we only need to create a default model in case the subclassing Estimator provides the partial_fit interface.
        if hasattr(self.__class__, 'partial_fit') and model is None:
            self._model = self._create_model()
        # TODO: not tested (e.g. by partially fitted models.
        elif model is not None:
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

    @abc.abstractmethod
    def _create_model(self):
        pass

    def __getattribute__(self, item):
        if item == 'fit':
            self._model = self._create_model()
        return super(_base_methods_mixin, self).__getattribute__(item)


class Transformer(object):

    @abc.abstractmethod
    def transform(self, data):
        pass
