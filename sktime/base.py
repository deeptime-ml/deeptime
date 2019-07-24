import abc
from inspect import signature
from sklearn.base import _pprint as pprint_sklearn


class _base_methods_mixin(object):

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

    def fetch_model(self) -> Model:
        return self._model

    @abc.abstractmethod
    def _create_model(self):
        pass


class Transformer(object):

    @abc.abstractmethod
    def transform(self, data):
        pass
