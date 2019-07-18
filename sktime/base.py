import abc
from inspect import signature


class Model(object):

    def copy(self):
        import copy
        return copy.deepcopy(self)

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

    def __repr__(self):
        return '{cls}-{id}: {params}]'.format(
            id=id(self),
            cls=self.__class__,
            params=self.get_params(),
        )

    def __str__(self):
        from io import StringIO
        import pprint
        buff = StringIO()
        start = '{cls_name}-{id}: '.format(
            cls_name=self.__class__.__name__,
            id=id(self),
        )
        buff.write(start)
        pprint.pprint(self.get_params(), stream=buff, width=120+buff.tell()+5, compact=True)
        buff.seek(0)
        return buff.read()


class Estimator(object):

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
