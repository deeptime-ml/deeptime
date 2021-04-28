import abc
from collections import defaultdict
from inspect import signature
from typing import Optional

from sklearn.base import _pprint as pprint_sklearn


class _base_methods_mixin(object, metaclass=abc.ABCMeta):
    """ Defines common methods used by both Estimator and Model classes. These are mostly static and low-level
    checking of conformity with respect to deeptime conventions.
    """

    def __repr__(self):
        name = '{cls}-{id}:'.format(id=id(self), cls=self.__class__.__name__)
        return '{name}{params}]'.format(
            name=name, params=pprint_sklearn(self.get_params(), offset=len(name), )
        )

    def get_params(self, deep=False):
        r"""Get the parameters.

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
                               % (cls,))
        for arg in args:
            params[arg] = getattr(self, arg, None)
        return params

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def __getstate__(self):
        try:
            state = super().__getstate__()
        except AttributeError:
            state = self.__dict__

        if type(self).__module__.startswith('deeptime.'):
            from deeptime import __version__
            return dict(state.items(), _deeptime_version=__version__)
        else:
            return state

    def __setstate__(self, state):
        from deeptime import __version__
        if type(self).__module__.startswith('deeptime.'):
            pickle_version = state.pop("_deeptime_version", None)
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
    r""" The model superclass. """

    def copy(self) -> "Model":
        r""" Makes a deep copy of this model.

        Returns
        -------
        copy
            A new copy of this model.
        """
        import copy
        return copy.deepcopy(self)

    def _update_params(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                setattr(self, k, v)


class Estimator(_base_methods_mixin):
    r""" Base class of all estimators

    Parameters
    ----------
    model : Model, optional, default=None
        A model which can be used for initialization. In case an estimator is capable of online learning, i.e.,
        capable of updating models, this can be used to resume the estimation process.
    """

    def __init__(self, model=None):
        self._model = model

    @abc.abstractmethod
    def fit(self, data, **kwargs):
        r""" Fits data to the estimator's internal :class:`Model` and overwrites it. This way, every call to
        :meth:`fetch_model` yields an autonomous model instance. Sometimes a :code:`partial_fit` method is available,
        in which case the model can get updated by the estimator.

        Parameters
        ----------
        data : array_like
            Data that is used to fit a model.
        **kwargs
            Additional kwargs.

        Returns
        -------
        self : Estimator
            Reference to self.
        """
        pass

    def fetch_model(self) -> Optional[Model]:
        r""" Yields the estimated model. Can be None if :meth:`fit` was not called.

        Returns
        -------
        model : Model or None
            The estimated model or None.
        """
        return self._model

    @property
    def model(self):
        """ Shortcut to :meth:`fetch_model`. """
        return self.fetch_model()

    @property
    def has_model(self) -> bool:
        r""" Property reporting whether this estimator contains an estimated model. This assumes that the model
        is initialized with `None` otherwise.

        :type: bool
        """
        return self._model is not None


class Transformer:
    r""" Base class of all transformers. """

    @abc.abstractmethod
    def transform(self, data, **kwargs):
        r"""Transforms the input data.

        Parameters
        ----------
        data : array_like
            Input data.

        Returns
        -------
        transformed : array_like
            The transformed data
        """
        pass

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)


class InputFormatError(ValueError):
    """Input data for Estimator is not allowed."""
