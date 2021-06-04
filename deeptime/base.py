import abc
from collections import defaultdict
from inspect import signature
from typing import Optional

from sklearn.base import _pprint as pprint_sklearn


class _BaseMethodsMixin(abc.ABC):
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


class Dataset(abc.ABC):
    r""" The Dataset superclass. It is an abstract class requiring implementations of

    * :meth:`__len__` to obtain its length,
    * :meth:`__getitem__` to obtain one or several items,
    * :meth:`setflags` to set its writeable state.

    See Also
    --------
    deeptime.util.data.TimeLaggedDataset
        A dataset implementation for pairs of instantaneous and timelagged data.
    deeptime.util.data.TimeLaggedConcatDataset
        A concatenation of several :class:`TimeLaggedDataset <deeptime.util.data.TimeLaggedDataset>` .
    deeptime.util.data.TrajectoryDataset
        A dataset for one trajectory with a lagtime.
    deeptime.util.data.TrajectoriesDataset
        A dataset for multiple trajectories with a lagtime.
    """

    @abc.abstractmethod
    def setflags(self, write=True):
        r""" Set writeable flags for contained arrays. """

    @abc.abstractmethod
    def __len__(self):
        r""" Length of this dataset. """

    @abc.abstractmethod
    def __getitem__(self, item):
        r""" Retrieves one or multiple items from this dataset. """


class Model(_BaseMethodsMixin):
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


class Estimator(_BaseMethodsMixin):
    r""" Base class of all estimators

    Parameters
    ----------
    model : Model, optional, default=None
        A model which can be used for initialization. In case an estimator is capable of online learning, i.e.,
        capable of updating models, this can be used to resume the estimation process.
    """

    """ class wide flag to control whether input of fit or partial_fit should be checked for modifications """
    _MUTABLE_INPUT_DATA = False

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

    def fetch_model(self) -> Optional[Model]:
        r""" Yields the estimated model. Can be None if :meth:`fit` was not called.

        Returns
        -------
        model : Model or None
            The estimated model or None.
        """
        return self._model

    def fit_fetch(self, data, **kwargs):
        r""" Fits the internal model on data and subsequently fetches it in one call.

        Parameters
        ----------
        data : array_like
            Data that is used to fit the model.
        **kwargs
            Additional arguments to :meth:`fit`.

        Returns
        -------
        model
            The estimated model.
        """
        self.fit(data, **kwargs)
        return self.fetch_model()

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

    def __getattribute__(self, item):
        if (item == 'fit' or item == 'partial_fit') and not self._MUTABLE_INPUT_DATA:
            fit = super(Estimator, self).__getattribute__(item)
            return _ImmutableInputData(fit)

        return super(_BaseMethodsMixin, self).__getattribute__(item)


class _ImmutableInputData:
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
        self._data = []

        # first argument is x
        if len(args) == 0:
            if 'data' in kwargs:
                args = [kwargs['data']]
            elif len(kwargs) == 1:
                args = [kwargs[k] for k in kwargs.keys()]
            else:
                raise InputFormatError(f'No input at all for fit(). Input was {args}, kw={kwargs}')
        value = args[0]
        if hasattr(value, 'setflags'):
            self._data.append(value)
        elif isinstance(value, (list, tuple)):
            for i, x in enumerate(value):
                if isinstance(x, np.ndarray):
                    self._data.append(x)
                else:
                    raise InputFormatError(f'Invalid input element in position {i}, only numpy.ndarrays allowed.')
        elif isinstance(value, (Model, Estimator)):
            self._data.append(value)
        else:
            raise InputFormatError(f'Only estimator, model, ndarray or list/tuple of ndarray allowed. '
                                   f'But was of type {type(value)}: {value}.')

    def __enter__(self):
        import numpy as np
        self.old_writable_flags = []
        for d in self.data:
            if isinstance(d, np.ndarray):
                self.old_writable_flags.append(d.flags.writeable)
                # set ndarray writabe flags to false
                try:
                    d.setflags(write=False)
                except:
                    # although this should not raise, occasionally it does raise
                    # due to arrays stemming from torch which then cannot be set immutable
                    ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        # restore ndarray writable flags to old state
        import numpy as np
        for d, writable in zip(self.data, self.old_writable_flags):
            if isinstance(d, np.ndarray):
                try:
                    d.setflags(write=writable)
                except:
                    # although this should not raise, occasionally it does raise
                    # due to arrays stemming from torch which then cannot be set immutable
                    ...

    def __call__(self, *args, **kwargs):
        # extract input data from args, **kwargs (namely x and y)
        self.data = args, kwargs

        # here we invoke the immutable setting context manager.
        with self:
            return self.fit_method(*args, **kwargs)


class Transformer(abc.ABC):
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

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)


class EstimatorTransformer(Estimator, Transformer, abc.ABC):

    def fit_transform(self, data, fit_options=None, transform_options=None):
        r""" Fits a model which simultaneously functions as transformer and subsequently transforms
        the input data. The estimated model can be accessed by calling :meth:`fetch_model`.

        Parameters
        ----------
        data : array_like
            The input data.
        fit_options : dict, optional, default=None
            Optional keyword arguments passed on to the fit method.
        transform_options : dict, optional, default=None
            Optional keyword arguments passed on to the transform method.

        Returns
        -------
        output : array_like
            Transformed data.
        """
        fit_options = {} if fit_options is None else fit_options
        transform_options = {} if transform_options is None else transform_options
        return self.fit(data, **fit_options).transform(data, **transform_options)

    def transform(self, data, **kwargs):
        r""" Transforms data with the encapsulated model.

        Parameters
        ----------
        data : array_like
            Input data
        **kwargs
            Optional arguments.

        Returns
        -------
        output : array_like
            Transformed data.
        """
        model = self.fetch_model()
        if model is None:
            raise ValueError("This estimator contains no model yet, fit should be called first.")
        return model.transform(data, **kwargs)


class InputFormatError(ValueError):
    """Input data for Estimator is not allowed."""
