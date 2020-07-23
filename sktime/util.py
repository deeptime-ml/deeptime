# This file is part of scikit-time
#
# Copyright (c) 2020 AI4Science Group, Freie Universitaet Berlin (GER)
#
# scikit-time is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import collections
import functools
import itertools
import numbers
import os
from typing import Optional, List
from weakref import WeakKeyDictionary

import numpy as np


def handle_n_jobs(value: Optional[int]) -> int:
    r"""Handles the n_jobs parameter consistently so that a non-negative number is returned.
    In particular, if

      * value is None, use 1 job
      * value is negative, use number cores available * 2
      * value is positive, use value

    Parameters
    ----------
    value : int or None
        The provided n_jobs argument

    Returns
    -------
    n_jobs : int
        A non-negative integer value describing how many threads can be started simultaneously.
    """
    if value is None:
        return 1
    elif value < 0:
        count = os.cpu_count()
        if count is None:
            raise ValueError("Could not determine number of cpus in system, please provide n_jobs manually.")
        return os.cpu_count() * 2
    else:
        return value


def ensure_ndarray(arr, shape: tuple = None, ndim: int = None, dtype=None, size=None,
                   allow_none=False) -> [np.ndarray, None]:
    if arr is None:
        if allow_none:
            return None
        else:
            raise ValueError("None not allowed!")
    assert arr is not None
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=dtype)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"Shape of provided array was {arr.shape} != {shape}")
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"ndim of provided array was {arr.ndim} != {ndim}")
    if size is not None and np.size(arr) != size:
        raise ValueError(f"size of provided array was {np.size(arr)} != {size}")
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype)
    return arr


def submatrix(M, sel):
    """Returns a submatrix of the quadratic matrix M, given by the selected columns and row
    Parameters
    ----------
    M : ndarray(n,n)
        symmetric matrix
    sel : int-array
        selection of rows and columns. Element i,j will be selected if both are in sel.
    Returns
    -------
    S : ndarray(m,m)
        submatrix with m=len(sel)
    """
    assert len(M.shape) == 2, 'M is not a matrix'
    assert M.shape[0] == M.shape[1], 'M is not quadratic'
    import scipy.sparse
    """Row slicing"""
    if scipy.sparse.issparse(M):
        C_cc = M.tocsr()
    else:
        C_cc = M
    C_cc = C_cc[sel, :]

    """Column slicing"""
    if scipy.sparse.issparse(M):
        C_cc = C_cc.tocsc()
    C_cc = C_cc[:, sel]

    if scipy.sparse.issparse(M):
        return C_cc.tocoo()

    return C_cc


def ensure_traj_list(trajs, dtype):
    """Makes sure that trajs is a list of trajectories (array of dtype)"""
    if len(trajs) > 0 and not isinstance(trajs, (list, tuple)) and not isinstance(trajs[0], dtype):
        return [ensure_ndarray(trajs, dtype=dtype)]
    return [ensure_ndarray(t, dtype=dtype) for t in trajs]


def ensure_dtraj_list(dtrajs):
    """Makes sure that dtrajs is a list of discrete trajectories (array of int)"""
    if len(dtrajs) > 0 and isinstance(dtrajs[0], numbers.Real):
        return [ensure_ndarray(dtrajs, dtype=np.int32)]
    return [ensure_ndarray(t, dtype=np.int32) for t in dtrajs]


def is_iterable(I):
    return isinstance(I, collections.Iterable)


def is_iterable_of_types(l, supertype):
    r""" Checks whether all elements of l are of type `supertype`. """
    return is_iterable(l) and all(issubclass(t, supertype) for t, _ in itertools.groupby(l, type))


def ensure_timeseries_data(input_data):
    r""" Ensures that the input data is a time series. This means it must be an iterable of ndarrays or an ndarray
    with dtype :attr:`np.float32` or :attr:`np.float64`.

    Parameters
    ----------
    input_data : ndarray or list of ndarray, dtype float32 or float64
        the input data

    Returns
    -------
    data : list of ndarray
        timeseries data
    """
    if not isinstance(input_data, list):
        if not isinstance(input_data, np.ndarray):
            raise ValueError('input data can not be converted to a list of arrays')
        elif isinstance(input_data, np.ndarray):
            if input_data.dtype not in (np.float32, np.float64):
                raise ValueError('only float and double dtype is supported')
            return [input_data]
    else:
        for i, x in enumerate(input_data):
            if not isinstance(x, np.ndarray):
                raise ValueError(f'element {i} of given input data list is not an array.')
            else:
                if x.dtype not in (np.float32, np.float64):
                    raise ValueError('only float and double dtype is supported')
                input_data[i] = x
    grouped = itertools.groupby(input_data, type)
    unique_types = [t for t, _ in grouped]
    if len(unique_types) > 1:
        raise ValueError("All arrays must be of same dtype, but got dtypes {}".format(unique_types))
    return input_data


def confidence_interval(data, conf=0.95):
    r""" Computes element-wise confidence intervals from a sample of ndarrays

    Given a sample of arbitrarily shaped ndarrays, computes element-wise
    confidence intervals

    Parameters
    ----------
    data : array-like of dimension 1 and 2
        array of numbers or arrays. The first index is used as the sample
        index, the remaining indexes are specific to the array of interest
    conf : float, optional, default = 0.95
        confidence interval

    Return
    ------
    lower : ndarray(shape)
        element-wise lower bounds
    upper : ndarray(shape)
        element-wise upper bounds

    """
    if conf < 0 or conf > 1:
        raise ValueError(f'Not a meaningful confidence level: {conf}')

    data = ensure_ndarray(data)

    def _confidence_interval_1d(x):
        """
        Computes the mean and alpha-confidence interval of the given sample set

        Parameters
        ----------
        x : ndarray
            a 1D-array of samples

        Returns
        -------
        (m, l, r) : m is the mean of the data, and (l, r) are the m-alpha/2
            and m+alpha/2 confidence interval boundaries.
        """
        import math, warnings
        assert x.ndim == 1, x.ndim

        if np.any(np.isnan(x)):
            return np.nan, np.nan, np.nan

        dmin, dmax = np.min(x), np.max(x)

        if np.isclose(dmin, dmax):
            warnings.warn('confidence interval for constant data is not meaningful', stacklevel=3)
            return dmin, dmin, dmax

        m = np.mean(x)
        x = np.sort(x)

        # index of the mean
        im = np.searchsorted(x, m)
        if im == 0 or im == len(x) or (np.isinf(m - x[im - 1]) and np.isinf(x[im] - x[im - 1])):
            pm = im
        else:
            pm = (im - 1) + (m - x[im - 1]) / (x[im] - x[im - 1])
        # left interval boundary
        pl = pm - conf * pm
        il1 = max(0, int(math.floor(pl)))
        il2 = min(len(x) - 1, int(math.ceil(pl)))
        if np.isclose(x[il1], x[il2]):  # catch infs
            l = x[il1]
        else:
            l = x[il1] + (pl - il1) * (x[il2] - x[il1])
        # right interval boundary
        pr = pm + conf * (len(x) - im)
        ir1 = max(0, int(math.floor(pr)))
        ir2 = min(len(x) - 1, int(math.ceil(pr)))
        if np.isclose(x[ir1], x[ir2]):  # catch infs
            r = x[ir1]
        else:
            r = x[ir1] + (pr - ir1) * (x[ir2] - x[ir1])

        # return
        return m, l, r

    if conf == 1.:
        return np.min(data, axis=0), np.max(data, axis=0)

    if data.ndim == 1:
        mean, lower, upper = _confidence_interval_1d(data)
        return lower, upper
    else:
        lower = np.zeros_like(data[0])
        upper = np.zeros_like(data[0])

        # compute interval for each column

        def _column(arr, indexes):
            """ Returns a column with given indexes from a deep array

            For example, if the array is a matrix and indexes is a single int, will
            return arr[:,indexes]. If the array is an order 3 tensor and indexes is a
            pair of ints, will return arr[:,indexes[0],indexes[1]], etc.

            """
            if arr.ndim == 2 and isinstance(indexes, (int, tuple)):
                if isinstance(indexes, tuple):
                    indexes = indexes[0]
                return arr[:, indexes]
            elif arr.ndim == 3 and len(indexes) == 2:
                return arr[:, indexes[0], indexes[1]]
            else:
                raise NotImplementedError('Only supporting arrays of dimension 2 and 3 as yet.')

        for i in np.ndindex(data[0].shape):
            col = _column(data, i)
            mean, lower[i], upper[i] = _confidence_interval_1d(col)

        return lower, upper


def call_member(obj, f, *args, **kwargs):
    """ Calls the specified method, property or attribute of the given object

    Parameters
    ----------
    obj : object
        The object that will be used
    f : str or function
        Name of or reference to method, property or attribute
    *args : list
        list of arguments to pass to f during evaluation
    ** kwargs: dict
        keyword arguments to pass to f during evaluation
    """
    import inspect
    # get function name
    if not isinstance(f, str):
        fname = f.__func__.__name__
    else:
        fname = f
    # get the method ref
    method = getattr(obj, fname)
    # handle cases
    if inspect.ismethod(method):
        return method(*args, **kwargs)

    # attribute or property
    return method


class QuantityStatistics(object):
    """ Container for statistical quantities computed on samples.

    Attributes
    ----------
    mean: array(n)
        mean along axis=0
    std: array(n)
        std deviation along axis=0
    L : ndarray(shape)
        element-wise lower bounds
    R : ndarray(shape)
        element-wise upper bounds
    """

    @staticmethod
    def gather(samples, quantity=None, store_samples=False, delimiter='/', confidence: float = 0.95, *args, **kwargs):
        r"""Obtain statistics about a sampled quantity. Can also be a chained call, separated by the delimiter.

        Parameters
        ----------
        samples : list of object
            The samples which contain sought after quantities.
        quantity : str, optional, default=None
            Name of attribute, which will be evaluated on samples. If None, no quantity is evaluated and the samples
            are assumed to already be the quantity that is to be evaluated.
        store_samples : bool, optional, default=False
            Whether to store the samples (array).
        delimiter : str, optional, default='/'
            Separator to call members of members.
        confidence : float, optional, default=0.95
            Confidence parameter for the confidence intervals.
        *args
            pass through
        **kwargs
            pass through

        Returns
        -------
        statistics : QuantityStatistics
            The collected statistics.
        """
        if quantity is not None and delimiter in quantity:
            qs = quantity.split(delimiter)
            quantity = qs[-1]
            for q in qs[:-1]:
                samples = [call_member(s, q) for s in samples]
        if quantity is not None:
            samples = [call_member(s, quantity, *args, **kwargs) for s in samples]
        return QuantityStatistics(samples, quantity=quantity, store_samples=store_samples, confidence=confidence)

    def __init__(self, samples: List[np.ndarray], quantity, confidence=0.95, store_samples=False):
        r""" Creates a new container instance.

        Parameters
        ----------
        samples: list of ndarrays
            the samples
        store_samples: bool, default=False
            whether to store the samples (array).
        """
        super().__init__()
        self.quantity = quantity
        # TODO: shall we refer to the original object?
        # we re-add the (optional) quantity, because the creation of a new array will strip it.
        unit = getattr(samples[0], 'u', None)
        if unit is not None:
            samples = np.array(tuple(x.magnitude for x in samples))
        else:
            samples = np.array(samples)
        if unit is not None:
            samples *= unit
        if store_samples:
            self.samples = samples
        else:
            self.samples = np.empty(0) * unit
        self.mean = samples.mean(axis=0)
        self.std = samples.std(axis=0)
        self.L, self.R = confidence_interval(samples, conf=confidence)
        if unit is not None:
            self.L *= unit
            self.R *= unit

    def __str__(self):
        return "QuantityStatistics(mean={}, std={})".format(self.mean, self.std)


class cached_property(property):
    r"""
    Property that gets cached, obeys property api and can also be invalidated and overridden. Inspired from
    https://github.com/pydanny/cached-property/ and  https://stackoverflow.com/a/17330273.
    """
    _default_cache_entry = object()

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        super(cached_property, self).__init__(fget, fset, fdel, doc)
        self.cache = WeakKeyDictionary()

    def __get__(self, instance, owner):
        if instance is None:
            return self
        value = self.cache.get(instance, self._default_cache_entry)
        if value is self._default_cache_entry:
            value = self.fget(instance)
            self.cache[instance] = value
        return value

    def __set__(self, instance, value):
        self.cache[instance] = value

    def __delete__(self, instance):
        del self.cache[instance]

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)

    def invalidate(self):
        self.cache.clear()


def plotting_function(fn):  # pragma: no cover
    r""" Decorator marking a function that is a plotting utility. This will exclude it from coverage and test
    whether dependencies are installed. """

    @functools.wraps(fn)
    def wrapper(*args, **kw):
        try:
            import matplotlib
            import networkx
        except (ModuleNotFoundError, ImportError):
            raise RuntimeError("Plotting functions require matplotlib and networkx to be installed.")
        return fn(*args, **kw)

    return wrapper
