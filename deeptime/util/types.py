import itertools
import numbers
from typing import Tuple, Optional, Union, List

import numpy as np
from scipy.sparse import spmatrix, issparse

from .data import TimeLaggedDataset, TrajectoryDataset, TrajectoriesDataset


def atleast_nd(ary, ndim, pos=0):
    r"""View input as array with at least `ndim` dimensions.
    New unit dimensions are inserted at the index given by `pos` if
    necessary.

    Parameters
    ----------
    ary : array_like
        The input array. Non-array inputs are converted to arrays.
        Arrays that already have `ndim` or more dimensions are
        preserved.
    ndim : scalar
        The minimum number of dimensions required.
    pos : int, optional
        The index to insert the new dimensions. May range from
        ``-ary.ndim - 1`` to ``+ary.ndim`` (inclusive). Non-negative
        indices indicate locations before the corresponding axis:
        ``pos=0`` means to insert at the very beginning. Negative
        indices indicate locations after the corresponding axis:
        ``pos=-1`` means to insert at the very end. 0 and -1 are always
        guaranteed to work. Any other number will depend on the
        dimensions of the existing array. Default is 0.

    Returns
    -------
    res : ndarray
        An array with ``res.ndim >= ndim``. A view is returned for array
        inputs. Dimensions are prepended if `pos` is 0, so for example,
        a 1-D array of shape ``(N,)`` with ``ndim=4`` becomes a view of
        shape ``(1, 1, 1, N)``. Dimensions are appended if `pos` is -1,
        so for example a 2-D array of shape ``(M, N)`` becomes a view of
        shape ``(M, N, 1, 1)`` when ``ndim=4``.

    See Also
    --------
    np.atleast_1d, np.atleast_2d, np.atleast_3d

    Notes
    -----
    Taken from https://github.com/numpy/numpy/pull/7804.

    This function does not follow the convention of the other atleast_*d
    functions in numpy in that it only accepts a single array argument.
    To process multiple arrays, use a comprehension or loop around the
    function call. See examples below.
    Setting ``pos=0`` is equivalent to how the array would be
    interpreted by numpy's broadcasting rules. There is no need to call
    this function for simple broadcasting. This is also roughly
    (but not exactly) equivalent to
    ``np.array(ary, copy=False, subok=True, ndmin=ndim)``.
    It is easy to create functions for specific dimensions similar to
    the other atleast_*d functions using Python's `functools.partial`
    function. An example is shown below.
    """
    ary = np.array(ary, copy=False, subok=True)
    if ary.ndim:
        pos = np.normalize_axis_index(pos, ary.ndim + 1)
    extra = ndim - ary.ndim
    if extra > 0:
        ind = pos * (slice(None),) + extra * (None,) + (Ellipsis,)
        ary = ary[ind]
    return ary


def ensure_number_array(arr, shape: Tuple = None, ndim: int = None, size=None,
                        accept_sparse=True) -> Union[np.ndarray, spmatrix]:
    return ensure_array(arr, shape=shape, ndim=ndim, size=size, dtype=np.number, accept_sparse=accept_sparse)


def ensure_integer_array(arr, shape: Tuple = None, ndim: int = None, size=None,
                         accept_sparse=True) -> Union[np.ndarray, spmatrix]:
    return ensure_array(arr, shape=shape, ndim=ndim, size=size, dtype=np.integer, accept_sparse=accept_sparse)


def ensure_floating_array(arr, shape: Tuple = None, ndim: int = None, size=None,
                          accept_sparse=True) -> Union[np.ndarray, spmatrix]:
    return ensure_array(arr, shape=shape, ndim=ndim, size=size, dtype=np.floating, accept_sparse=accept_sparse)


def ensure_array(arr, shape: Optional[Tuple] = None, ndim: Optional[int] = None,
                 dtype=None, size=None, accept_sparse=True) -> Union[np.ndarray, spmatrix]:
    if issparse(arr) and not accept_sparse:
        arr = arr.toarray()
    else:
        if isinstance(arr, set):
            if dtype is not None:
                arr = np.fromiter(arr, dtype, len(arr))
            else:
                arr = np.asarray(list(arr))
        if not isinstance(arr, np.ndarray) and not issparse(arr):
            arr = np.asanyarray(arr)
            if ndim is not None and arr.ndim < ndim:
                arr = atleast_nd(arr, ndim=ndim)

    if shape is not None and arr.shape != shape:
        raise ValueError(f"Shape of provided array was {arr.shape} != {shape}")
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"ndim of provided array was {arr.ndim} != {ndim}")
    if size is not None and np.size(arr) != size:
        raise ValueError(f"size of provided array was {np.size(arr)} != {size}")
    if dtype is not None and not np.issubdtype(arr.dtype, dtype):
        raise ValueError(f"Array got incompatible dtype: {arr.dtype} is not a subtype of {dtype}.")
    return arr


def ensure_traj_list(trajs, dtype=None):
    """Makes sure that trajs is a list of trajectories (array of dtype)"""
    if len(trajs) > 0 and not isinstance(trajs, (list, tuple)) and (dtype is None or not isinstance(trajs[0], dtype)):
        return [ensure_floating_array(trajs) if dtype is None else ensure_array(trajs, dtype=dtype)]
    return [ensure_floating_array(t) if dtype is None else ensure_array(t, dtype=dtype) for t in trajs]


def ensure_dtraj_list(dtrajs):
    """Makes sure that dtrajs is a list of discrete trajectories (array of int)"""
    if len(dtrajs) > 0 and isinstance(dtrajs[0], numbers.Real):
        return [ensure_integer_array(dtrajs)]
    return [ensure_integer_array(t) for t in dtrajs]


def ensure_timeseries_data(input_data) -> List[np.ndarray]:
    r""" Ensures that the input data is a time series. This means it must be an iterable of ndarrays or an ndarray
    with a consistent dtype.

    Parameters
    ----------
    input_data : ndarray or list of ndarray
        the input data

    Returns
    -------
    data : list of np.ndarray
        timeseries data
    """
    if isinstance(input_data, np.ndarray) and input_data.ndim >= 3:
        input_data = [x for x in input_data]
    if not isinstance(input_data, (list, tuple)):
        input_data = [input_data]
    grouped = itertools.groupby(input_data, type)
    unique_types = [t for t, _ in grouped]
    if len(unique_types) > 1:
        raise ValueError("All arrays must be of same dtype, but got dtypes {}".format(unique_types))
    assert isinstance(input_data, (list, tuple))
    return input_data


def is_timelagged_dataset(dataset) -> bool:
    r""" Checks whether an object can be ducktyped into a (time-lagged) dataset.

    In particular, this method checks for presence of attributes

    * `__len__`
    * `__getitem__`

    and, if length is positive, checks whether the first element is a tuple of length 2

    Parameters
    ----------
    dataset

    Returns
    -------
    is_dataset : bool
        Whether the input is a time-lagged dataset.
    """
    has_len_and_getitem = hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__')
    if has_len_and_getitem:
        if len(dataset) > 0:
            first_elem = dataset[0]
            return isinstance(first_elem, tuple) and len(first_elem) == 2
        else:
            return False
    return False


def to_dataset(data: Union[TimeLaggedDataset, Tuple[np.ndarray, np.ndarray], np.ndarray],
               lagtime: Optional[int] = None):
    r"""Converts input data to a TimeLaggedDataset if possible, otherwise assumes that data implements `__len__` as well
    as `__getitem__`, where `__getitem__` yields a tuple of data.

    The possible cases are:

    * input data is already a time-lagged dataset, then return immediately (see :meth:`is_timelagged_dataset`).
    * input data is a tuple of (X, Y), where X and Y are ndarrays - in this case they are interpreted as time-lagged
      versions of another, i.e., :math:`Y_i = \mathcal{g}(X_i)`, where :math:`g(\cdot )` describes the temporal
      evolution. In this case lagtime is ignored.
    * input data is a list of trajectories, in this case a concatenated :class:`TrajectoriesDataset` is created
    * input is a ndarray, in this case `Y[i] = X[i+lagtime]` and the result is a dataset of length `len(data) - lagtime`

    Parameters
    ----------
    data : TimeLaggedDataset or tuple of arrays or array
        Input data.
    lagtime : int, optional, default=None
        Lagtime, only is considered if input is array.

    Returns
    -------
    dataset
        A dataset based on input arguments.

    Raises
    ------
    ValueError
        If data is single array but no lagtime is provided or input is list or tuple of length not equal to 2

    Examples
    --------
    Create dataset via trajectory + lagtime

    >>> data = np.arange(0, 6)
    >>> dataset = to_dataset(data, lagtime=1)
    >>> print(dataset[:])
    (array([0, 1, 2, 3, 4]), array([1, 2, 3, 4, 5]))

    Create dataset via corresponding data matrices

    >>> data_instantaneous = np.array([0, 1, 2, 3])
    >>> data_timelagged = np.zeros((4, 2))
    >>> dataset = to_dataset((data_instantaneous, data_timelagged))

    Printing instantaneous data

    >>> print(dataset[:][0])
    [0 1 2 3]

    Printing timelagged data

    >>> print(dataset[:][1])
    [[0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]]
    """
    if isinstance(data, np.ndarray) and data.ndim >= 3:
        data = [x for x in data]
    if isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError(f"If data is provided as tuple the length must be 2 but was {len(data)}.")
        return TimeLaggedDataset(*data)
    if isinstance(data, np.ndarray):
        if lagtime is None:
            raise ValueError("In case data is a single trajectory the lagtime must be given.")
        return TrajectoryDataset(lagtime, data)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], np.ndarray):
        data = ensure_timeseries_data(data)
        return TrajectoriesDataset.from_numpy(lagtime, data)
    assert hasattr(data, '__len__') and len(data) > 0, "Data is empty."
    assert is_timelagged_dataset(data), \
        "Data is not a time-lagged dataset, i.e., yielding tuples of instantaneous and time-lagged data. " \
        "In case of multiple trajectories, deeptime.util.data.TrajectoriesDataset may be used."
    return data
