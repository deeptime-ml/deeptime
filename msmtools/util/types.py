
# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# MSMTools is free software: you can redistribute it and/or modify
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

__author__ = 'noe'

import numpy as np
import scipy.sparse as scisp
import numbers
import collections.abc

# ======================================================================================================================
# BASIC TYPE CHECKS
# ======================================================================================================================

def is_int(l):
    r"""Checks if l is an integer

    """
    return isinstance(l, numbers.Integral)

def is_float(l):
    r"""Checks if l is a float

    """
    return isinstance(l, numbers.Real)

def is_iterable_of_int(l):
    r""" Checks if l is iterable and contains only integral types """
    if not is_iterable(l):
        return False

    return all(is_int(value) for value in l)

def is_list_of_int(l):
    r"""Checks if l is a list of integers

    """
    return is_iterable_of_int(l)

def is_tuple_of_int(l):
    r"""Checks if l is a list of integers

    """
    return is_iterable_of_int(l)


def is_iterable_of_float(l):
    r""" Checks if l is iterable and contains only floating point types """
    if not is_iterable(l):
        return False

    return all(is_float(value) for value in l)

def is_list_of_float(l):
    r"""Checks if l is a list of integers

    """
    return is_iterable_of_float(l)

def is_tuple_of_float(l):
    r"""Checks if l is a list of integers

    """
    return is_iterable_of_float(l)

def is_int_vector(l):
    r"""Checks if l is a numpy array of integers

    """
    if isinstance(l, np.ndarray):
        if l.ndim == 1 and (l.dtype.kind == 'i' or l.dtype.kind == 'u'):
            return True
    return False

def is_int_matrix(l):
    r"""Checks if l is a numpy array of floats

    """
    if isinstance(l, np.ndarray):
        if l.ndim == 2 and (l.dtype.kind == 'i' or l.dtype.kind == 'u'):
            return True
    return False

def is_float_vector(l):
    r"""Checks if l is a 1D numpy array of floats

    """
    if isinstance(l, np.ndarray):
        if l.ndim == 1 and (l.dtype.kind == 'f'):
            return True
    return False

def is_float_matrix(l):
    r"""Checks if l is a 2D numpy array of floats

    """
    if isinstance(l, np.ndarray):
        if l.ndim == 2 and (l.dtype.kind == 'f'):
            return True
    return False

def is_float_array(l):
    r"""Checks if l is a numpy array of floats (any dimension

    """
    if isinstance(l, np.ndarray):
        if l.dtype.kind == 'f':
            return True
    return False

def is_string(s):
    return isinstance(s, str)

def is_iterable(I):
    return isinstance(I, collections.abc.Iterable)

def is_list(S):
    # FIXME: name states check for list, but checks for tuple __and__ list. Thats confusing.
    return isinstance(S, (list, tuple))

def is_list_of_string(S):
    return all(is_string(s) for s in S)

def ensure_dtraj(dtraj):
    r"""Makes sure that dtraj is a discrete trajectory (array of int)

    """
    if is_int_vector(dtraj):
        return dtraj
    elif is_list_of_int(dtraj):
        return np.array(dtraj, dtype=int)
    else:
        raise TypeError('Argument dtraj is not a discrete trajectory - '
                        'only list of integers or int-ndarrays are allowed. '
                        'Check type of %s' % repr(dtraj))

def ensure_dtraj_list(dtrajs):
    r"""Makes sure that dtrajs is a list of discrete trajectories (array of int)

    """
    if isinstance(dtrajs, list):
        # elements are ints? then wrap into a list
        if is_list_of_int(dtrajs):
            return [np.array(dtrajs, dtype=int)]
        else:
            for i in range(len(dtrajs)):
                dtrajs[i] = ensure_dtraj(dtrajs[i])
            return dtrajs
    else:
        return [ensure_dtraj(dtrajs)]

def ensure_int_vector(I, require_order = False):
    """Checks if the argument can be converted to an array of ints and does that.

    Parameters
    ----------
    I: int or iterable of int
    require_order : bool
        If False (default), an unordered set is accepted. If True, a set is not accepted.

    Returns
    -------
    arr : ndarray(n)
        numpy array with the integers contained in the argument

    """
    if is_int_vector(I):
        return I
    elif is_int(I):
        return np.array([I])
    elif is_list_of_int(I):
        return np.array(I)
    elif is_tuple_of_int(I):
        return np.array(I)
    elif isinstance(I, set):
        if require_order:
            raise TypeError('Argument is an unordered set, but I require an ordered array of integers')
        else:
            lI = list(I)
            if is_list_of_int(lI):
                return np.array(lI)
    else:
        raise TypeError('Argument is not of a type that is convertible to an array of integers.')

def ensure_int_vector_or_None(F, require_order = False):
    """Ensures that F is either None, or a numpy array of floats

    If F is already either None or a numpy array of floats, F is returned (no copied!)
    Otherwise, checks if the argument can be converted to an array of floats and does that.

    Parameters
    ----------
    F: None, float, or iterable of float

    Returns
    -------
    arr : ndarray(n)
        numpy array with the floats contained in the argument

    """
    if F is None:
        return F
    else:
        return ensure_int_vector(F, require_order = require_order)

def ensure_float_vector(F, require_order = False):
    """Ensures that F is a numpy array of floats

    If F is already a numpy array of floats, F is returned (no copied!)
    Otherwise, checks if the argument can be converted to an array of floats and does that.

    Parameters
    ----------
    F: float, or iterable of float
    require_order : bool
        If False (default), an unordered set is accepted. If True, a set is not accepted.

    Returns
    -------
    arr : ndarray(n)
        numpy array with the floats contained in the argument

    """
    if is_float_vector(F):
        return F
    elif is_float(F):
        return np.array([F])
    elif is_iterable_of_float(F):
        return np.array(F)
    elif isinstance(F, set):
        if require_order:
            raise TypeError('Argument is an unordered set, but I require an ordered array of floats')
        else:
            lF = list(F)
            if is_list_of_float(lF):
                return np.array(lF)
    else:
        raise TypeError('Argument is not of a type that is convertible to an array of floats.')

def ensure_float_vector_or_None(F, require_order = False):
    """Ensures that F is either None, or a numpy array of floats

    If F is already either None or a numpy array of floats, F is returned (no copied!)
    Otherwise, checks if the argument can be converted to an array of floats and does that.

    Parameters
    ----------
    F: float, list of float or 1D-ndarray of float

    Returns
    -------
    arr : ndarray(n)
        numpy array with the floats contained in the argument

    """
    if F is None:
        return F
    else:
        return ensure_float_vector(F, require_order = require_order)

def ensure_dtype_float(x, default=np.float64):
    r"""Makes sure that x is type of float

    """
    if isinstance(x, np.ndarray):
        if x.dtype.kind == 'f':
            return x
        elif x.dtype.kind == 'i':
            return x.astype(default)
        else:
            raise TypeError('x is of type '+str(x.dtype)+' that cannot be converted to float')
    else:
        raise TypeError('x is not an array')

# ======================================================================================================================
# NDARRAY AND SPARSE ARRAY CHECKS
# ======================================================================================================================

def assert_square_matrix(A):
    r""" Asserts if A is a square matrix

    Returns
    -------
    n : int
        the number of rows or columns

    Raises
    ------
    AssertionError
        If assertions has failed

    """
    assert_array(A, ndim=2, uniform=True)

def assert_array(A, shape=None, uniform=None, ndim=None, size=None, dtype=None, kind=None):
    r""" Asserts whether the given array or sparse matrix has the given properties

    Parameters
    ----------
    A : ndarray, scipy.sparse matrix or array-like
        the array under investigation

    shape : shape, optional, default=None
        asserts if the array has the requested shape. Be careful with vectors
        because this will distinguish between row vectors (1,n), column vectors
        (n,1) and arrays (n,). If you want to be less specific, consider using
        size

    square : None | True | False
        if not None, asserts whether the array dimensions are uniform (e.g.
        square for a ndim=2 array) (True), or not uniform (False).

    size : int, optional, default=None
        asserts if the arrays has the requested number of elements

    ndim : int, optional, default=None
        asserts if the array has the requested dimension

    dtype : type, optional, default=None
        asserts if the array data has the requested data type. This check is
        strong, e.g. int and int64 are not equal. If you want a weaker check,
        consider the kind option

    kind : string, optional, default=None
        Checks if the array data is of the specified kind. Options include 'i'
        for integer types, 'f' for float types Check numpy.dtype.kind for
        possible options. An additional option is 'numeric' for either integer
        or float.

    Raises
    ------
    AssertionError
        If assertions has failed

    """
    try:
        if shape is not None:
            if not np.array_equal(np.shape(A), shape):
                raise AssertionError('Expected shape '+str(shape)+' but given array has shape '+str(np.shape(A)))
        if uniform is not None:
            shapearr = np.array(np.shape(A))
            is_uniform = np.count_nonzero(shapearr-shapearr[0]) == 0
            if uniform and not is_uniform:
                raise AssertionError('Given array is not uniform \n'+str(shapearr))
            elif not uniform and is_uniform:
                raise AssertionError('Given array is not nonuniform: \n'+str(shapearr))
        if size is not None:
            if not np.size(A) == size:
                raise AssertionError('Expected size '+str(size)+' but given array has size '+str(np.size(A)))
        if ndim is not None:
            if not ndim == np.ndim(A):
                raise AssertionError('Expected shape '+str(ndim)+' but given array has shape '+str(np.ndim(A)))
        if dtype is not None:
            # now we must create an array if we don't have one yet
            if not isinstance(A, (np.ndarray)) and not scisp.issparse(A):
                A = np.array(A)
            if not np.dtype(dtype) == A.dtype:
                raise AssertionError('Expected data type '+str(dtype)+' but given array has data type '+str(A.dtype))
        if kind is not None:
            # now we must create an array if we don't have one yet
            if not isinstance(A, (np.ndarray)) and not scisp.issparse(A):
                A = np.array(A)
            if kind == 'numeric':
                if not (A.dtype.kind == 'i' or A.dtype.kind == 'f'):
                    raise AssertionError('Expected numerical data, but given array has data kind '+str(A.dtype.kind))
            elif not A.dtype.kind == kind:
                raise AssertionError('Expected data kind '+str(kind)
                                     +' but given array has data kind '+str(A.dtype.kind))
    except Exception as ex:
        if isinstance(ex, AssertionError):
            raise ex
        else:  # other exception raised in the test code above
            print('Found exception: ',ex)
            raise AssertionError('Given argument is not an array of the expected shape or type:\n'+
                                 'arg = '+str(A)+'\ntype = '+str(type(A)))

def ensure_ndarray(A, shape=None, uniform=None, ndim=None, size=None, dtype=None, kind=None):
    r""" Ensures A is an ndarray and does an assert_array with the given parameters

    Returns
    -------
    A : ndarray
        If A is already an ndarray, it is just returned. Otherwise this is an independent copy as an ndarray

    """
    if not isinstance(A, np.ndarray):
        try:
            A = np.array(A)
        except:
            raise AssertionError('Given argument cannot be converted to an ndarray:\n'+str(A))
    assert_array(A, shape=shape, uniform=uniform, ndim=ndim, size=size, dtype=dtype, kind=kind)
    return A

def ensure_ndarray_or_sparse(A, shape=None, uniform=None, ndim=None, size=None, dtype=None, kind=None):
    r""" Ensures A is an ndarray or a scipy sparse matrix and does an assert_array with the given parameters

    Returns
    -------
    A : ndarray
        If A is already an ndarray, it is just returned. Otherwise this is an independent copy as an ndarray

    """
    if not isinstance(A, np.ndarray) and not scisp.issparse(A):
        try:
            A = np.array(A)
        except:
            raise AssertionError('Given argument cannot be converted to an ndarray:\n'+str(A))
    assert_array(A, shape=shape, uniform=uniform, ndim=ndim, size=size, dtype=dtype, kind=kind)
    return A

def ensure_ndarray_or_None(A, shape=None, uniform=None, ndim=None, size=None, dtype=None, kind=None):
    r""" Ensures A is None or an ndarray and does an assert_array with the given parameters """
    if A is not None:
        return ensure_ndarray(A, shape=shape, uniform=uniform, ndim=ndim, size=size, dtype=dtype, kind=kind)
    else:
        return None


# ======================================================================================================================
# EMMA TRAJECTORY TYPES
# ======================================================================================================================

def ensure_traj(traj):
    r"""Makes sure that dtraj is a discrete trajectory (array of float)

    """
    if is_float_matrix(traj):
        return traj
    elif is_float_vector(traj):
        return traj[:,None]
    else:
        try:
            arr = np.array(traj)
            arr = ensure_dtype_float(arr)
            if is_float_matrix(arr):
                return arr
            if is_float_vector(arr):
                return arr[:,None]
            else:
                raise TypeError('Argument traj cannot be cast into a two-dimensional array. Check type.')
        except:
            raise TypeError('Argument traj is not a trajectory - only float-arrays or list of float-arrays are allowed. Check type.')

def ensure_traj_list(trajs):
    if isinstance(trajs, list):
        # elements are ints? make it a matrix and wrap into a list
        if is_list_of_float(trajs):
            return [np.array(trajs)[:,None]]
        else:
            res = []
            for i in range(len(trajs)):
                res.append(ensure_traj(trajs[i]))
            return res
    else:
        # looks like this is one trajectory
        return [ensure_traj(trajs)]
