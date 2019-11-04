
# This file is part of PyEMMA
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

__author__ = 'noe'

import numpy as np
import numbers
import collections


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
    return isinstance(s, basestring)


def is_iterable(I):
    return isinstance(I, collections.Iterable)


def is_list(S):
    # FIXME: name states check for list, but checks for tuple __and__ list. Thats confusing.
    return isinstance(S, (list, tuple))


def is_list_of_string(S):
    return isinstance(S, (list, tuple)) and (all(isinstance(s, basestring) for s in S))


def ensure_dtraj(dtraj):
    r"""Makes sure that dtraj is a discrete trajectory (array of int)

    """
    if is_int_vector(dtraj):
        return dtraj
    elif is_list_of_int(dtraj):
        return np.array(dtraj, dtype=int)
    else:
        raise TypeError('Argument dtraj is not a discrete trajectory - only list of integers or int-ndarrays are allowed. Check type.')


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


def ensure_int_vector(I, require_order=False):
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


def ensure_int_vector_or_None(F, require_order=False):
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
        return ensure_int_vector(F, require_order=require_order)


def ensure_float_vector(F, require_order=False):
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


def ensure_float_vector_or_None(F, require_order=False):
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
        return ensure_float_vector(F, require_order=require_order)


def ensure_dtype_float(x, default=np.float64):
    r"""Makes sure that x is type of float

    """
    if isinstance(x, np.ndarray):
        if x.dtype.kind == 'f':
            return x
        elif x.dtype.kind == 'i':
            return x.astype(default)
        else:
            raise TypeError('x is of type ' + str(x.dtype) + ' that cannot be converted to float')
    else:
        raise TypeError('x is not an array')


def ensure_traj(traj):
    r"""Makes sure that dtraj is a discrete trajectory (array of int)

    """
    if is_float_matrix(traj):
        return traj
    elif is_float_vector(traj):
        return traj[:, None]
    else:
        try:
            arr = np.array(traj)
            arr = ensure_dtype_float(arr)
            if is_float_matrix(arr):
                return arr
            if is_float_vector(arr):
                return arr[:, None]
            else:
                raise TypeError('Argument traj cannot be cast into a two-dimensional array. Check type.')
        except:
            raise TypeError('Argument traj is not a trajectory - only float-arrays or list of float-arrays are allowed. Check type.')


def ensure_traj_list(trajs):
    if isinstance(trajs, list):
        # elements are ints? make it a matrix and wrap into a list
        if is_list_of_float(trajs):
            return [np.array(trajs)[:, None]]
        else:
            res = []
            for i in range(len(trajs)):
                res.append(ensure_traj(trajs[i]))
            return res
    else:
        # looks like this is one trajectory
        return [ensure_traj(trajs)]