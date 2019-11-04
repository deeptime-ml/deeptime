import numpy
import ctypes
cimport numpy

cdef extern from "_gaussian.h":
    void _p_o(const double o, const double* mus, const double* sigmas, const int N, double* p)

cdef extern from "_gaussian.h":
    void _p_obs(const double* o, const double* mus, const double* sigmas, const int N, const int T, double* p)

def cdef_double_vector(n):
    cdef numpy.ndarray[double, ndim=1, mode="c"] out = numpy.zeros( (n), dtype=ctypes.c_double, order='C' )
    return out

def cdef_single_vector(n):
    cdef numpy.ndarray[float, ndim=1, mode="c"] out = numpy.zeros( (n), dtype=ctypes.c_float, order='C' )
    return out

def cdef_double_matrix(n1,n2):
    cdef numpy.ndarray[double, ndim=2, mode="c"] out = numpy.zeros( (n1,n2), dtype=ctypes.c_double, order='C' )
    return out

def cdef_single_matrix(n1,n2):
    cdef numpy.ndarray[float, ndim=2, mode="c"] out = numpy.zeros( (n1,n2), dtype=ctypes.c_float, order='C' )
    return out

# def p_o_32(o, mus, sigmas, out=None):
#     # number of states
#     N = mus.shape[0]
#
#     pmus    = <float*> numpy.PyArray_DATA(mus)
#     psigmas = <float*> numpy.PyArray_DATA(sigmas)
#     p  = cdef_single_vector(N)
#     pp = <float*> numpy.PyArray_DATA(p)
#
#     return _p_o(o, pmus, psigmas, N, pp)


def p_o_64(o, mus, sigmas, out=None):
    # number of states
    N = mus.shape[0]

    pmus    = <double*> numpy.PyArray_DATA(mus)
    psigmas = <double*> numpy.PyArray_DATA(sigmas)
    if out is None:
        p = cdef_double_vector(N)
    else:
        p = out
    pp = <double*> numpy.PyArray_DATA(p)

    _p_o(o, pmus, psigmas, N, pp)

    return p


def p_o(o, mus, sigmas, out=None, dtype=numpy.float32):
    # check types
    assert(mus.dtype == dtype)
    assert(sigmas.dtype == dtype)

    # pointers to arrays
    if dtype == numpy.float32:
        raise ValueError
    elif dtype == numpy.float64:
        return p_o_64(o, mus, sigmas, out=out)
    else:
        raise TypeError


def p_obs_64(obs, mus, sigmas, out=None):
    N = mus.shape[0]
    T = obs.shape[0]
    pobs    = <double*> numpy.PyArray_DATA(obs)
    pmus    = <double*> numpy.PyArray_DATA(mus)
    psigmas = <double*> numpy.PyArray_DATA(sigmas)
    if out is None:
        p = cdef_double_matrix(T,N)
    else:
        p = out
    pp      = <double*> numpy.PyArray_DATA(p)

    _p_obs(pobs, pmus, psigmas, N, T, pp)

    return p


def p_obs(obs, mus, sigmas, out=None, dtype=numpy.float32):
    if (obs.dtype != dtype):
        obs = obs.astype(dtype)
    if (mus.dtype != dtype):
        mus = mus.astype(dtype)
    if (sigmas.dtype != dtype):
        sigmas = sigmas.astype(dtype)
    # check types
    assert(obs.dtype == dtype)
    assert(mus.dtype == dtype)
    assert(sigmas.dtype == dtype)

    # pointers to arrays
    if dtype == numpy.float32:
        raise ValueError
    elif dtype == numpy.float64:
        return p_obs_64(obs, mus, sigmas, out=out)
    else:
        raise TypeError
