import numpy
import ctypes
cimport numpy

cdef extern from "_covartools.h":
    void _variable_cols_char(int* cols, char* X, int M, int N, int min_constant)

cdef extern from "_covartools.h":
    void _variable_cols_int(int* cols, int* X, int M, int N, int min_constant)

cdef extern from "_covartools.h":
    void _variable_cols_long(int* cols, long* X, int M, int N, int min_constant)

cdef extern from "_covartools.h":
    void _variable_cols_float(int* cols, float* X, int M, int N, int min_constant)

cdef extern from "_covartools.h":
    void _variable_cols_double(int* cols, double* X, int M, int N, int min_constant)

#cdef extern from "_covartools.h":
#    void _variable_cols_double(int* cols, double* X, int M, int N)

cdef extern from "_covartools.h":
    void _variable_cols_float_approx(int* cols, float* X, int M, int N, float tol, int min_constant)

cdef extern from "_covartools.h":
    void _variable_cols_double_approx(int* cols, double* X, int M, int N, double tol, int min_constant)

cdef extern from "_covartools.h":
    void _subtract_row_double(double* X, double* row, int M, int N)

cdef extern from "_covartools.h":
    void _subtract_row_float(float* X, float* row, int M, int N)

cdef extern from "_covartools.h":
    void _subtract_row_double_copy(double* X0, double* X, double* row, int M, int N)

cdef extern from "_covartools.h":
    void _subtract_row_float_copy(float* X0, float* X, float* row, int M, int N)


# ================================================
# Check for constant columns
# ================================================

def variable_cols_char(cols, X, M, N, min_constant=0):
    pcols = <int*> numpy.PyArray_DATA(cols)
    pX = <char*> numpy.PyArray_DATA(X)
    return _variable_cols_char(pcols, pX, M, N, min_constant)

def variable_cols_int(cols, X, M, N, min_constant=0):
    pcols = <int*> numpy.PyArray_DATA(cols)
    pX = <int*> numpy.PyArray_DATA(X)
    return _variable_cols_int(pcols, pX, M, N, min_constant)

def variable_cols_long(cols, X, M, N, min_constant=0):
    pcols = <int*> numpy.PyArray_DATA(cols)
    pX = <long*> numpy.PyArray_DATA(X)
    return _variable_cols_long(pcols, pX, M, N, min_constant)

def variable_cols_float(cols, X, M, N, tol=0.0, min_constant=0):
    pcols = <int*> numpy.PyArray_DATA(cols)
    pX = <float*> numpy.PyArray_DATA(X)
    if tol == 0.0:
        return _variable_cols_float(pcols, pX, M, N, min_constant)
    else:
        return _variable_cols_float_approx(pcols, pX, M, N, numpy.float32(tol), min_constant)

def variable_cols_double(cols, X, M, N, tol=0.0, min_constant=0):
    pcols = <int*> numpy.PyArray_DATA(cols)
    pX = <double*> numpy.PyArray_DATA(X)
    if tol == 0.0:
        return _variable_cols_double(pcols, pX, M, N, min_constant)
    else:
        return _variable_cols_double_approx(pcols, pX, M, N, tol, min_constant)

def variable_cols(X, tol=0, min_constant=0):
    """ Evaluates which columns are constant (0) or variable (1)

    Parameters
    ----------
    X : ndarray
        Matrix whose columns will be checked for constant or variable.
    tol : float
        Tolerance for float-matrices. When set to 0 only equal columns with
        values will be considered constant. When set to a positive value,
        columns where all elements have absolute differences to the first
        element of that column are considered constant.
    min_constant : int
        Minimal number of constant columns to resume operation. If at one
        point the number of constant columns drops below min_constant, the
        computation will stop and all columns will be assumed to be variable.
        In this case, an all-True array will be returned.

    Returns
    -------
    variable : bool-array
        Array with number of elements equal to the columns. True: column is
        variable / nonconstant. False: column is constant.

    """
    if X is None:
        return None
    M, N = X.shape

    # prepare column array
    cols = numpy.zeros( (N), dtype=ctypes.c_int, order='C' )

    if X.dtype == numpy.float64:
        completed = variable_cols_double(cols, X, M, N, tol=tol, min_constant=min_constant)
    elif X.dtype == numpy.float32:
        completed = variable_cols_float(cols, X, M, N, tol=tol, min_constant=min_constant)
    elif X.dtype == numpy.int32:
        completed = variable_cols_int(cols, X, M, N, min_constant=min_constant)
    elif X.dtype == numpy.int64:
        completed = variable_cols_long(cols, X, M, N, min_constant=min_constant)
    elif X.dtype == numpy.bool:
        completed = variable_cols_char(cols, X, M, N, min_constant=min_constant)
    else:
        raise TypeError('unsupported type of X: '+str(X.dtype))

    # if interrupted, return all ones. Otherwise return the variable columns as bool array
    if completed == 0:
        return numpy.ones(cols, dtype=numpy.bool)
    else:
        return numpy.array(cols, dtype=numpy.bool)

# ================================================
# Row subtraction
# ================================================

def subtract_row_float(X, row, M, N):
    prow = <float*> numpy.PyArray_DATA(row)
    pX = <float*> numpy.PyArray_DATA(X)
    _subtract_row_float(pX, prow, M, N)

def subtract_row_double(X, row, M, N):
    prow = <double*> numpy.PyArray_DATA(row)
    pX = <double*> numpy.PyArray_DATA(X)
    _subtract_row_double(pX, prow, M, N)

def subtract_row_double_copy(X, row, M, N):
    X0 = numpy.zeros( X.shape, dtype=ctypes.c_double, order='C' )
    pX0 = <double*> numpy.PyArray_DATA(X0)
    pX = <double*> numpy.PyArray_DATA(X)
    prow = <double*> numpy.PyArray_DATA(row)
    _subtract_row_double_copy(pX0, pX, prow, M, N)
    return X0

def subtract_row_float_copy(X, row, M, N):
    X0 = numpy.zeros( X.shape, dtype=ctypes.c_double, order='C' )
    pX0 = <float*> numpy.PyArray_DATA(X0)
    pX = <float*> numpy.PyArray_DATA(X)
    prow = <float*> numpy.PyArray_DATA(row)
    _subtract_row_float_copy(pX0, pX, prow, M, N)
    return X0


def subtract_row(X, row, inplace=False):
    """ Subtracts given row from each row of array

    Parameters
    ----------
    X : ndarray (M, N)
        Matrix whose rows will be shifted.
    row : ndarray (N)
        Row vector that will be subtracted from each row of X.
    inplace : bool
        True: X will be changed. False: A copy of X will be created and X will remain unchanged.

    Returns
    -------
    X0 : ndarray (M, N)
        The row-shifted data

    """
    M, N = X.shape

    if X.dtype == numpy.float64 and row.dtype == numpy.float64:
        if inplace:
            subtract_row_double(X, row, M, N)
        else:
            X = subtract_row_double_copy(X, row, M, N)
    elif X.dtype == numpy.float32 and row.dtype == numpy.float32:
        if inplace:
            subtract_row_float(X, row, M, N)
        else:
            X = subtract_row_float_copy(X, row, M, N)
    else:
        raise TypeError('unsupported or inconsistent types: '+str(X.dtype)+' '+str(row.dtype))

    return X