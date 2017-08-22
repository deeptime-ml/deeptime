import numpy
from pyemma._ext.variational.estimators.covar_c._covartools import (variable_cols_double,
                                                                    variable_cols_float,
                                                                    variable_cols_int,
                                                                    variable_cols_long,
                                                                    variable_cols_char,
                                                                    variable_cols_approx_float,
                                                                    variable_cols_approx_double,
                                                                    )


def variable_cols(X, tol=0.0, min_constant=0):
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
    cols = numpy.zeros(N, dtype=numpy.int32, order='C')

    if X.dtype == numpy.float64:
        if tol == 0.0:
            completed = variable_cols_double(cols, X, M, N, min_constant)
        else:
            completed = variable_cols_approx_double(cols, X, M, N, tol, min_constant)
    elif X.dtype == numpy.float32:
        if tol == 0.0:
            completed = variable_cols_float(cols, X, M, N, min_constant)
        else:
            completed = variable_cols_approx_float(cols, X, M, N, tol, min_constant)
    elif X.dtype == numpy.int32:
        completed = variable_cols_int(cols, X, M, N, min_constant)
    elif X.dtype == numpy.int64:
        completed = variable_cols_long(cols, X, M, N, min_constant)
    elif X.dtype == numpy.bool:
        completed = variable_cols_char(cols, X, M, N, min_constant)
    else:
        raise TypeError('unsupported type of X: '+str(X.dtype))

    # if interrupted, return all ones. Otherwise return the variable columns as bool array
    if completed == 0:
        return numpy.ones_like(cols, dtype=numpy.bool)
    else:
        return numpy.array(cols, dtype=numpy.bool)
