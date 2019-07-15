import numpy


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
        variable / non-constant. False: column is constant.

    """
    if X is None:
        return None
    from ._covartools import (variable_cols_double,
                              variable_cols_float,
                              variable_cols_int,
                              variable_cols_long,
                              variable_cols_char)
    # prepare column array
    cols = numpy.zeros(X.shape[1], dtype=numpy.bool, order='C')

    if X.dtype == numpy.float64:
        completed = variable_cols_double(cols, X, tol, min_constant)
    elif X.dtype == numpy.float32:
        completed = variable_cols_float(cols, X, tol, min_constant)
    elif X.dtype == numpy.int32:
        completed = variable_cols_int(cols, X, 0, min_constant)
    elif X.dtype == numpy.int64:
        completed = variable_cols_long(cols, X, 0, min_constant)
    elif X.dtype == numpy.bool:
        completed = variable_cols_char(cols, X, 0, min_constant)
    else:
        raise TypeError('unsupported type of X: %s' % X.dtype)

    # if interrupted, return all ones. Otherwise return the variable columns as bool array
    if completed == 0:
        return numpy.ones_like(cols, dtype=numpy.bool)

    return cols
