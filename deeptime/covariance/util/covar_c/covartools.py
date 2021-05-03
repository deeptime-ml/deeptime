import numpy as np


def variable_cols(data: np.ndarray, tol=0.0, min_constant=0):
    """ Evaluates which columns are constant (0) or variable (1)

    Parameters
    ----------
    data : ndarray
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
    from ._covartools import variable_cols as impl
    # prepare column array
    cols = np.zeros(data.shape[1], dtype=bool, order='C')
    completed = impl(cols, data, tol, min_constant)

    # if interrupted, return all ones. Otherwise return the variable columns as bool array
    return cols if completed == 1 else np.ones_like(cols, dtype=bool)
