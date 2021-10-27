"""
Data Types
----------
The standard data type for covariance computations is
float64, because the double precision (but not single precision) is
usually sufficient to compute the long sums involved in covariance
matrix computations. Integer types are avoided even if the data is integer,
because the BLAS matrix multiplication is very fast with floats, but very
slow with integers. If X is of boolean type (0/1), the standard data type
is float32, because this will be sufficient to represent numbers up to 2^23
without rounding error, which is usually sufficient sufficient as the
largest element in np.dot(X.T, X) can then be T, the number of data points.

Efficient Use
-------------
In order to get speedup with boolean input, remove_mean=False is required.
Note that you can still do TICA that way.

Sparsification
--------------
We aim at computing covariance matrices. For large (T x N) data matrices X, Y,
the bottleneck of this operation is computing the matrix product np.dot(X.T, X),
or np.dot(X.T, Y), with algorithmic complexity O(N^2 T). If X, Y have zero or
constant columns, we can reduce N and thus reduce the algorithmic complexity.

However, the BLAS matrix product used by np.dot() is highly Cache optimized -
the data is accessed in a way that most operations are done in cache, making the
calculation extremely efficient. Thus, even if X, Y have zero or constant columns,
it does not always pay off to interfere with this operation - one one hand by
spending compute time to determine the sparsity of the matrices, one the other
hand by using slicing operations that reduce the algorithmic complexity, but may
destroy the order of the data and thus produce more cache failures.

In order to make an informed decision, we have compared the runtime of the following
operations using matrices of various different sizes (T x N) and different degrees
of sparsity. (using an Intel Core i7 with OS/X 10.10.1):

    1. Compute np.dot(X.T, X)
    2. Compute np.dot(X[:, sel].T, X[:, sel]) where sel selects the nonzero columns
    3. Make a copy X0 = X[:, sel].copy() and then compute np.dot(X0.T, X0)

It may seem that step 3 is not a good idea because we make the extra effort of
copying the matrix. However, the new copy will have data ordered sequentially in
memory, and therefore better prepared for the algorithmically more expensive but
cache-optimized matrix product.

We have empirically found that:

    * Making a copy before running np.dot (option 3) is in most cases better than
      using the dot product on sliced arrays (option 2). Exceptions are when the
      data is extremely sparse, such that only a few columns are selected.
    * Copying and subselecting columns (option 3) is only faster than the full
      dot product (option 1), if 50% or less columns are selected. This observation
      is roughly independent of N.
    * The observations above are valid for  matrices (T x N) that are sufficiently
      large. We assume that "sufficiently large" means that they don't fully fit
      in the cache. For small matrices, the trends are less clear and different
      rules may apply.

In order to optimize covariance calculation for large matrices, we therefore
take the following actions:

    1. Given matrix size of X (and Y), determine the minimum number of columns
       that need to be constant in order to use sparse computation.
    2. Efficiently determine sparsity of X (and Y). Give up as soon as the
       number of constant column candidates drops below the minimum number, to
       avoid wasting time on the decision.
    3. Subselect the desired columns and copy the data to a new array X0 (Y0).
    4. Run operation on the new array X0 (Y0), including in-place substraction
       of the mean if needed.

"""

__author__ = 'noe'

import math
import numpy as np
from .covar_c import covartools


def _is_zero(x):
    """ Returns True if x is numerically 0 or an array with 0's. """
    return x is None or np.all(np.asarray(x) == 0)


def _sparsify(X, remove_mean=False, modify_data=False, sparse_mode='auto', sparse_tol=0.0):
    """ Determines the sparsity of X and returns a selected sub-matrix

    Only conducts sparsification if the number of constant columns is at least
    max(a N - b, min_const_col_number),

    Parameters
    ----------
    X : ndarray
        data matrix
    remove_mean : bool
        True: remove column mean from the data, False: don't remove mean.
    modify_data : bool
        If remove_mean=True, the mean will be removed in the data matrix X,
        without creating an independent copy. This option is faster but might
        lead to surprises because your input array is changed.
    sparse_mode : str
        one of:
            * 'dense' : always use dense mode
            * 'sparse' : always use sparse mode if possible
            * 'auto' : automatic

    Returns
    -------
    X0 : ndarray (view of X)
        Either X itself (if not sufficiently sparse), or a sliced view of X,
        containing only the variable columns
    mask : ndarray(N, dtype=bool) or None
        Bool selection array that indicates which columns of X were selected for
        X0, i.e. X0 = X[:, mask]. mask is None if no sparse selection was made.
    xconst : ndarray(N)
        Constant column values that are outside the sparse selection, i.e.
        X[i, ~mask] = xconst for any row i. xconst=0 if no sparse selection was made.

    """
    if sparse_mode.lower() == 'sparse':
        min_const_col_number = 0  # enforce sparsity. A single constant column will lead to sparse treatment
    elif sparse_mode.lower() == 'dense':
        min_const_col_number = X.shape[1] + 1  # never use sparsity
    else:
        if remove_mean and not modify_data:  # in this case we have to copy the data anyway, and can be permissive
            min_const_col_number = max(0.1 * X.shape[1], 50)
        else:
            # This is a rough heuristic to choose a minimum column number for which sparsity may pay off.
            # This heuristic is good for large number of samples, i.e. it may be inadequate for small matrices X.
            if X.shape[1] < 250:
                min_const_col_number = X.shape[1] - 0.25 * X.shape[1]
            elif X.shape[1] < 1000:
                min_const_col_number = X.shape[1] - (0.5 * X.shape[1] - 100)
            else:
                min_const_col_number = X.shape[1] - (0.8 * X.shape[1] - 400)
        # ensure we have an integer again.
        min_const_col_number = int(min_const_col_number)

    if X.shape[1] > min_const_col_number:
        mask = covartools.variable_cols(X, tol=sparse_tol, min_constant=min_const_col_number)  # bool vector
        nconst = len(np.where(~mask)[0])
        if nconst > min_const_col_number:
            xconst = X[0, ~mask]
            X = X[:, mask]  # sparsify
        else:
            xconst = None
            mask = None
    else:
        xconst = None
        mask = None

    return X, mask, xconst  # None, 0 if not sparse


def _sparsify_pair(X, Y, remove_mean=False, modify_data=False, symmetrize=False, sparse_mode='auto', sparse_tol=0.0):
    """
    """
    T = X.shape[0]
    N = math.sqrt(X.shape[1] * Y.shape[1])
    # check each data set separately for sparsity.
    X0, mask_X, xconst = _sparsify(X, sparse_mode=sparse_mode, sparse_tol=sparse_tol)
    Y0, mask_Y, yconst = _sparsify(Y, sparse_mode=sparse_mode, sparse_tol=sparse_tol)
    # if we have nonzero constant columns and the number of samples is too small, do not treat as
    # sparse, because then the const-specialized dot product function doesn't pay off.
    is_const = not (_is_zero(xconst) and _is_zero(yconst))
    if is_const and (symmetrize or not remove_mean) and 10*T < N:
        return X, None, None, Y, None, None
    else:
        return X0, mask_X, xconst, Y0, mask_Y, yconst


def _copy_convert(X, const=None, remove_mean=False, copy=True):
    r""" Makes a copy or converts the data type if needed

    Copies the data and converts the data type if unsuitable for covariance
    calculation. The standard data type for covariance computations is
    float64, because the double precision (but not single precision) is
    usually sufficient to compute the long sums involved in covariance
    matrix computations. Integer types are avoided even if the data is integer,
    because the BLAS matrix multiplication is very fast with floats, but very
    slow with integers. If X is of boolean type (0/1), the standard data type
    is float32, because this will be sufficient to represent numbers up to 2^23
    without rounding error, which is usually sufficient sufficient as the
    largest element in np.dot(X.T, X) can then be T, the number of data points.

    Parameters
    ----------
    remove_mean : bool
        If True, will enforce float64 even if the input is boolean
    copy : bool
        If True, enforces a copy even if the data type doesn't require it.

    Return
    ------
    X : ndarray
        copy or reference to X if no copy was needed.
    const : ndarray or None
        copy or reference to const if no copy was needed.

    """
    # determine type
    dtype = np.float64  # default: convert to float64 in order to avoid cancellation errors
    if X.dtype.kind == 'b' and X.shape[0] < 2**23 and not remove_mean:
        dtype = np.float32  # convert to float32 if we can represent all numbers
    # copy/convert if needed
    if X.dtype not in (np.float64, dtype):  # leave as float64 (conversion is expensive), otherwise convert to dtype
        X = X.astype(dtype, order='C')
        if const is not None:
            const = const.astype(dtype, order='C')
    elif copy:
        X = X.copy(order='C')
        if const is not None:
            const = const.copy(order='C')

    return X, const


def _sum_sparse(xsum, mask_X, xconst, T):
    s = np.zeros(len(mask_X))
    s[mask_X] = xsum
    s[~mask_X] = T * xconst
    return s


def _sum(X, xmask=None, xconst=None, Y=None, ymask=None, yconst=None, symmetric=False, remove_mean=False,
         weights=None):
    r""" Computes the column sums and centered column sums.

    If symmetric = False, the sums will be determined as
    .. math:
        sx &=& \frac{1}{2} \sum_t x_t
        sy &=& \frac{1}{2} \sum_t y_t

    If symmetric, the sums will be determined as

    .. math:
        sx = sy = \frac{1}{2T} \sum_t x_t + y_t

    Returns
    -------
    w : float
        statistical weight of sx, sy
    sx : ndarray
        effective row sum of X (including symmetrization if requested)
    sx_raw_centered : ndarray
        centered raw row sum of X

    optional returns (only if Y is given):

    sy : ndarray
        effective row sum of X (including symmetrization if requested)
    sy_raw_centered : ndarray
        centered raw row sum of Y

    """
    T = X.shape[0]
    # Check if weights are given:
    if weights is not None:
        X = weights[:, None] * X
        if Y is not None:
            Y = weights[:, None] * Y
    # compute raw sums on variable data
    sx_raw = X.sum(axis=0)  # this is the mean before subtracting it.
    sy_raw = 0
    if Y is not None:
        sy_raw = Y.sum(axis=0)

    # expand raw sums to full data
    if xmask is not None:
        if weights is not None:
            sx_raw = _sum_sparse(sx_raw, xmask, xconst, weights.sum())
        else:
            sx_raw = _sum_sparse(sx_raw, xmask, xconst, T)
    if ymask is not None:
        if weights is not None:
            sy_raw = _sum_sparse(sy_raw, ymask, yconst, weights.sum())
        else:
            sy_raw = _sum_sparse(sy_raw, ymask, yconst, T)

    # compute effective sums and centered sums
    if Y is not None and symmetric:
        sx = sx_raw + sy_raw
        sy = sx
        if weights is not None:
            w = 2*np.sum(weights)
        else:
            w = 2 * T
    else:
        sx = sx_raw
        sy = sy_raw
        if weights is not None:
            w = np.sum(weights)
        else:
            w = T

    sx_raw_centered = sx_raw.copy()
    if Y is not None:
        sy_raw_centered = sy_raw.copy()

    # center mean.
    if remove_mean:
        if Y is not None and symmetric:
            sx_raw_centered -= 0.5 * sx
            sy_raw_centered -= 0.5 * sy
        else:
            sx_raw_centered = np.zeros(sx.size)
            if Y is not None:
                sy_raw_centered = np.zeros(sy.size)

    # return
    if Y is not None:
        return w, sx, sx_raw_centered, sy, sy_raw_centered
    else:
        return w, sx, sx_raw_centered


def _center(X, w, s, mask=None, const=None, inplace=True):
    """ Centers the data.

    Parameters
    ----------
    w : float
        statistical weight of s
    inplace : bool
        center in place

    Returns
    -------
    sx : ndarray
        uncentered row sum of X
    sx_centered : ndarray
        row sum of X after centering

    optional returns (only if Y is given):

    sy_raw : ndarray
        uncentered row sum of Y
    sy_centered : ndarray
        row sum of Y after centering

    """
    xmean = s / float(w)
    if mask is None:
        X = np.subtract(X, xmean, out=X if inplace else None)
    else:
        X = np.subtract(X, xmean[mask], out=X if inplace else None)
        const = np.subtract(const, xmean[~mask], const if inplace else None)

    return X, const


def _filter_variable_indices(mask, column_selection):
    """ Returns column indices restricted to the variable columns as determined by the given mask.

    Parameters
    ----------
    mask : ndarray(N, dtype=bool)
        Array indicating the variable columns.
    column_selection : ndarray(k, dtype=int)
        Column indices to be filtered and mapped.

    Returns
    -------
    ix : ndarray(l, dtype=int)
        Column indices restricted to the variable columns, mapped to the correct index range.

    """
    a = np.where(mask)[0]
    b = column_selection[np.in1d(column_selection, a)]
    return np.searchsorted(a, b)


# ====================================================================================
# SECOND MOMENT MATRICES / COVARIANCES
# ====================================================================================

def _M2_dense(X, Y, weights=None, diag_only=False):
    """ 2nd moment matrix using dense matrix computations.

    This function is encapsulated such that we can make easy modifications of the basic algorithms

    """
    if weights is not None:
        if diag_only:
            return np.sum(weights[:, None] * X * Y, axis=0)
        else:
            return np.dot((weights[:, None] * X).T, Y)
    else:
        if diag_only:
            return np.sum(X * Y, axis=0)
        else:
            return np.dot(X.T, Y)


def _M2_const(Xvar, mask_X, xvarsum, xconst, Yvar, mask_Y, yvarsum, yconst, weights=None):
    r""" Computes the unnormalized covariance matrix between X and Y, exploiting constant input columns

    Computes the unnormalized covariance matrix :math:`C = X^\top Y`
    (for symmetric=False) or :math:`C = \frac{1}{2} (X^\top Y + Y^\top X)`
    (for symmetric=True). Suppose the data matrices can be column-permuted
    to have the form

    .. math:
        X &=& (X_{\mathrm{var}}, X_{\mathrm{const}})
        Y &=& (Y_{\mathrm{var}}, Y_{\mathrm{const}})

    with rows:

    .. math:
        x_t &=& (x_{\mathrm{var},t}, x_{\mathrm{const}})
        y_t &=& (y_{\mathrm{var},t}, y_{\mathrm{const}})

    where :math:`x_{\mathrm{const}},\:y_{\mathrm{const}}` are constant vectors.
    The resulting matrix has the general form:

    .. math:
        C &=& [X_{\mathrm{var}}^\top Y_{\mathrm{var}}  x_{sum} y_{\mathrm{const}}^\top ]
          & & [x_{\mathrm{const}}^\top y_{sum}^\top    x_{sum} x_{sum}^\top            ]

    where :math:`x_{sum} = \sum_t x_{\mathrm{var},t}` and
    :math:`y_{sum} = \sum_t y_{\mathrm{var},t}`.

    Parameters
    ----------
    Xvar : ndarray (T, m)
        Part of the data matrix X with :math:`m \le M` variable columns.
    mask_X : ndarray (M)
        Boolean array of size M of the full columns. False for constant column,
        True for variable column in X.
    xvarsum : ndarray (m)
        Column sum of variable part of data matrix X
    xconst : ndarray (M-m)
        Values of the constant part of data matrix X
    Yvar : ndarray (T, n)
        Part of the data matrix Y with :math:`n \le N` variable columns.
    mask_Y : ndarray (N)
        Boolean array of size N of the full columns. False for constant column,
        True for variable column in Y.
    yvarsum : ndarray (n)
        Column sum of variable part of data matrix Y
    yconst : ndarray (N-n)
        Values of the constant part of data matrix Y
    weights : None or ndarray (N)
        weights for all time steps.

    Returns
    -------
    C : ndarray (M, N)
        Unnormalized covariance matrix.

    """
    C = np.zeros((len(mask_X), len(mask_Y)))
    # Block 11
    C[np.ix_(mask_X, mask_Y)] = _M2_dense(Xvar, Yvar, weights=weights)
    # other blocks
    xsum_is_0 = _is_zero(xvarsum)
    ysum_is_0 = _is_zero(yvarsum)
    xconst_is_0 = _is_zero(xconst)
    yconst_is_0 = _is_zero(yconst)
    # TODO: maybe we don't need the checking here, if we do the decision in the higher-level function M2
    # TODO: if not zero, we could still exploit the zeros in const and compute (and write!) this outer product
    # TODO: only to a sub-matrix
    # Block 12 and 21
    if weights is not None:
        wsum = np.sum(weights)
        xvarsum = np.sum(weights[:, None] * Xvar, axis=0)
        yvarsum = np.sum(weights[:, None] * Yvar, axis=0)
    else:
        wsum = Xvar.shape[0]
    if not (xsum_is_0 or yconst_is_0) or not (ysum_is_0 or xconst_is_0):
        C[np.ix_(mask_X, ~mask_Y)] = np.outer(xvarsum, yconst)
        C[np.ix_(~mask_X, mask_Y)] = np.outer(xconst, yvarsum)
    # Block 22
    if not (xconst_is_0 or yconst_is_0):
        C[np.ix_(~mask_X, ~mask_Y)] = np.outer(wsum*xconst, yconst)
    return C


def _M2_sparse(Xvar, mask_X, Yvar, mask_Y, weights=None):
    """ 2nd moment matrix exploiting zero input columns """
    C = np.zeros((len(mask_X), len(mask_Y)))
    C[np.ix_(mask_X, mask_Y)] = _M2_dense(Xvar, Yvar, weights=weights)
    return C


def _M2_sparse_sym(Xvar, mask_X, Yvar, mask_Y, weights=None, column_selection=None):
    """ 2nd self-symmetric moment matrix exploiting zero input columns

    Computes X'X + Y'Y and X'Y + Y'X

    """
    assert len(mask_X) == len(mask_Y), 'X and Y need to have equal sizes for symmetrization'

    if column_selection is None:
        mask_Xk = mask_X
        mask_Yk = mask_Y
        Xvark = Xvar
        Yvark = Yvar
    else:
        mask_Xk = mask_X[column_selection]
        mask_Yk = mask_Y[column_selection]
        Xvark = Xvar[:, _filter_variable_indices(mask_X, column_selection)]
        Yvark = Yvar[:, _filter_variable_indices(mask_Y, column_selection)]

    Cxxyy = np.zeros((len(mask_X), len(mask_Yk)))
    Cxxyy[np.ix_(mask_X, mask_Xk)] = _M2_dense(Xvar, Xvark, weights=weights)
    Cxxyy[np.ix_(mask_Y, mask_Yk)] += _M2_dense(Yvar, Yvark, weights=weights)

    Cxyyx = np.zeros((len(mask_X), len(mask_Yk)))
    Cxy = _M2_dense(Xvar, Yvark, weights=weights)
    Cyx = _M2_dense(Yvar, Xvark, weights=weights)
    Cxyyx[np.ix_(mask_X, mask_Yk)] = Cxy
    Cxyyx[np.ix_(mask_Y, mask_Xk)] += Cyx

    return Cxxyy, Cxyyx


def _default_mask(n):
    return np.ones(n, dtype=np.bool_), np.ones(0, dtype=float)


def _M2(Xvar, Yvar, mask_X=None, mask_Y=None, xsum=0, xconst=0, ysum=0, yconst=0, weights=None, diag_only=False):
    """ direct (nonsymmetric) second moment matrix. Decide if we need dense, sparse, const"""
    if mask_X is None and mask_Y is None:
        return _M2_dense(Xvar, Yvar, weights=weights, diag_only=diag_only)
    else:
        # Check if one of the masks is not None, modify it and also adjust the constant columns:
        mask_X, xconst = _default_mask(Xvar.shape[1]) if mask_X is None else (mask_X, xconst)
        mask_Y, yconst = _default_mask(Yvar.shape[1]) if mask_Y is None else (mask_Y, yconst)
    if _is_zero(xsum) and _is_zero(ysum) or _is_zero(xconst) and _is_zero(yconst):
        return _M2_sparse(Xvar, mask_X, Yvar, mask_Y, weights=weights)
    else:
        return _M2_const(Xvar, mask_X, xsum[mask_X], xconst, Yvar, mask_Y, ysum[mask_Y], yconst, weights=weights)


def _M2_symmetric(Xvar, Yvar, mask_X=None, mask_Y=None, xsum=0, xconst=0, ysum=0, yconst=0, weights=None,
                  column_selection=None, diag_only=False):
    """ symmetric second moment matrices. Decide if we need dense, sparse, const"""
    if mask_X is None and mask_Y is None:
        if column_selection is None:
            Xvark = Xvar
            Yvark = Yvar
        else:
            Xvark = Xvar[:, column_selection]
            Yvark = Yvar[:, column_selection]
        Cxxyy = _M2_dense(Xvar, Xvark, weights=weights, diag_only=diag_only) \
                + _M2_dense(Yvar, Yvark, weights=weights, diag_only=diag_only)
        Cxy = _M2_dense(Xvar, Yvark, weights=weights, diag_only=diag_only)
        Cyx = _M2_dense(Yvar, Xvark, weights=weights, diag_only=diag_only)
        Cxyyx = Cxy + Cyx
    else:
        # Check if one of the masks is not None, modify it and also adjust the constant columns:
        mask_X, xconst = _default_mask(Xvar.shape[1]) if mask_X is None else (mask_X, xconst)
        mask_Y, yconst = _default_mask(Yvar.shape[1]) if mask_Y is None else (mask_Y, yconst)
        if _is_zero(xsum) and _is_zero(ysum) or _is_zero(xconst) and _is_zero(yconst):
            Cxxyy, Cxyyx = _M2_sparse_sym(Xvar, mask_X, Yvar, mask_Y, weights=weights, column_selection=column_selection)
        else:
            xvarsum = xsum[mask_X]  # to variable part
            yvarsum = ysum[mask_Y]  # to variable part
            if column_selection is None:
                Xvark = Xvar
                mask_Xk = mask_X
                xkvarsum = xvarsum
                xkconst = xconst
                Yvark = Yvar
                mask_Yk = mask_Y
                ykvarsum = yvarsum
                ykconst = yconst
            else:
                Xvark = Xvar[:, _filter_variable_indices(mask_X, column_selection)]
                mask_Xk = mask_X[column_selection]
                xksum = xsum[column_selection]
                xkvarsum = xksum[mask_Xk]
                xkconst = xconst[_filter_variable_indices(~mask_X, column_selection)]
                Yvark = Yvar[:, _filter_variable_indices(mask_Y, column_selection)]
                mask_Yk = mask_Y[column_selection]
                yksum = ysum[column_selection]
                ykvarsum = yksum[mask_Yk]
                ykconst = yconst[_filter_variable_indices(~mask_Y, column_selection)]
            Cxxyy = _M2_const(Xvar, mask_X, xvarsum, xconst, Xvark, mask_Xk, xkvarsum, xkconst, weights=weights) \
                    + _M2_const(Yvar, mask_Y, yvarsum, yconst, Yvark, mask_Yk, ykvarsum, ykconst, weights=weights)
            Cxy = _M2_const(Xvar, mask_X, xvarsum, xconst, Yvark, mask_Yk, ykvarsum, ykconst, weights=weights)
            Cyx = _M2_const(Yvar, mask_Y, yvarsum, yconst, Xvark, mask_Xk, xkvarsum, xkconst, weights=weights)
            Cxyyx = Cxy + Cyx
    return Cxxyy, Cxyyx


# =================================================
# USER API
# =================================================


def moments_XX(X, remove_mean=False, modify_data=False, weights=None, sparse_mode='auto', sparse_tol=0.0,
               column_selection=None, diag_only=False):
    r""" Computes the first two unnormalized moments of X.

    Computes :math:`s = \sum_t x_t` and :math:`C = X^{\top} X` while exploiting
    zero or constant columns in the data matrix.

    Parameters
    ----------
    X : ndarray (T, M)
        Data matrix
    remove_mean : bool
        True: remove column mean from the data, False: don't remove mean.
    modify_data : bool
        If remove_mean=True, the mean will be removed in the data matrix X,
        without creating an independent copy. This option is faster but might
        lead to surprises because your input array is changed.
    weights: None or ndarray(T, )
        weights assigned to each trajectory point. If None, all data points have weight one.
        If ndarray, each data point is assigned a separate weight.
    sparse_mode : str
        one of:
            * 'dense' : always use dense mode
            * 'sparse' : always use sparse mode if possible
            * 'auto' : automatic
    sparse_tol: float
        Threshold for considering column to be zero in order to save computing
        effort when the data is sparse or almost sparse.
        If :code:`max(abs(X[:, i])) < sparse_tol`, then row i (and also column i if Y
        is not given) of the covariance matrix will be set to zero. If Y is
        given and :code:`max(abs(Y[:, i])) < sparse_tol`, then column i of the
        covariance matrix will be set to zero.
    column_selection: ndarray(k, dtype=int) or None
        Indices of those columns that are to be computed. If None, all columns are computed.
    diag_only: bool
        If True, the computation is restricted to the diagonal entries (autocorrelations) only.

    Returns
    -------
    w : float
        statistical weight
    s : ndarray (M)
        sum
    C : ndarray (M, M)
        unnormalized covariance matrix

    """
    # Check consistency of inputs:
    if weights is not None:
        assert X.shape[0] == weights.shape[0], 'X and weights_x must have equal length'
    # diag_only is only implemented for dense mode
    if diag_only and sparse_mode != 'dense':
        if sparse_mode == 'sparse':
            import warnings
            warnings.warn('Computing diagonal entries only is not implemented for sparse mode. Switching to dense mode.')
        sparse_mode = 'dense'
    # sparsify
    X0, mask_X, xconst = _sparsify(X, remove_mean=remove_mean, modify_data=modify_data,
                                   sparse_mode=sparse_mode, sparse_tol=sparse_tol)
    is_sparse = mask_X is not None
    # copy / convert
    # TODO: do we need to copy xconst?
    X0, xconst = _copy_convert(X0, const=xconst, remove_mean=remove_mean,
                               copy=is_sparse or (remove_mean and not modify_data))
    # sum / center
    w, sx, sx0_centered = _sum(X0, xmask=mask_X, xconst=xconst, symmetric=False, remove_mean=remove_mean,
                               weights=weights)
    if remove_mean:
        _center(X0, w, sx, mask=mask_X, const=xconst, inplace=True)  # fast in-place centering
    # TODO: we could make a second const check here. If after summation not enough zeros have appeared in the
    # TODO: consts, we switch back to dense treatment here.
    # compute covariance matrix
    if column_selection is not None:
        if is_sparse:
            Xk = X[:, column_selection]
            mask_Xk = mask_X[column_selection]
            X0k = Xk[:, mask_Xk]
            xksum = sx0_centered[column_selection]
            xkconst = Xk[0, ~mask_Xk]
            X0k, xkconst = _copy_convert(X0k, const=xkconst, remove_mean=remove_mean,
                                         copy=True)
            C = _M2(X0, X0k, mask_X=mask_X, mask_Y=mask_Xk, xsum=sx0_centered, xconst=xconst, ysum=xksum, yconst=xkconst,
                    weights=weights)
        else:
            X0k = X0[:, column_selection]
            C = _M2(X0, X0k, mask_X=mask_X, mask_Y=mask_X, xsum=sx0_centered, xconst=xconst,
                    ysum=sx0_centered[column_selection], yconst=xconst, weights=weights)
    else:
        C = _M2(X0, X0, mask_X=mask_X, mask_Y=mask_X, xsum=sx0_centered, xconst=xconst, ysum=sx0_centered, yconst=xconst,
                weights=weights, diag_only=diag_only)
    return w, sx, C


def moments_XXXY(X, Y, remove_mean=False, symmetrize=False, weights=None,
                 modify_data=False, sparse_mode='auto', sparse_tol=0.0,
                 column_selection=None, diag_only=False):
    r""" Computes the first two unnormalized moments of X and Y.

    If symmetrize is False, computes

    .. math::

        \begin{aligned}
        s_x  &= \sum_t x_t\\
        s_y  &= \sum_t y_t\\
        C_{XX} &= X^{\top} X\\
        C_{XY} &= X^{\top} Y
        \end{aligned}

    If symmetrize is True, computes

    .. math::

        \begin{aligned}
        s_x = s_y &= \frac{1}{2} \sum_t (x_t + y_t)\\
        C_XX      &= \frac{1}{2} (X^\top X + Y^\top Y)\\
        C_XY      &= \frac{1}{2} (X^\top Y + Y^\top X)
        \end{aligned}

    while exploiting zero or constant columns in the data matrix.

    Parameters
    ----------
    X : ndarray (T, M)
        Data matrix
    Y : ndarray (T, N)
        Second data matrix
    remove_mean : bool
        True: remove column mean from the data, False: don't remove mean.
    symmetrize : bool
        Computes symmetrized means and moments (see above)
    weights : None or ndarray(T, )
        weights assigned to each trajectory point of X. If None, all data points have weight one.
        If ndarray, each data point is assigned a separate weight.
    modify_data : bool
        If remove_mean=True, the mean will be removed in the data matrix X,
        without creating an independent copy. This option is faster but might
        lead to surprises because your input array is changed.
    sparse_mode : str
        one of:
            * 'dense' : always use dense mode
            * 'sparse' : always use sparse mode if possible
            * 'auto' : automatic
    sparse_tol: float
        Threshold for considering column to be zero in order to save computing
        effort when the data is sparse or almost sparse.
        If max(abs(X[:, i])) < sparse_tol, then row i (and also column i if Y
        is not given) of the covariance matrix will be set to zero. If Y is
        given and max(abs(Y[:, i])) < sparse_tol, then column i of the
        covariance matrix will be set to zero.
    column_selection: ndarray(k, dtype=int) or None
        Indices of those columns that are to be computed. If None, all columns are computed.
    diag_only: bool
        If True, the computation is restricted to the diagonal entries (autocorrelations) only.

    Returns
    -------
    w : float
        statistical weight
    s_x : ndarray (M)
        x-sum
    s_y : ndarray (N)
        y-sum
    C_XX : ndarray (M, M)
        unnormalized covariance matrix of X
    C_XY : ndarray (M, N)
        unnormalized covariance matrix of XY

    """
    # Check consistency of inputs:
    if Y is not None:
        assert Y.shape[0] == X.shape[0], 'X and Y must have equal length.'
    if weights is not None:
        assert X.shape[0] == weights.shape[0], 'X and weights_x must have equal length'
    # diag_only is only implemented for dense mode
    if diag_only and sparse_mode != 'dense':
        if sparse_mode == 'sparse':
            import warnings
            warnings.warn('Computing diagonal entries only is not implemented for sparse mode. Switching to dense mode.')
        sparse_mode = 'dense'
    if diag_only and X.shape[1] != Y.shape[1]:
        raise ValueError('Computing diagonal entries only does not make sense for rectangular covariance matrix.')
    # sparsify
    X0, mask_X, xconst, Y0, mask_Y, yconst = _sparsify_pair(X, Y, remove_mean=remove_mean, modify_data=modify_data,
                                                            symmetrize=symmetrize, sparse_mode=sparse_mode, sparse_tol=sparse_tol)
    is_sparse = mask_X is not None and mask_Y is not None
    # copy / convert
    copy = is_sparse or (remove_mean and not modify_data)
    X0, xconst = _copy_convert(X0, const=xconst, remove_mean=remove_mean, copy=copy)
    Y0, yconst = _copy_convert(Y0, const=yconst, remove_mean=remove_mean, copy=copy)
    # sum / center
    w, sx, sx_centered, sy, sy_centered = _sum(X0, xmask=mask_X, xconst=xconst, Y=Y0, ymask=mask_Y, yconst=yconst,
                                               symmetric=symmetrize, remove_mean=remove_mean, weights=weights)
    if remove_mean:
        _center(X0, w, sx, mask=mask_X, const=xconst, inplace=True)  # fast in-place centering
        _center(Y0, w, sy, mask=mask_Y, const=yconst, inplace=True)  # fast in-place centering

    if symmetrize:
        Cxx, Cxy = _M2_symmetric(X0, Y0, mask_X=mask_X, mask_Y=mask_Y,
                                 xsum=sx_centered, xconst=xconst, ysum=sy_centered, yconst=yconst, weights=weights,
                                 column_selection=column_selection, diag_only=diag_only)
    else:
        if column_selection is not None:
            if is_sparse:
                Xk = X[:, column_selection]
                mask_Xk = mask_X[column_selection]
                X0k = Xk[:, mask_Xk]
                xksum = sx_centered[column_selection]
                xkconst = Xk[0, ~mask_Xk]
                X0k, xkconst = _copy_convert(X0k, const=xkconst, remove_mean=remove_mean,
                                             copy=True)

                Yk = Y[:, column_selection]
                mask_Yk = mask_Y[column_selection]
                Y0k = Yk[:, mask_Yk]
                yksum = sy_centered[column_selection]
                ykconst = Yk[0, ~mask_Yk]
                Y0k, ykconst = _copy_convert(Y0k, const=ykconst, remove_mean=remove_mean,
                                             copy=True)

                Cxx = _M2(X0, X0k, mask_X=mask_X, mask_Y=mask_Xk, xsum=sx_centered, xconst=xconst, ysum=xksum, yconst=xkconst,
                        weights=weights)
                Cxy = _M2(X0, Y0k, mask_X=mask_X, mask_Y=mask_Yk, xsum=sx_centered, xconst=xconst, ysum=yksum, yconst=ykconst,
                        weights=weights)
            else:
                X0k = X0[:, column_selection]
                Y0k = Y0[:, column_selection]
                Cxx = _M2(X0, X0k, mask_X=mask_X, mask_Y=mask_X, xsum=sx_centered, xconst=xconst,
                        ysum=sx_centered[column_selection], yconst=xconst, weights=weights)
                Cxy = _M2(X0, Y0k, mask_X=mask_X, mask_Y=mask_Y, xsum=sx_centered, xconst=xconst,
                        ysum=sy_centered[column_selection], yconst=yconst, weights=weights)
        else:
            Cxx = _M2(X0, X0, mask_X=mask_X, mask_Y=mask_X, xsum=sx_centered, xconst=xconst, ysum=sx_centered, yconst=xconst,
                      weights=weights, diag_only=diag_only)
            Cxy = _M2(X0, Y0, mask_X=mask_X, mask_Y=mask_Y, xsum=sx_centered, xconst=xconst, ysum=sy_centered, yconst=yconst,
                      weights=weights, diag_only=diag_only)

    return w, sx, sy, Cxx, Cxy


def moments_block(X, Y, remove_mean=False, modify_data=False,
                  sparse_mode='auto', sparse_tol=0.0,
                  column_selection=None, diag_only=False):
    r""" Computes the first two unnormalized moments of X and Y forward and backward.

    Computes

    .. math::

        \begin{aligned}
        s_x  &= \sum_t x_t\\
        s_y  &= \sum_t y_t\\
        C_{XX} &= X^\top X\\
        C_{XY} &= X^\top Y\\
        C_{YX} &= Y^\top X\\
        C_{YY} &= Y^\top Y
        \end{aligned}

    while exploiting zero or constant columns in the data matrix.

    Parameters
    ----------
    X : ndarray (T, M)
        Data matrix
    Y : ndarray (T, N)
        Second data matrix
    remove_mean : bool
        True: remove column mean from the data, False: don't remove mean.
    modify_data : bool
        If remove_mean=True, the mean will be removed in the data matrix X,
        without creating an independent copy. This option is faster but might
        lead to surprises because your input array is changed.
    sparse_mode : str
        one of:
            * 'dense' : always use dense mode
            * 'sparse' : always use sparse mode if possible
            * 'auto' : automatic
    sparse_tol: float
        Threshold for considering column to be zero in order to save computing
        effort when the data is sparse or almost sparse.
        If max(abs(X[:, i])) < sparse_tol, then row i (and also column i if Y
        is not given) of the covariance matrix will be set to zero. If Y is
        given and max(abs(Y[:, i])) < sparse_tol, then column i of the
        covariance matrix will be set to zero.
    column_selection: ndarray(k, dtype=int) or None
        Indices of those columns that are to be computed. If None, all columns are computed.
    diag_only: bool
        If True, the computation is restricted to the diagonal entries (autocorrelations) only.

    Returns
    -------
    w : float
        statistical weight of this estimation
    s : [ndarray (M), ndarray (M)]
        list of two elements with s[0]=sx and s[1]=sy
    C : [[ndarray(M,M), ndarray(M,N)], [ndarray(N,M),ndarray(N,N)]]
        list of two lists with two elements.
        C[0,0] = Cxx, C[0,1] = Cxy, C[1,0] = Cyx, C[1,1] = Cyy
    """
    # diag_only is only implemented for dense mode
    if diag_only and sparse_mode != 'dense':
        if sparse_mode == 'sparse':
            import warnings
            warnings.warn('Computing diagonal entries only is not implemented for sparse mode. Switching to dense mode.')
        sparse_mode = 'dense'
    # sparsify
    X0, mask_X, xconst = _sparsify(X, sparse_mode=sparse_mode, sparse_tol=sparse_tol)
    Y0, mask_Y, yconst = _sparsify(Y, sparse_mode=sparse_mode, sparse_tol=sparse_tol)
    is_sparse = mask_X is not None and mask_Y is not None
    # copy / convert
    copy = is_sparse or (remove_mean and not modify_data)
    X0, xconst = _copy_convert(X0, const=xconst, copy=copy)
    Y0, yconst = _copy_convert(Y0, const=yconst, copy=copy)
    # sum / center
    w, sx, sx_centered, sy, sy_centered = _sum(X0, xmask=mask_X, xconst=xconst, Y=Y0, ymask=mask_Y, yconst=yconst,
                                               symmetric=False, remove_mean=remove_mean)
    if remove_mean:
        _center(X0, w, sx, mask=mask_X, const=xconst, inplace=True)  # fast in-place centering
        _center(Y0, w, sy, mask=mask_Y, const=yconst, inplace=True)  # fast in-place centering

    if column_selection is not None:
        if is_sparse:
            Xk = X[:, column_selection]
            mask_Xk = mask_X[column_selection] if mask_X is not None else mask_X
            X0k = Xk[:, mask_Xk]
            xksum = sx_centered[column_selection]
            xkconst = Xk[0, ~mask_Xk]
            X0k, xkconst = _copy_convert(X0k, const=xkconst, remove_mean=remove_mean,
                                         copy=True)

            Yk = Y[:, column_selection]
            mask_Yk = mask_Y[column_selection] if mask_Y is not None else mask_Y
            Y0k = Yk[:, mask_Yk]
            yksum = sy_centered[column_selection]
            ykconst = Yk[0, ~mask_Yk]
            Y0k, ykconst = _copy_convert(Y0k, const=ykconst, remove_mean=remove_mean,
                                         copy=True)

            Cxx = _M2(X0, X0k, mask_X=mask_X, mask_Y=mask_Xk,
                      xsum=sx_centered, xconst=xconst, ysum=xksum, yconst=xkconst)
            Cxy = _M2(X0, Y0k, mask_X=mask_X, mask_Y=mask_Yk,
                      xsum=sx_centered, xconst=xconst, ysum=yksum, yconst=ykconst)
            Cyx = _M2(Y0, X0k, mask_X=mask_Y, mask_Y=mask_Xk,
                      xsum=sy_centered, xconst=yconst, ysum=xksum, yconst=xkconst)
            Cyy = _M2(Y0, Y0k, mask_X=mask_Y, mask_Y=mask_Yk,
                      xsum=sy_centered, xconst=yconst, ysum=yksum, yconst=ykconst)
        else:
            X0k = X0[:, column_selection]
            Y0k = Y0[:, column_selection]
            Cxx = _M2(X0, X0k, mask_X=mask_X, mask_Y=mask_X,
                      xsum=sx_centered, xconst=xconst,
                      ysum=sx_centered[column_selection], yconst=xconst)
            Cxy = _M2(X0, Y0k, mask_X=mask_X, mask_Y=mask_Y,
                      xsum=sx_centered, xconst=xconst,
                      ysum=sy_centered[column_selection], yconst=yconst)
            Cyx = _M2(Y0, X0k, mask_X=mask_Y, mask_Y=mask_X,
                      xsum=sy_centered, xconst=yconst,
                      ysum=sx_centered[column_selection], yconst=xconst)
            Cyy = _M2(Y0, Y0k, mask_X=mask_Y, mask_Y=mask_Y,
                      xsum=sy_centered, xconst=yconst,
                      ysum=sy_centered[column_selection], yconst=yconst)
    else:
        Cxx = _M2(X0, X0, mask_X=mask_X, mask_Y=mask_X,
                  xsum=sx_centered, xconst=xconst, ysum=sx_centered, yconst=xconst,
                  diag_only=diag_only)
        Cxy = _M2(X0, Y0, mask_X=mask_X, mask_Y=mask_Y,
                  xsum=sx_centered, xconst=xconst, ysum=sy_centered, yconst=yconst,
                  diag_only=diag_only)
        Cyx = Cxy.T
        Cyy = _M2(Y0, Y0, mask_X=mask_Y, mask_Y=mask_Y,
                  xsum=sy_centered, xconst=yconst, ysum=sy_centered, yconst=yconst,
                  diag_only=diag_only)

    return w, (sx, sy), ((Cxx, Cxy), (Cyx, Cyy))


def covar(X, remove_mean=False, modify_data=False, weights=None, sparse_mode='auto', sparse_tol=0.0):
    """ Computes the covariance matrix of X.

    Computes :math:`C_{XX} = X^{\\top} X` while exploiting zero or constant columns in the data matrix.

    WARNING: Directly use :meth:`moments_XX` if you can. This function does an additional
    constant-matrix multiplication and does not return the mean.

    Parameters
    ----------
    X : ndarray (T, M)
        Data matrix
    remove_mean : bool
        True: remove column mean from the data, False: don't remove mean.
    modify_data : bool
        If remove_mean=True, the mean will be removed in the data matrix X,
        without creating an independent copy. This option is faster but might
        lead to surprises because your input array is changed.
    weights : None or ndarray(T, )
        weights assigned to each trajectory point of X. If None, all data points have weight one.
        If ndarray, each data point is assigned a separate weight.
    sparse_mode : str
        one of:
            * 'dense' : always use dense mode
            * 'sparse' : always use sparse mode if possible
            * 'auto' : automatic
    sparse_tol: float
        Threshold for considering column to be zero in order to save computing
        effort when the data is sparse or almost sparse.
        If :code:`max(abs(X[:, i])) < sparse_tol`, then row i (and also column i if Y
        is not given) of the covariance matrix will be set to zero. If Y is
        given and :code:`max(abs(Y[:, i])) < sparse_tol`, then column i of the
        covariance matrix will be set to zero.

    Returns
    -------
    C_XX : ndarray (M, M)
        Covariance matrix of X

    See also
    --------
    moments_XX

    """
    w, s, M = moments_XX(X, remove_mean=remove_mean, weights=weights, modify_data=modify_data,
                         sparse_mode=sparse_mode, sparse_tol=sparse_tol)
    return M / float(w)


def covars(X, Y, remove_mean=False, modify_data=False, symmetrize=False, weights=None, sparse_mode='auto',
           sparse_tol=0.0):
    r""" Computes the covariance and cross-covariance matrix of X and Y.

    If symmetrize is False, computes

    .. math::

        \begin{aligned}
        C_{XX} &= X^{\top} X \\
        C_{XY} &= X^{\top} Y
        \end{aligned}

    If symmetrize is True, computes

    .. math::

        \begin{aligned}
        C_{XX} &= \frac{1}{2} (X^{\top} X + Y^{\top} Y)\\
        C_{XY} &= \frac{1}{2} (X^{\top} Y + Y^{\top} X)
        \end{aligned}

    while exploiting zero or constant columns in the data matrix.
    WARNING: Directly use :meth:`moments_XXXY` if you can. The `covars` function does an additional
    constant-matrix multiplication and does not return the mean.

    Parameters
    ----------
    X : ndarray (T, M)
        Data matrix
    Y : ndarray (T, N)
        Second data matrix
    remove_mean : bool
        True: remove column mean from the data, False: don't remove mean.
    modify_data : bool
        If remove_mean=True, the mean will be removed in the data matrix X,
        without creating an independent copy. This option is faster but might
        lead to surprises because your input array is changed.
    symmetrize : bool
        Computes symmetrized means and moments (see above)
    weights : None or ndarray(T, )
        weights assigned to each trajectory point of X. If None, all data points have weight one.
        If ndarray, each data point is assigned a separate weight.
    sparse_mode : str
        one of:
            * 'dense' : always use dense mode
            * 'sparse' : always use sparse mode if possible
            * 'auto' : automatic
    sparse_tol: float
        Threshold for considering column to be zero in order to save computing
        effort when the data is sparse or almost sparse.
        If max(abs(X[:, i])) < sparse_tol, then row i (and also column i if Y
        is not given) of the covariance matrix will be set to zero. If Y is
        given and max(abs(Y[:, i])) < sparse_tol, then column i of the
        covariance matrix will be set to zero.

    Returns
    -------
    C_XX : ndarray (M, M)
        Covariance matrix of X
    C_XY : ndarray (M, N)
        Covariance matrix of XY

    See also
    --------
    moments_XXXY

    """
    w, sx, sy, Mxx, Mxy = moments_XXXY(X, Y, remove_mean=remove_mean, modify_data=modify_data, weights=weights,
                                       symmetrize=symmetrize, sparse_mode=sparse_mode, sparse_tol=sparse_tol)
    return Mxx / float(w), Mxy / float(w)
