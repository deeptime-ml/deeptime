import numbers

import numpy as np


def ensure_ndarray(arr, shape: tuple = None, ndim: int = None, dtype=None, size=None) -> np.ndarray:
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


def ensure_dtraj_list(dtrajs):
    """Makes sure that dtrajs is a list of discrete trajectories (array of int)"""
    if len(dtrajs) > 0 and isinstance(dtrajs[0], numbers.Integral):
        return [ensure_ndarray(dtrajs, dtype=np.int32)]
    return [ensure_ndarray(t, dtype=np.int32) for t in dtrajs]


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
