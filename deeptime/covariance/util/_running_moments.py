import numbers
import warnings

import numpy as np

from ._moments import moments_XX, moments_XXXY, moments_block

__author__ = 'noe'


class Moments:

    def __init__(self, w, sx, sy, Mxy):
        r"""
        Parameters
        ----------
        w : float
            statistical weight.
                w = \sum_t w_t
            In most cases, :math:`w_t=1`, and then w is just the number of samples that went into s1, S2.
        s : ndarray(n,)
            sum over samples:
            .. math:
                s = \sum_t w_t x_t
        M : ndarray(n, n)
            .. math:
                M = (X-s)^T (X-s)
        """
        self.w = float(w)
        self.sx = sx
        self.sy = sy
        self.Mxy = Mxy

    def copy(self):
        return Moments(self.w, self.sx.copy(), self.sy.copy(), self.Mxy.copy())

    def combine(self, other, mean_free=False):
        """ According to `this report <http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf>`__.
        """
        w1 = self.w
        w2 = other.w
        w = w1 + w2
        q = w2 / w1
        dsx = q * self.sx - other.sx
        dsy = q * self.sy - other.sy
        # update
        self.w = w1 + w2
        self.sx = self.sx + other.sx
        self.sy = self.sy + other.sy
        #
        if mean_free:
            if len(self.Mxy.shape) == 1:  # diagonal only
                d = dsx*dsy
            else:
                d = np.outer(dsx, dsy)
            self.Mxy += other.Mxy + (w1 / (w2 * w)) * d
        else:
            self.Mxy += other.Mxy
        return self

    @property
    def mean_x(self):
        return self.sx / self.w

    @property
    def mean_y(self):
        return self.sy / self.w

    def covar(self, bessels_correction=True):
        """
        Return covariance matrix:

        Parameters:
        -----------
        bessels_correction : bool, optional, default=True
            Use Bessel's correction in order to
            obtain an unbiased estimator of sample covariances.

        """
        if bessels_correction:
            return self.Mxy / (self.w - 1)
        else:
            return self.Mxy / self.w


class MomentsStorage:

    def __init__(self, nsave, remove_mean=False, rtol=1.5):
        """
        Parameters
        ----------
        rtol : float
            To decide when to merge two Moments. Ideally I'd like to merge two
            Moments when they have equal weights (i.e. equally many data points
            went into them). If we always add data chunks with equal weights,
            this can be achieved by using a binary tree, i.e. let M1 be the
            moment estimates from one chunk. Two of them are added to M2, Two
            M2 are added to M4, and so on. This way you need to store log2
            (n_chunks) number of Moment estimates.
            In practice you might get data in chunks of unequal length or weight.
            Therefore we need some heuristic when two Moment estimates should get
            merged. This is the role of rtol.

        """
        self.nsave = nsave
        self.storage = []
        self.rtol = rtol
        self.remove_mean = remove_mean

    def _can_merge_tail(self):
        """ Checks if the two last list elements can be merged
        """
        if len(self.storage) < 2:
            return False
        return self.storage[-2].w <= self.storage[-1].w * self.rtol

    def store(self, moments):
        """ Store object X with weight w
        """
        if len(self.storage) == self.nsave:  # merge if we must
            self.storage[-1].combine(moments, mean_free=self.remove_mean)
        else:  # append otherwise
            self.storage.append(moments)
        # merge if possible
        while self._can_merge_tail():
            M = self.storage.pop()
            self.storage[-1].combine(M, mean_free=self.remove_mean)

    @property
    def moments(self):
        """
        """
        # collapse storage if necessary
        while len(self.storage) > 1:
            M = self.storage.pop()
            self.storage[-1].combine(M, mean_free=self.remove_mean)
        return self.storage[0]

    def clear(self):
        self.storage.clear()


class RunningCovar:
    """ Running covariance estimator

    Estimator object that can be fed chunks of X and Y data, and
    that can generate on-the-fly estimates of mean, covariance, running sum
    and second moment matrix.

    Parameters
    ----------
    compute_XX : bool
        Estimate the covariance of X
    compute_XY : bool
        Estimate the cross-covariance of X and Y
    compute_YY : bool
        Estimate the covariance of Y
    remove_mean : bool
        Remove the data mean in the covariance estimation
    symmetrize : bool
        Use symmetric estimates with sum defined by sum_t x_t + y_t and
        second moment matrices defined by X'X + Y'Y and Y'X + X'Y.
    modify_data : bool
        If remove_mean=True, the mean will be removed in the input data,
        without creating an independent copy. This option is faster but should
        only be selected if the input data is not used elsewhere.
    sparse_mode : str
        one of:
            * 'dense' : always use dense mode
            * 'sparse' : always use sparse mode if possible
            * 'auto' : automatic
    diag_only: bool
        If True, the computation is restricted to the diagonal entries (autocorrelations) only.
    nsave : int
        Depth of Moment storage. Moments computed from each chunk will be
        combined with Moments of similar statistical weight using the pairwise
        combination algorithm described in :footcite:`chan1982updating`.

    References
    ----------
    .. footbibliography::
    """

    # to get the Y mean, but this is currently not stored.
    def __init__(self, compute_XX=True, compute_XY=False, compute_YY=False,
                 remove_mean=False, symmetrize=False, sparse_mode='auto', modify_data=False,
                 diag_only=False, nsave=5):
        # check input
        if not compute_XX and not compute_XY:
            raise ValueError('One of compute_XX or compute_XY must be True.')
        if symmetrize and compute_YY:
            raise ValueError('Combining compute_YY and symmetrize=True is meaningless.')
        if symmetrize and not compute_XY:
            warnings.warn('symmetrize=True has no effect with compute_XY=False.')
        if diag_only and sparse_mode != 'dense':
            if sparse_mode == 'sparse':
                warnings.warn('Computing diagonal entries only is not implemented for sparse mode. '
                              'Switching to dense mode.')
            sparse_mode = 'dense'
        # storage
        self.compute_XX = compute_XX
        self.compute_XY = compute_XY
        self.compute_YY = compute_YY

        self.storage_XX = MomentsStorage(nsave, remove_mean=remove_mean)
        self.storage_XY = MomentsStorage(nsave, remove_mean=remove_mean)
        self.storage_YY = MomentsStorage(nsave, remove_mean=remove_mean)
        # symmetry
        self.remove_mean = remove_mean
        self.symmetrize = symmetrize
        # flags
        self.sparse_mode = sparse_mode
        self.modify_data = modify_data
        # whether to compute only matrix diagonals
        self.diag_only = diag_only

    def add(self, X, Y=None, weights=None, column_selection=None):
        """
        Add trajectory to estimate.

        Parameters
        ----------
        X : ndarray(T, N)
            array of N time series.
        Y : ndarray(T, N)
            array of N time series, usually time shifted version of X.
        weights : None or float or ndarray(T, ):
            weights assigned to each trajectory point. If None, all data points have weight one. If float,
            the same weight will be given to all data points. If ndarray, each data point is assigned a separate
            weight.
        column_selection : ndarray((d,), dtype=int), optional, default=None
            Optional column subselection.
        """

        # check input
        X = X if X.ndim >= 2 else X[..., np.newaxis]
        if Y is not None:
            Y = Y if Y.ndim >= 2 else Y[..., np.newaxis]
        T = X.shape[0]
        if Y is not None:
            assert Y.shape[0] == T, 'X and Y must have equal length'
        if column_selection is not None and self.diag_only:
            raise ValueError('Computing only parts of the diagonal is not supported.')

        # Weights cannot be used for compute_YY:
        if weights is not None and self.compute_YY:
            raise ValueError('Use of weights is not implemented for compute_YY==True')
        if weights is not None:
            # Convert to array of length T if weights is a single number:
            if isinstance(weights, numbers.Real):
                weights = weights * np.ones(T, dtype=float)
            # Check appropriate length if weights is an array:
            elif isinstance(weights, np.ndarray):
                if len(weights) != T:
                    raise ValueError(f'Weights and X must have equal length. '
                                     f'Was {len(weights)} and {len(X)}, respectively.')
            else:
                raise TypeError(f'Weights is of type {type(weights)}, must be a number or ndarray.')
        # estimate and add to storage
        if self.compute_XX and not self.compute_XY and not self.compute_YY:
            w, s_X, C_XX = moments_XX(X, remove_mean=self.remove_mean, weights=weights, sparse_mode=self.sparse_mode,
                                      modify_data=self.modify_data, column_selection=column_selection,
                                      diag_only=self.diag_only)
            if column_selection is not None:
                s_Xk = s_X[column_selection]
            else:
                s_Xk = s_X
            self.storage_XX.store(Moments(w, s_X, s_Xk, C_XX))
        elif self.compute_XX and self.compute_XY and not self.compute_YY:
            assert Y is not None
            w, s_X, s_Y, C_XX, C_XY = moments_XXXY(X, Y, remove_mean=self.remove_mean, symmetrize=self.symmetrize,
                                                   weights=weights, sparse_mode=self.sparse_mode, modify_data=self.modify_data,
                                                   column_selection=column_selection, diag_only=self.diag_only)
            # make copy in order to get independently mergeable moments
            if column_selection is not None:
                s_Xk = s_X[column_selection]
                s_Yk = s_Y[column_selection]
            else:
                s_Xk = s_X
                s_Yk = s_Y
            self.storage_XX.store(Moments(w, s_X, s_Xk, C_XX))
            self.storage_XY.store(Moments(w, s_X, s_Yk, C_XY))
        else:  # compute block
            assert Y is not None
            assert not self.symmetrize
            w, s, C = moments_block(X, Y, remove_mean=self.remove_mean,
                                    sparse_mode=self.sparse_mode, modify_data=self.modify_data,
                                    column_selection=column_selection, diag_only=self.diag_only)
            # make copy in order to get independently mergeable moments
            if column_selection is not None:
                s0k = s[0][column_selection]
                s1k = s[1][column_selection]
            else:
                s0k = s[0]
                s1k = s[1]
            if self.compute_XX:
                self.storage_XX.store(Moments(w, s[0], s0k, C[0][0]))
            if self.compute_XY:
                self.storage_XY.store(Moments(w, s[0], s1k, C[0][1]))
            self.storage_YY.store(Moments(w, s[1], s1k, C[1][1]))

    def sum_X(self):
        if self.compute_XX:
            return self.storage_XX.moments.sx
        elif self.compute_XY:
            return self.storage_XY.moments.sx
        else:
            raise RuntimeError('sum_X is not available')

    def sum_Y(self):
        if self.compute_XY:
            return self.storage_XY.moments.sy
        elif self.compute_YY:
            return self.storage_YY.moments.sy
        else:
            raise RuntimeError('sum_Y is not available')

    def mean_X(self):
        if self.compute_XX:
            return self.storage_XX.moments.mean_x
        elif self.compute_XY:
            return self.storage_XY.moments.mean_y
        else:
            raise RuntimeError('mean_X is not available')

    def mean_Y(self):
        if self.compute_XY:
            return self.storage_XY.moments.mean_y
        elif self.compute_YY:
            return self.storage_YY.moments.mean_y
        else:
            raise RuntimeError('mean_Y is not available')

    def weight_XX(self):
        return self.storage_XX.moments.w

    def weight_XY(self):
        return self.storage_XY.moments.w

    def weight_YY(self):
        return self.storage_YY.moments.w

    def moments_XX(self):
        return self.storage_XX.moments.Mxy

    def moments_XY(self):
        return self.storage_XY.moments.Mxy

    def moments_YY(self):
        return self.storage_YY.moments.Mxy

    def cov_XX(self, bessel=True):
        return self.storage_XX.moments.covar(bessels_correction=bessel)

    def cov_XY(self, bessel=True):
        return self.storage_XY.moments.covar(bessels_correction=bessel)

    def cov_YY(self, bessel):
        return self.storage_YY.moments.covar(bessels_correction=bessel)

    def clear(self):
        self.storage_XX.clear()
        self.storage_XY.clear()
        self.storage_YY.clear()


def running_covar(xx=True, xy=False, yy=False, remove_mean=False, symmetrize=False, sparse_mode='auto',
                  modify_data=False, diag_only=False, nsave=5):
    """ Returns a running covariance estimator

    Returns an estimator object that can be fed chunks of X and Y data, and
    that can generate on-the-fly estimates of mean, covariance, running sum
    and second moment matrix.

    Parameters
    ----------
    xx : bool
        Estimate the covariance of X
    xy : bool
        Estimate the cross-covariance of X and Y
    yy : bool
        Estimate the covariance of Y
    remove_mean : bool
        Remove the data mean in the covariance estimation
    symmetrize : bool
        Use symmetric estimates with sum defined by sum_t x_t + y_t and
        second moment matrices defined by X'X + Y'Y and Y'X + X'Y.
    modify_data : bool
        If remove_mean=True, the mean will be removed in the input data,
        without creating an independent copy. This option is faster but should
        only be selected if the input data is not used elsewhere.
    sparse_mode : str
        one of:
            * 'dense' : always use dense mode
            * 'sparse' : always use sparse mode if possible
            * 'auto' : automatic
    diag_only: bool
        If True, the computation is restricted to the diagonal entries (autocorrelations) only.
    nsave : int
        Depth of Moment storage. Moments computed from each chunk will be
        combined with Moments of similar statistical weight using the pairwise
        combination algorithm described in :footcite:`chan1982updating`.

    References
    ----------
    .. footbibliography::
    """
    return RunningCovar(compute_XX=xx, compute_XY=xy, compute_YY=yy, sparse_mode=sparse_mode, modify_data=modify_data,
                        remove_mean=remove_mean, symmetrize=symmetrize,
                        diag_only=diag_only, nsave=nsave)
