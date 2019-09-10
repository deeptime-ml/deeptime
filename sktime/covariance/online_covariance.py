import itertools

import numpy as np
from scipy.linalg import eig

from sktime.base import Estimator, Model
from sktime.data.util import timeshifted_split
from sktime.numeric.eigen import spd_inv_split, sort_by_norm
from .util.running_moments import running_covar as running_covar

__all__ = ['OnlineCovariance']

__author__ = 'paul, nueske, marscher, clonker'


def ensure_timeseries_data(input_data):
    if not isinstance(input_data, list):
        if not isinstance(input_data, np.ndarray):
            raise ValueError('input data can not be converted to a list of arrays')
        elif isinstance(input_data, np.ndarray):
            if input_data.dtype not in (np.float32, np.float64):
                raise ValueError('only float and double dtype is supported')
            return [input_data]
    else:
        for i, x in enumerate(input_data):
            if not isinstance(x, np.ndarray):
                raise ValueError(f'element {i} of given input data list is not an array.')
            else:
                if x.dtype not in (np.float32, np.float64):
                    raise ValueError('only float and double dtype is supported')
                input_data[i] = x
    return input_data


class OnlineCovarianceModel(Model):
    def __init__(self, cov_00=None, cov_0t=None, cov_tt=None, mean_0=None, mean_t=None, bessels_correction=True):
        self._cov_00 = cov_00
        self._cov_0t = cov_0t
        self._cov_tt = cov_tt
        self._mean_0 = mean_0
        self._mean_t = mean_t
        self._bessel = bessels_correction

    @property
    def cov_00(self):
        return self._cov_00

    @property
    def cov_0t(self):
        return self._cov_0t

    @property
    def cov_tt(self):
        return self._cov_tt

    @property
    def mean_0(self):
        return self._mean_0

    @property
    def mean_t(self):
        return self._mean_t

    @property
    def bessels_correction(self):
        return self._bessel


class OnlineCovariance(Estimator):
    r"""Compute (potentially lagged) covariances between data in an online fashion.

    Parameters
    ----------
    compute_c00 : bool, optional, default=True
        compute instantaneous correlations over the first part of the data. If lag==0, use all of the data.
        Makes the C00_ attribute available.
    compute_c0t : bool, optional, default=False
        compute lagged correlations. Does not work with lag==0.
        Makes the C0t_ attribute available.
    compute_ctt : bool, optional, default=False
        compute instantaneous correlations over the time-shifted chunks of the data. Does not work with lag==0.
        Makes the Ctt_ attribute available.
    remove_data_mean : bool, optional, default=False
        subtract the sample mean from the time series (mean-free correlations).
    reversible : bool, optional, default=False
        symmetrize correlations.
    bessel : bool, optional, default=True
        use Bessel's correction for correlations in order to use an unbiased estimator
    sparse_mode : str, optional, default='auto'
        one of:
            * 'dense' : always use dense mode
            * 'auto' : automatic
            * 'sparse' : always use sparse mode if possible
    diag_only: bool
        If True, the computation is restricted to the diagonal entries (autocorrelations) only.
    """
    def __init__(self, lagtime=None, compute_c00=True, compute_c0t=False, compute_ctt=False, remove_data_mean=False,
                 reversible=False, bessels_correction=True, sparse_mode='auto', ncov=5, diag_only=False, model=None):

        if diag_only and sparse_mode is not 'dense':
            if sparse_mode is 'sparse':
                import warnings
                warnings.warn('Computing diagonal entries only is not implemented for sparse mode. Switching to dense '
                              'mode.')
            sparse_mode = 'dense'

        if (compute_c0t or compute_ctt) and lagtime is None:
            raise ValueError('lagtime parameter mandatory due to requested covariance matrices.')

        super(OnlineCovariance, self).__init__(model=model)
        self.lagtime = lagtime
        self.compute_c00 = compute_c00
        self.compute_c0t = compute_c0t
        self.compute_ctt = compute_ctt
        self.remove_data_mean = remove_data_mean
        self.reversible = reversible
        self.bessels_correction = bessels_correction
        self.sparse_mode = sparse_mode
        self.ncov = ncov
        self.diag_only = diag_only

        self._rc = running_covar(xx=self.compute_c00, xy=self.compute_c0t, yy=self.compute_ctt,
                                 remove_mean=self.remove_data_mean, symmetrize=self.reversible,
                                 sparse_mode=self.sparse_mode, modify_data=False, diag_only=self.diag_only,
                                 nsave=ncov)

    def _create_model(self) -> OnlineCovarianceModel:
        return OnlineCovarianceModel()

    @property
    def is_lagged(self) -> bool:
        return self.compute_c0t or self.compute_ctt

    def fit(self, data, lagtime=None, weights=None, n_splits=None, column_selection=None):
        """
         column_selection: ndarray(k, dtype=int) or None
         Indices of those columns that are to be computed. If None, all columns are computed.
        :param data: list of sequences (n elements)
        :param weights: list of weight arrays (n elements) or array (shape
        :param n_splits:
        :param column_selection:
        :return:
        """
        # TODO: constistent dtype
        data = ensure_timeseries_data(data)

        self._rc.clear()

        if n_splits is None:
            dlen = min(len(d) for d in data)
            n_splits = int(dlen // 100 if dlen >= 1e4 else 1)

        if lagtime is None:
            lagtime = self.lagtime
        else:
            self.lagtime = lagtime
        assert lagtime is not None

        lazy_weights = False
        wsplit = itertools.repeat(None)

        if weights is not None:
            if hasattr(weights, 'weights'):
                lazy_weights = True
            elif len(np.atleast_1d(weights)) != len(data[0]):
                raise ValueError(
                    "Weights have incompatible shape "
                    f"(#weights={len(weights) if weights is not None else None} != {len(data[0])}=#frames.")
            elif isinstance(weights, np.ndarray):
                wsplit = np.array_split(weights, n_splits)

        if self.is_lagged:
            for (x, y), w in zip(timeshifted_split(data, lagtime=lagtime, n_splits=n_splits), wsplit):
                if lazy_weights:
                    w = weights.weights(x)
                # weights can weights be shorter than actual data
                if isinstance(w, np.ndarray):
                    w = w[:len(x)]
                self.partial_fit((x, y), weights=w, column_selection=column_selection)
        else:
            for x in data:
                self.partial_fit(x, weights=weights, column_selection=column_selection)

        return self

    def partial_fit(self, data, weights=None, column_selection=None):
        """ incrementally update the estimates

        Parameters
        ----------
        data: numpy array or tuple of two numpy arrays in case of time-lagged estimation
        weights: the weights as 1d array with length len(data)
        column_selection: the column selection
        """
        if self.is_lagged:
            x, y = data
        else:
            x, y = data, None
        # TODO: types, shapes checking!
        try:
            self._rc.add(x, y, column_selection=column_selection, weights=weights)
        except MemoryError:
            raise MemoryError('Covariance matrix does not fit into memory. '
                              'Input is too high-dimensional ({} dimensions). '.format(x.shape[1]))
        return self

    def fetch_model(self) -> OnlineCovarianceModel:
        cov_00 = cov_tt = cov_0t = mean_0 = mean_t = None
        if self.compute_c0t:
            cov_0t = self._rc.cov_XY(self.bessels_correction)
        if self.compute_ctt:
            cov_tt = self._rc.cov_YY(self.bessels_correction)
        if self.compute_c00:
            cov_00 = self._rc.cov_XX(self.bessels_correction)

        if self.compute_c00 or self.compute_c0t:
            mean_0 = self._rc.mean_X()
        if self.compute_ctt or self.compute_c0t:
            mean_t = self._rc.mean_Y()
        self._model.__init__(cov_00=cov_00, cov_0t=cov_0t, cov_tt=cov_tt, mean_0=mean_0, mean_t=mean_t,
                             bessels_correction=self.bessels_correction)
        return self._model


class KoopmanWeights(Model):

    def __init__(self, u=None, u_const=0):
        self.u = u
        self.u_const = u_const

    def weights(self, X):
        assert self.u is not None and self.u_const is not None
        return X.dot(self.u) + self.u_const


class KoopmanEstimator(Estimator):

    def __init__(self, lagtime, epsilon=1e-6, ncov=float('inf')):
        super(KoopmanEstimator, self).__init__()
        self.epsilon = epsilon
        self._cov = OnlineCovariance(lagtime=lagtime, compute_c00=True, compute_c0t=True, remove_data_mean=True, reversible=False,
                                     bessels_correction=False, ncov=ncov)

    def fit(self, data, y=None, lagtime=None):
        self._cov.fit(data, lagtime=lagtime)
        self.fetch_model()  # pre-compute Koopman operator
        return self

    def partial_fit(self, data):
        self._cov.partial_fit(data)
        return self

    def _create_model(self) -> KoopmanWeights:
        return KoopmanWeights()

    @staticmethod
    def _compute_u(K):
        """
        Estimate an approximation of the ratio of stationary over empirical distribution from the basis.
        Parameters:
        -----------
        K0, ndarray(M+1, M+1),
            time-lagged correlation matrix for the whitened and padded data set.
        Returns:
        --------
        u : ndarray(M,)
            coefficients of the ratio stationary / empirical dist. from the whitened and expanded basis.
        """
        M = K.shape[0] - 1
        # Compute right and left eigenvectors:
        l, U = eig(K.T)
        l, U = sort_by_norm(l, U)
        # Extract the eigenvector for eigenvalue one and normalize:
        u = np.real(U[:, 0])
        v = np.zeros(M + 1)
        v[M] = 1.0
        u = u / np.dot(u, v)
        return u

    def fetch_model(self) -> KoopmanWeights:
        cov = self._cov.fetch_model()

        R = spd_inv_split(cov.cov_00, epsilon=self.epsilon, canonical_signs=True)
        # Set the new correlation matrix:
        M = R.shape[1]
        K = np.dot(R.T, np.dot(cov.cov_0t, R))
        K = np.vstack((K, np.dot((cov.mean_t - cov.mean_0), R)))
        ex1 = np.zeros((M + 1, 1))
        ex1[M, 0] = 1.0
        K = np.hstack((K, ex1))

        u = self._compute_u(K)
        N = R.shape[0]
        u_input = np.zeros(N+1)
        u_input[0:N] = R.dot(u[0:-1])  # in input basis
        u_input[N] = u[-1] - cov.mean_0.dot(R.dot(u[0:-1]))

        self._model.u = u_input[:-1]
        self._model.u_const = u_input[-1]
        return self._model

    @property
    def lagtime(self):
        return self._cov.lagtime

    @lagtime.setter
    def lagtime(self, value):
        self._cov.lagtime = value
