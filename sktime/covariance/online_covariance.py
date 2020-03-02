import itertools
from typing import Optional

import numpy as np
from scipy.linalg import eig

from ..base import Estimator, Model
from ..data.util import timeshifted_split
from ..numeric.eigen import spd_inv_split, sort_by_norm
from .util.running_moments import running_covar as running_covar

__all__ = ['OnlineCovariance', 'OnlineCovarianceModel', 'KoopmanEstimator', 'KoopmanWeights']

__author__ = 'paul, nueske, marscher, clonker'

from ..util import ensure_timeseries_data


class OnlineCovarianceModel(Model):
    r""" A model which in particular carries the estimated covariances, means from a :class:`OnlineCovariance`.
    """
    def __init__(self, cov_00=None, cov_0t=None, cov_tt=None, mean_0=None, mean_t=None, bessels_correction=True):
        r"""
        Initializes a new online covariance model.

        Parameters
        ----------
        cov_00 : (n, n) ndarray, optional, default=None
            The instantaneous covariances if computed (see :attr:`OnlineCovariance.compute_c00`).
        cov_0t : (n, n) ndarray, optional, default=None
            The time-lagged covariances if computed (see :attr:`OnlineCovariance.compute_c0t`).
        cov_tt : (n, n) ndarray, optional, default=None
            The time-lagged instantaneous covariances if computed (see :attr:`OnlineCovariance.compute_ctt`).
        mean_0 : (n,) ndarray, optional, default=None
            The instantaneous means if computed.
        mean_t : (n,) ndarray, optional, default=None
            The time-shifted means if computed.
        bessels_correction : bool, optional, default=True
            Whether Bessel's correction was used during estimation.
        """
        self._cov_00 = cov_00
        self._cov_0t = cov_0t
        self._cov_tt = cov_tt
        self._mean_0 = mean_0
        self._mean_t = mean_t
        self._bessel = bessels_correction

    @property
    def cov_00(self) -> Optional[np.ndarray]:
        r""" The instantaneous covariances.

        :type: (n, n) ndarray or None
        """
        return self._cov_00

    @property
    def cov_0t(self) -> Optional[np.ndarray]:
        r""" The time-shifted covariances.

        :type: (n, n) ndarray or None
        """
        return self._cov_0t

    @property
    def cov_tt(self) -> Optional[np.ndarray]:
        r""" The time-shifted instantaneous covariances.

        :type: (n, n) ndarray or None
        """
        return self._cov_tt

    @property
    def mean_0(self) -> Optional[np.ndarray]:
        r""" The instantaneous means.

        :type: (n,) ndarray or None
        """
        return self._mean_0

    @property
    def mean_t(self) -> Optional[np.ndarray]:
        r""" The time-shifted means.

        :type: (n,) ndarray or None
        """
        return self._mean_t

    @property
    def bessels_correction(self) -> bool:
        r""" Whether Bessel's correction was applied during estimation.

        :type: bool
        """
        return self._bessel


class OnlineCovariance(Estimator):
    r"""Compute (potentially lagged) covariances between data in an online fashion.

    This means computing

    .. math:: \mathrm{cov}[ X_t, Y_t ] = \mathbb{E}[(X_t - \mathbb{E}[X_t])(Y_t - \mathbb{E}[Y_t])],

    where :math:`X_t` and :math:`Y_t` are contiguous blocks of frames from the timeseries data. The estimator
    implements the online algorithm proposed in [1]_, report available in [2]_.

    References
    ----------
    .. [1] Chan, Tony F., Gene Howard Golub, and Randall J. LeVeque. "Updating formulae and a pairwise algorithm
           for computing sample variances." COMPSTAT 1982 5th Symposium held
           at Toulouse 1982. Physica, Heidelberg, 1982.
    .. [2] http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
    """
    def __init__(self, lagtime, compute_c00=True, compute_c0t=False, compute_ctt=False, remove_data_mean=False,
                 reversible=False, bessels_correction=True, sparse_mode='auto', ncov=5, diag_only=False, model=None):
        r""" Initializes a new online covariance estimator.

        Parameters
        ----------
        lagtime : int
            The lagtime, must be :math:`\geq 0` .
        compute_c00 : bool, optional, default=True
            Compute instantaneous correlations over the first part of the data. If :attr:`lagtime` ==0,
            use all of the data.
        compute_c0t : bool, optional, default=False
            Compute lagged correlations. Does not work with :attr:`lagtime` ==0.
        compute_ctt : bool, optional, default=False
            Compute instantaneous correlations over the time-shifted chunks of the data.
            Does not work with :attr:`lagtime` ==0.
        remove_data_mean : bool, optional, default=False
            Subtract the sample mean from the time series (mean-free correlations).
        reversible : bool, optional, default=False
            Symmetrize correlations, i.e., use estimates defined by :math:`\sum_t X_t + Y_t` and second
            moment matrices defined by :math:`X_t^\top X_t + Y_t^\top Y_t` and :math:`Y_t^\top X_t + X_t^\top Y_t` .
        bessels_correction : bool, optional, default=True
            Use Bessel's correction for correlations in order to use an unbiased estimator.
        sparse_mode : str, optional, default='auto'
            one of:
                * 'dense' : always use dense mode
                * 'auto' : automatic
                * 'sparse' : always use sparse mode if possible
        ncov : int, optional, default=5
            Depth of moment storage. Moments computed from each chunk will be combined with Moments of similar
            statistical weight using the pairwise combination algorithm described in [1]_.
        diag_only: bool
            If True, the computation is restricted to the diagonal entries (autocorrelations) only.
        model : OnlineCovarianceModel, optional, default=None
            A model instance with which the estimator can be initialized.
        """

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
        self.diag_only = diag_only
        self.ncov = ncov
        self.sparse_mode = sparse_mode

        self._rc = running_covar(xx=self.compute_c00, xy=self.compute_c0t, yy=self.compute_ctt,
                                 remove_mean=self.remove_data_mean, symmetrize=self.reversible,
                                 sparse_mode=self.sparse_mode, modify_data=False, diag_only=self.diag_only,
                                 nsave=ncov)

    @property
    def ncov(self) -> int:
        r""" The depth of the moments storage.

        :type: int
        """
        return self._ncov

    @ncov.setter
    def ncov(self, value: int):
        self._ncov = value

    @property
    def diag_only(self) -> bool:
        r"""" Whether the computation should be restricted to diagonal entries (autocorrelations) only.

        :type: bool
        """
        return self._diag_only

    @diag_only.setter
    def diag_only(self, value: bool):
        self._diag_only = value

    @property
    def sparse_mode(self) -> str:
        r""" The sparse mode of the estimator. Can be one of 'auto', 'sparse', and 'dense'.

        :type: str
        """
        return self._sparse_mode

    @sparse_mode.setter
    def sparse_mode(self, value: str):
        valid_modes = ('auto', 'sparse', 'dense')
        if self.diag_only and value is not 'dense':
            if value is 'sparse':
                import warnings
                warnings.warn('Computing diagonal entries only is not implemented for sparse mode. Switching to dense '
                              'mode.')
            value = 'dense'
        if value not in valid_modes:
            raise ValueError("Unknown sparse mode: {}, must be one of {}.".format(value, valid_modes))
        self._sparse_mode = value

    @property
    def bessels_correction(self) -> bool:
        r""" Whether to apply Bessel's correction for an unbiased estimator.

        :type: bool
        """
        return self._bessels_correction

    @bessels_correction.setter
    def bessels_correction(self, value: bool):
        self._bessels_correction = value

    @property
    def reversible(self) -> bool:
        r"""Whether to symmetrize correlations.

        :type: bool
        """
        return self._reversible

    @reversible.setter
    def reversible(self, value: bool):
        self._reversible = value

    @property
    def remove_data_mean(self) -> bool:
        r""" Whether to remove the sample mean, i.e., compute mean-free correlations.

        :type: bool
        """
        return self._remove_data_mean

    @remove_data_mean.setter
    def remove_data_mean(self, value: bool):
        self._remove_data_mean = value

    @property
    def compute_c00(self) -> bool:
        r""" Whether to compute instantaneous correlations.

        :type: bool
        """
        return self._compute_c00

    @compute_c00.setter
    def compute_c00(self, value: bool):
        self._compute_c00 = value

    @property
    def compute_c0t(self) -> bool:
        r"""Whether to compute time lagged correlations with a defined :attr:`lagtime`.

        :type: bool
        """
        return self._compute_c0t

    @compute_c0t.setter
    def compute_c0t(self, value: bool):
        self._compute_c0t = value

    @property
    def compute_ctt(self) -> bool:
        r"""Whether to compute instantaneous correlations over the time-shifted chunks of the data.

        :type: bool
        """
        return self._compute_ctt

    @compute_ctt.setter
    def compute_ctt(self, value: bool):
        self._compute_ctt = value

    @property
    def lagtime(self) -> int:
        r"""
        The lagtime of the estimator. This attribute determines how big the temporal difference for timelagged
        autocorrelations are.

        :getter: Yields the currently selected lagtime.
        :setter: Sets a new lagtime, must be :math:`\geq 0`, for :attr:`compute_c0t` and :attr:`compute_ctt` it
                 must be :math:`> 0`.
        :type: int
        """
        return self._lagtime

    @lagtime.setter
    def lagtime(self, value: int):
        self._lagtime = value

    @property
    def is_lagged(self) -> bool:
        r"""
        Determines whether this estimator also computes time-lagged covariances.

        :type: bool
        """
        return self.compute_c0t or self.compute_ctt

    def fit(self, data, lagtime=None, weights=None, n_splits=None, column_selection=None):
        r"""Computes covariances for the input data and produces a new model. If an existing model should be updated,
        call :meth:`partial_fit`.

        Parameters
        ----------
        data : array_like or list of array_like
            The input data. If it is a list of trajectories, all elements of the list must have the same dtype and
            size in the second dimension, i.e., the elements of :code:`[x.shape[1] for x in data]` must all be equal.
        lagtime : int, optional, default=None
            Override for :attr:`lagtime`.
        weights : array_like or list of array_like, optional, default=None
            Optional weights for the input data. Must be of matching shape.
        n_splits : int, optional, default=None
            The number of times the data is split uniformly when performing the covariance estimation. If no value
            is given, it estimates the number of splits by :code:`min(trajectory_lengths) // 100` if the shortest
            trajectory contains at least 1000 frames. Otherwise, the number of splits is set to one.
        column_selection : ndarray, optional, default=None
            Columns of the trajectories to restrict estimation to. Must be given in terms of an index array.

        Returns
        -------
        self : OnlineCovariance
            Reference to self.
        """
        data = ensure_timeseries_data(data)

        self._model = None
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
        """ Incrementally update the estimates. For a detailed description of the parameters, see :meth:`fit` with
        the exception of the :code:`data` argument, it must be a ndarray and cannot be a list of ndarray."""
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
        if self._model is None:
            self._model = OnlineCovarianceModel()
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

    def __init__(self, lagtime, epsilon=1e-6, ncov=int(2**10000)):
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

        self._model = KoopmanWeights(u=u_input[:-1], u_const=u_input[-1])

        return self._model

    @property
    def lagtime(self):
        return self._cov.lagtime

    @lagtime.setter
    def lagtime(self, value):
        self._cov.lagtime = value
