import itertools
from typing import Optional

import numpy as np
from scipy.linalg import eig

from ..base import Estimator, Model, Transformer, EstimatorTransformer
from ..basis import Observable
from ..util.data import timeshifted_split
from ..numeric import spd_inv_split, sort_eigs, spd_inv_sqrt
from .util import running_covar
from ..util.types import ensure_timeseries_data

__all__ = ['Covariance', 'CovarianceModel', 'KoopmanWeightingEstimator', 'KoopmanWeightingModel']

__author__ = 'paul, nueske, marscher, clonker'


class WhiteningTransform(Observable):
    r""" Transformation of data into a whitened space. It is assumed that for a covariance matrix :math:`C` the
    square-root inverse :math:`C^{-1/2}` was already computed. Optionally a mean :math:`\mu` can be provided.
    This yields the transformation

    .. math::
        y = C^{-1/2}(x-\mu).

    Parameters
    ----------
    sqrt_inv_cov : (n, k) ndarray
        Square-root inverse of covariance matrix.
    mean : (n, ) ndarray, optional, default=None
        The mean if it should be subtracted.
    dim : int, optional, default=None
        Additional restriction in the dimension, removes all but the first `dim` components of the output.

    See Also
    --------
    deeptime.numeric.spd_inv_sqrt : Method to obtain (regularized) inverses of covariance matrices.
    """

    def __init__(self, sqrt_inv_cov: np.ndarray, mean: Optional[np.ndarray] = None, dim: Optional[int] = None):
        self.sqrt_inv_cov = sqrt_inv_cov
        self.mean = mean
        self.dim = dim

    def _evaluate(self, x: np.ndarray):
        if self.mean is not None:
            x = x - self.mean
        return x @ self.sqrt_inv_cov[..., :self.dim]


class CovarianceModel(Model):
    r""" A model which in particular carries the estimated covariances, means from a :class:`Covariance`.

    Parameters
    ----------
    cov_00 : (n, n) ndarray, optional, default=None
        The instantaneous covariances if computed (see :attr:`Covariance.compute_c00`).
    cov_0t : (n, n) ndarray, optional, default=None
        The time-lagged covariances if computed (see :attr:`Covariance.compute_c0t`).
    cov_tt : (n, n) ndarray, optional, default=None
        The time-lagged instantaneous covariances if computed (see :attr:`Covariance.compute_ctt`).
    mean_0 : (n,) ndarray, optional, default=None
        The instantaneous means if computed.
    mean_t : (n,) ndarray, optional, default=None
        The time-shifted means if computed.
    bessels_correction : bool, optional, default=True
        Whether Bessel's correction was used during estimation.
    lagtime : int, default=None
        The lagtime that was used during estimation.
    data_mean_removed : bool, default=False
        Whether the data mean was removed. This can have an influence on the effective VAMP score.
    """

    def __init__(self, cov_00: Optional[np.ndarray] = None, cov_0t: Optional[np.ndarray] = None,
                 cov_tt: Optional[np.ndarray] = None, mean_0: Optional[np.ndarray] = None,
                 mean_t: Optional[np.ndarray] = None, bessels_correction: bool = True,
                 symmetrized: bool = False, lagtime: Optional[int] = None, data_mean_removed: bool = False):
        super(CovarianceModel, self).__init__()
        self._cov_00 = cov_00
        self._cov_0t = cov_0t
        self._cov_tt = cov_tt
        self._mean_0 = mean_0
        self._mean_t = mean_t
        self._bessel = bessels_correction
        self._lagtime = lagtime
        self._symmetrized = symmetrized
        self._data_mean_removed = data_mean_removed

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

    @property
    def lagtime(self) -> Optional[int]:
        r""" The lagtime at which estimation was performed.

        :type: int or None
        """
        return self._lagtime

    @property
    def symmetrized(self) -> bool:
        r""" Whether correlations and second moments are symmetrized in time.

        :type: bool
        """
        return self._symmetrized

    def whiten(self, data: np.ndarray, epsilon=1e-10, method='QR') -> np.ndarray:
        r"""Whiten a (T, N)-shaped chunk of data by transforming it into the PCA basis. In case of rank deficiency
        this reduces the dimension.

        Parameters
        ----------
        data : (T,N) ndarray
            The data to be whitened.
        epsilon : float, optional, default=1e-10
            Truncation parameter. See :meth:`deeptime.numeric.spd_inv_sqrt`.
        method : str, optional, default='QR'
            Decomposition method. See :meth:`deeptime.numeric.spd_inv_sqrt`.

        Returns
        -------
        whitened_data : (T, n) ndarray
            Whitened data.
        """
        assert self.cov_00 is not None and self.mean_0 is not None
        projection = np.atleast_2d(spd_inv_sqrt(self.cov_00, epsilon=epsilon, method=method))
        whitened_data = (data - self.mean_0[None, ...]) @ projection.T
        return whitened_data

    @property
    def data_mean_removed(self) -> bool:
        r"""Whether the data mean was removed.

        :type: bool
        """
        return self._data_mean_removed


class Covariance(Estimator):
    r"""Compute (potentially lagged) covariances between data in an online fashion.

    This means computing

    .. math:: \mathrm{cov}[ X_t, Y_t ] = \mathbb{E}[(X_t - \mathbb{E}[X_t])(Y_t - \mathbb{E}[Y_t])],

    where :math:`X_t` and :math:`Y_t` are contiguous blocks of frames from the timeseries data. The estimator
    implements the online algorithm proposed in :footcite:`chan1982updating`, report available
    `here <http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf>`__.

    Parameters
    ----------
    lagtime : int, default=0
        The lagtime, must be :math:`\geq 0` .
    compute_c00 : bool, optional, default=True
        Compute instantaneous correlations over the first part of the data. If :attr:`lagtime` ==0,
        use all of the data.
    compute_c0t : bool, optional, default=False
        Compute lagged correlations. Does not work with :attr:`lagtime` ==0.
    compute_ctt : bool, optional, default=False
        Compute instantaneous covariance over the time-shifted chunks of the data.
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
        statistical weight using the pairwise combination algorithm
        described in :footcite:`chan1982updating`.
    diag_only: bool
        If True, the computation is restricted to the diagonal entries (autocorrelations) only.
    model : CovarianceModel, optional, default=None
        A model instance with which the estimator can be initialized.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, lagtime: int = 0, compute_c00: bool = True, compute_c0t: bool = False,
                 compute_ctt: bool = False, remove_data_mean: bool = False, reversible: bool = False,
                 bessels_correction: bool = True, sparse_mode: str = 'auto', ncov: int = 5, diag_only: bool = False,
                 model=None):
        super(Covariance, self).__init__(model=model)
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
        self._dirty = False  # Flag which gets set to true once fit/partial_fit was called and the model was not fetched

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
        if self.diag_only and value != 'dense':
            if value == 'sparse':
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
        if self.compute_ctt and value:
            raise ValueError("Computing covariances reversibly and also computing cov_tt is meaningless, as then "
                             "cov_tt = cov_00. Please set compute_ctt to False prior setting reversible to True "
                             "if symmetrized covariances are desired.")
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
    def lagtime(self, value: Optional[int]):
        if value is not None and value < 0:
            raise ValueError("Negative lagtime are not supported.")
        self._lagtime = value

    @property
    def is_lagged(self) -> bool:
        r""" Determines whether this estimator also computes time-lagged covariances.

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
        weights : array_like or list of array_like or object, optional, default=None

            * Optional weights for the input data. Must be of matching shape.

            * Can also be another arbitrary object. The only requirement is that weights possesses a method weights(X),
              that accepts a trajectory X (np.ndarray(T, n)) and returns a vector of
              re-weighting factors (np.ndarray(T,)). See, e.g.,

              * :class:`KoopmanEstimator <deeptime.covariance.KoopmanEstimator>`

        n_splits : int, optional, default=None
            The number of times the data is split uniformly when performing the covariance estimation. If no value
            is given, it estimates the number of splits by :code:`min(trajectory_lengths) // 100` if the shortest
            trajectory contains at least 1000 frames. Otherwise, the number of splits is set to one.
        column_selection : ndarray, optional, default=None
            Columns of the trajectories to restrict estimation to. Must be given in terms of an index array.

        Returns
        -------
        self : Covariance
            Reference to self.
        """
        data = ensure_timeseries_data(data)

        self._model = None
        self._rc.clear()

        n_splits = 1 if n_splits is None else n_splits

        if lagtime is None:
            lagtime = self.lagtime
        else:
            self.lagtime = lagtime
        assert lagtime is not None

        lazy_weights = False
        wsplit = itertools.repeat((None, None))  # tuple of None that can be unpacked into two Nones

        if weights is not None:
            if hasattr(weights, 'weights'):
                lazy_weights = True
            elif len(np.atleast_1d(weights)) != len(data[0]):
                raise ValueError(
                    "Weights have incompatible shape "
                    f"(#weights={len(weights) if weights is not None else None} != {len(data[0])}=#frames.")
            elif isinstance(weights, np.ndarray):
                wsplit = timeshifted_split(weights, lagtime=lagtime, n_splits=n_splits)

        if self.is_lagged:
            t = 0
            for (x, y), (w, _) in zip(timeshifted_split(data, lagtime=lagtime, n_splits=n_splits), wsplit):
                if lazy_weights:
                    w = weights.weights(x)
                # weights can be longer than actual data
                if isinstance(w, np.ndarray):
                    w = w[:len(x)]
                self.partial_fit((x, y), weights=w, column_selection=column_selection)
                t += len(x)
        else:
            for x in data:
                self.partial_fit(x, weights=weights, column_selection=column_selection)

        return self

    def partial_fit(self, data, weights=None, column_selection=None):
        """ Incrementally update the estimates. For a detailed description of the parameters, see :meth:`fit` with
        the exception of the :code:`data` argument, it must be a ndarray and cannot be a list of ndarray."""
        self._dirty = True
        if self.is_lagged:
            x, y = data
        else:
            x, y = data, None
        if weights is not None and hasattr(weights, 'weights'):
            weights = weights(x)
        try:
            self._rc.add(x, y, column_selection=column_selection, weights=weights)
        except MemoryError:
            raise MemoryError(f'Covariance matrix does not fit into memory. '
                              f'Input is too high-dimensional ({x.shape[1]} dimensions).')
        return self

    def fetch_model(self) -> CovarianceModel:
        r""" Finalizes the covariance computation by aggregating all moment storages.

        Returns
        -------
        model : CovarianceModel
            The covariance model.
        """
        if self._dirty:
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
            if self.reversible:
                cov_tt = cov_00
            self._model = CovarianceModel(cov_00=cov_00, cov_0t=cov_0t, cov_tt=cov_tt, mean_0=mean_0, mean_t=mean_t,
                                          bessels_correction=self.bessels_correction, lagtime=self.lagtime,
                                          symmetrized=self.reversible, data_mean_removed=self.remove_data_mean)
            self._dirty = False  # catches multiple calls to fetch_model even though it hasn't changed
        return self._model


class KoopmanWeightingModel(Model, Transformer):
    r""" A model which contains the Koopman operator in a modified basis `(PC|1)` and can transform data into Koopman
    weights.

    Weights are computed according to :footcite:`wu2017variational`.

    Parameters
    ----------
    u : ndarray
        Reweighting vector in input basis
    u_const : float
        Constant offset for reweighting in input basis.
    koopman_operator : ndarray
        Koopman operator in modified basis.
    whitening_transformation : ndarray, optional, default=None
        Whitening transformation.
    covariances : CovarianceModel, optional, default=None
        Estimated covariances.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, u, u_const, koopman_operator, whitening_transformation=None, covariances=None):
        super().__init__()
        self._u = u
        self._u_const = u_const
        self._koopman_operator = koopman_operator
        self._whitening_transformation = whitening_transformation
        self._covariances = covariances

    def weights(self, X):
        r""" Applies reweighting vectors to data, yielding corresponding weights.

        Parameters
        ----------
        X : (T, d) ndarray
            The input data.

        Returns
        -------
        weights : (T, 1) ndarray
            Weights for input data.
        """
        return X.dot(self.weights_input) + self.const_weight_input

    def transform(self, data, **kw):
        r""" Same as :meth:`weights`. """
        return self.weights(data)

    @property
    def weights_input(self) -> np.ndarray:
        r""" Yields the reweighting vector in input basis.

        :type: (T, d) ndarray
        """
        return self._u

    @property
    def const_weight_input(self) -> float:
        r""" Yields the constant offset for reweighting in input basis.

        :type: float
        """
        return self._u_const

    @property
    def koopman_operator(self) -> np.ndarray:
        r""" The Koopman operator in modified basis (PC|1).

        :type: ndarray
        """
        return self._koopman_operator

    @property
    def whitening_transformation(self) -> np.ndarray:
        r""" Estimated whitening transformation for data

        :type: ndarray or None
        """
        return self._whitening_transformation

    @property
    def covariances(self) -> CovarianceModel:
        r""" Covariance model which was used to compute the Koopman model.

        :type: CovarianceModel or None
        """
        return self._covariances


class KoopmanWeightingEstimator(EstimatorTransformer):
    r"""Computes Koopman operator and weights that can be plugged into the :class:`Covariance` estimator.
    The weights are determined by the procedure described in :footcite:`wu2017variational`.

    Parameters
    ----------
    lagtime : int
        The lag time at which the operator is estimated.
    epsilon : float, optional, default=1e-6
        Truncation parameter. Eigenvalues with norms smaller than this cutoff will be removed.
    ncov : int or str, optional, default=infinity
        Depth of moment storage. Per default no moments are collapsed while estimating covariances, perform
        aggregation only at the very end after all data has been processed.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, lagtime, epsilon=1e-6, ncov='inf'):
        super(KoopmanWeightingEstimator, self).__init__()
        self.epsilon = epsilon
        if ncov == 'inf':
            ncov = int(2 ** 10000)
        self._cov = Covariance(lagtime=lagtime, compute_c00=True, compute_c0t=True, remove_data_mean=True,
                               reversible=False, bessels_correction=False, ncov=ncov)

    def fit(self, data, lagtime=None, **kw):
        r""" Fits a new model.

        Parameters
        ----------
        data : (T, d) ndarray
            The input data.
        lagtime : int, optional, default=None
            Optional override for estimator's :attr:`lagtime`.
        **kw
            Ignored keyword args for scikit-learn compatibility.

        Returns
        -------
        self : KoopmanWeightingEstimator
            Reference to self.
        """
        self._model = None
        self._cov.fit(data, lagtime=lagtime)
        return self

    def partial_fit(self, data):
        r""" Updates the current model using a chunk of data.

        Parameters
        ----------
        data : (T, d) ndarray
            A chunk of data.

        Returns
        -------
        self : KoopmanWeightingEstimator
            Reference to self.
        """
        self._cov.partial_fit(data)
        return self

    def transform(self, data, **kw):
        r""" Computes weights for a chunk of data. This requires that a model was :meth:`fit`.

        Parameters
        ----------
        data : (T, d) ndarray
            A chunk of data.
        **kw
            Ignored kwargs.

        Returns
        -------
        weights : (T, 1) ndarray
            Koopman weights.
        """
        return self.fetch_model().transform(data)

    @staticmethod
    def _compute_u(K):
        r"""Estimate an approximation of the ratio of stationary over empirical distribution from the basis.

        Parameters:
        -----------
        K0 : (M+1, M+1) ndarray
            Time-lagged correlation matrix for the whitened and padded data set.

        Returns:
        --------
        weights : (M,) ndarray
            Coefficients of the ratio stationary / empirical distribution from the whitened and expanded basis.
        """
        M = K.shape[0] - 1
        # Compute right and left eigenvectors:
        l, U = eig(K.T)
        l, U = sort_eigs(l, U)
        # Extract the eigenvector for eigenvalue one and normalize:
        u = np.real(U[:, 0])
        v = np.zeros(M + 1)
        v[M] = 1.0
        u = u / np.dot(u, v)
        return u

    def fetch_model(self) -> KoopmanWeightingModel:
        r""" Finalizes the model.

        Returns
        -------
        koopman_model : KoopmanWeightingModel
            The Koopman model, in particular containing operator and weights.
        """
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
        u_input = np.zeros(N + 1)
        u_input[0:N] = R.dot(u[0:-1])  # in input basis
        u_input[N] = u[-1] - cov.mean_0.dot(R.dot(u[0:-1]))

        self._model = KoopmanWeightingModel(u=u_input[:-1], u_const=u_input[-1], koopman_operator=K,
                                            whitening_transformation=R, covariances=cov)

        return self._model

    @property
    def lagtime(self) -> int:
        r""" The lagtime at which the Koopman operator is estimated.

        :getter: Yields the currently configured lagtime.
        :setter: Sets a new lagtime, must be >= 0.
        :type: int
        """
        return self._cov.lagtime

    @lagtime.setter
    def lagtime(self, value: int):
        self._cov.lagtime = value
