"""
@author: paul, marscher, wu, noe, clonker
"""

from collections import namedtuple
from numbers import Integral
from typing import Optional, Union, Callable

import numpy as np

from ._koopman import CovarianceKoopmanModel, KoopmanChapmanKolmogorovValidator
from ..base import EstimatorTransformer
from ..basis import Identity
from ..covariance import Covariance, CovarianceModel
from ..numeric import spd_inv_split
from ..util.types import to_dataset


class VAMP(EstimatorTransformer):
    r"""Variational approach for Markov processes (VAMP).

    The implementation is based on :footcite:`wu2020variational`, :footcite:`noe2015kinetic`.

    Parameters
    ----------
    lagtime : int or None, optional, default=None
        The lagtime under which covariances are estimated. This is only relevant when estimating from data, in case
        covariances are provided this should either be None or exactly the value that was used to estimate
        said covariances.
    dim : int, optional, default=None
        Number of dimensions to keep:

        * if dim is not set (None) all available ranks are kept:
          :code:`n_components == min(n_samples, n_uncorrelated_features)`
        * if dim is an integer >= 1, this number specifies the number
          of dimensions to keep.
    var_cutoff : float, optional, default=None
        Determines the number of output dimensions by including dimensions until their cumulative kinetic variance
        exceeds the fraction subspace variance. var_cutoff=1.0 means all numerically available dimensions
        (see epsilon) will be used, unless set by dim. Setting var_cutoff smaller than 1.0 is exclusive with dim.
    scaling : str, optional, default=None
        Scaling to be applied to the VAMP order parameters upon transformation

        * None: no scaling will be applied, variance of the order parameters is 1
        * 'kinetic_map' or 'km': order parameters are scaled by singular value.
          Only the left singular functions induce a kinetic map wrt the
          conventional forward propagator. The right singular functions induce
          a kinetic map wrt the backward propagator.
    epsilon : float, optional, default=1e-6
        Eigenvalue cutoff. Eigenvalues of :math:`C_{00}` and :math:`C_{11}`
        with norms <= epsilon will be cut off. The remaining number of
        eigenvalues together with the value of `dim` define the size of the output.
    observable_transform : callable, optional, default=Identity
        A feature transformation on the raw data which is used to estimate the model.

    See Also
    --------
    CovarianceKoopmanModel : type of model produced by this estimator

    Notes
    -----
    VAMP is a method for dimensionality reduction of Markov processes.

    The Koopman operator :math:`\mathcal{K}` is an integral operator
    that describes conditional future expectation values. Let
    :math:`p(\mathbf{x},\,\mathbf{y})` be the conditional probability
    density of visiting an infinitesimal phase space volume around
    point :math:`\mathbf{y}` at time :math:`t+\tau` given that the phase
    space point :math:`\mathbf{x}` was visited at the earlier time
    :math:`t`. Then the action of the Koopman operator on a function
    :math:`f` can be written as follows:

    .. math::

      \mathcal{K}f=\int p(\mathbf{x},\,\mathbf{y})f(\mathbf{y})\,\mathrm{dy}=\mathbb{E}\left[f(\mathbf{x}_{t+\tau}\mid\mathbf{x}_{t}=\mathbf{x})\right]

    The Koopman operator is defined without any reference to an
    equilibrium distribution. Therefore it is well-defined in
    situations where the dynamics is irreversible or/and non-stationary
    such that no equilibrium distribution exists.

    If we approximate :math:`f` by a linear superposition of ansatz
    functions :math:`\boldsymbol{\chi}` of the conformational
    degrees of freedom (features), the operator :math:`\mathcal{K}`
    can be approximated by a (finite-dimensional) matrix :math:`\mathbf{K}`.

    The approximation is computed as follows: From the time-dependent
    input features :math:`\boldsymbol{\chi}(t)`, we compute the mean
    :math:`\boldsymbol{\mu}_{0}` (:math:`\boldsymbol{\mu}_{1}`) from
    all data excluding the last (first) :math:`\tau` steps of every
    trajectory as follows:

    .. math::

      \boldsymbol{\mu}_{0}	:=\frac{1}{T-\tau}\sum_{t=0}^{T-\tau}\boldsymbol{\chi}(t)

      \boldsymbol{\mu}_{1}	:=\frac{1}{T-\tau}\sum_{t=\tau}^{T}\boldsymbol{\chi}(t)

    Next, we compute the instantaneous covariance matrices
    :math:`\mathbf{C}_{00}` and :math:`\mathbf{C}_{11}` and the
    time-lagged covariance matrix :math:`\mathbf{C}_{01}` as follows:

    .. math::

        \begin{aligned}
      \mathbf{C}_{00}&:=\frac{1}{T-\tau}\sum_{t=0}^{T-\tau}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{0}\right]\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{0}\right]\\
      \mathbf{C}_{11}&:=\frac{1}{T-\tau}\sum_{t=\tau}^{T}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{1}\right]\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{1}\right]\\
      \mathbf{C}_{01}&:=\frac{1}{T-\tau}\sum_{t=0}^{T-\tau}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{0}\right]\left[\boldsymbol{\chi}(t+\tau)-\boldsymbol{\mu}_{1}\right]
        \end{aligned}

    The Koopman matrix is then computed as follows:

    .. math::

      \mathbf{K}=\mathbf{C}_{00}^{-1}\mathbf{C}_{01}

    It can be shown :footcite:`wu2020variational` that the leading singular functions of the
    half-weighted Koopman matrix

    .. math::

      \bar{\mathbf{K}}:=\mathbf{C}_{00}^{-\frac{1}{2}}\mathbf{C}_{01}\mathbf{C}_{11}^{-\frac{1}{2}}

    encode the best reduced dynamical model for the time series.

    The singular functions can be computed by first performing the
    singular value decomposition

    .. math::

      \bar{\mathbf{K}}=\mathbf{U}^{\prime}\mathbf{S}\mathbf{V}^{\prime}

    and then mapping the input conformation to the left singular
    functions :math:`\boldsymbol{\psi}` and right singular
    functions :math:`\boldsymbol{\phi}` as follows:

    .. math::

        \begin{aligned}
        \boldsymbol{\psi}(t)&:=\mathbf{U}^{\prime\top}\mathbf{C}_{00}^{-\frac{1}{2}}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{0}\right]\\
        \boldsymbol{\phi}(t)&:=\mathbf{V}^{\prime\top}\mathbf{C}_{11}^{-\frac{1}{2}}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{1}\right]
        \end{aligned}


    References
    ----------
    .. footbibliography::
    """

    def __init__(self, lagtime: Optional[int] = None,
                 dim: Optional[int] = None,
                 var_cutoff: Optional[float] = None,
                 scaling: Optional[str] = None,
                 epsilon: float = 1e-6,
                 observable_transform: Callable[[np.ndarray], np.ndarray] = Identity()):
        super(VAMP, self).__init__()
        self.dim = dim
        self.var_cutoff = var_cutoff
        self.scaling = scaling
        self.epsilon = epsilon
        self.lagtime = lagtime
        self.observable_transform = observable_transform
        self._covariance_estimator = None  # internal covariance estimator

    _DiagonalizationResults = namedtuple("DiagonalizationResults", ['rank0', 'rankt', 'singular_values',
                                                                    'left_singular_vecs', 'right_singular_vecs'])

    @staticmethod
    def _decomposition(covariances, epsilon, scaling, dim, var_cutoff) -> _DiagonalizationResults:
        """Performs SVD on covariance matrices and save left, right singular vectors and values in the model."""
        L0 = spd_inv_split(covariances.cov_00, epsilon=epsilon)
        rank0 = L0.shape[1] if L0.ndim == 2 else 1
        Lt = spd_inv_split(covariances.cov_tt, epsilon=epsilon)
        rankt = Lt.shape[1] if Lt.ndim == 2 else 1

        W = np.dot(L0.T, covariances.cov_0t).dot(Lt)
        from scipy.linalg import svd
        A, s, BT = svd(W, compute_uv=True, lapack_driver='gesvd')

        singular_values = s

        m = CovarianceKoopmanModel.effective_output_dimension(rank0, rankt, dim, var_cutoff, singular_values)

        U = np.dot(L0, A[:, :m])
        V = np.dot(Lt, BT[:m, :].T)

        # scale vectors
        if scaling is not None and scaling in ("km", "kinetic_map"):
            U *= s[np.newaxis, 0:m]  # scaled left singular functions induce a kinetic map
            V *= s[np.newaxis, 0:m]  # scaled right singular functions induce a kinetic map wrt. backward propagator

        return VAMP._DiagonalizationResults(
            rank0=rank0, rankt=rankt, singular_values=singular_values, left_singular_vecs=U, right_singular_vecs=V
        )

    @classmethod
    def covariance_estimator(cls, lagtime: int, ncov: Union[int] = float('inf')):
        r""" Yields a properly configured covariance estimator so that its model can be used as input for the vamp
        estimator.

        Parameters
        ----------
        lagtime : int
            Positive integer denoting the time shift which is considered for autocorrelations.
        ncov : int or float('inf'), optional, default=float('inf')
            Limit the memory usage of the algorithm from :footcite:`chan1982updating` to an amount that corresponds
            to ncov additional copies of each correlation matrix.

        Returns
        -------
        estimator : Covariance
            Covariance estimator.
        """
        return Covariance(lagtime=lagtime, compute_c0t=True, compute_ctt=True, remove_data_mean=True, reversible=False,
                          bessels_correction=False, ncov=ncov)

    @staticmethod
    def _to_covariance_model(covariances: Union[Covariance, CovarianceModel]) -> CovarianceModel:
        if isinstance(covariances, Covariance):
            covariances = covariances.fetch_model()
        return covariances

    def partial_fit(self, data):
        r""" Updates the covariance estimates through a new batch of data.

        Parameters
        ----------
        data : tuple(ndarray, ndarray)
            A tuple of ndarrays which have to have same shape and are :math:`X_t` and :math:`X_{t+\tau}`, respectively.
            Here, :math:`\tau` denotes the lagtime.

        Returns
        -------
        self : VAMP
            Reference to self.
        """
        if self._covariance_estimator is None:
            self._covariance_estimator = self.covariance_estimator(lagtime=self.lagtime)
        x, y = to_dataset(data, lagtime=self.lagtime)[:]
        self._covariance_estimator.partial_fit((self.observable_transform(x),
                                                self.observable_transform(y)))
        return self

    def fit_from_covariances(self, covariances: Union[Covariance, CovarianceModel]):
        r"""Fits from existing covariance model (or covariance estimator containing model).

        Parameters
        ----------
        covariances : CovarianceModel or Covariance
            Covariance model containing covariances or Covariance estimator containing a covariance model. The model
            in particular has matrices :math:`C_{00}, C_{0t}, C_{tt}`.

        Returns
        -------
        self : VAMP
            Reference to self.
        """
        self._covariance_estimator = None
        covariances = self._to_covariance_model(covariances)
        self._model = self._decompose(covariances)
        return self

    def fit_from_timeseries(self, data, weights=None):
        r""" Estimates a :class:`CovarianceKoopmanModel` directly from time-series data using the :class:`Covariance`
        estimator. For parameters `dim`, `scaling`, `epsilon`.

        Parameters
        ----------
        data
            Input data, see :meth:`to_dataset <deeptime.util.types.to_dataset>` for options.
        weights
            See the :class:`Covariance <deeptime.covariance.Covariance>` estimator.

        Returns
        -------
        self : VAMP
            Reference to self.
        """
        dataset = to_dataset(data, lagtime=self.lagtime)
        self._covariance_estimator = self.covariance_estimator(lagtime=self.lagtime)
        x, y = dataset[:]
        transformed = (self.observable_transform(x), self.observable_transform(y))
        covariances = self._covariance_estimator.partial_fit(transformed, weights=weights)\
            .fetch_model()
        return self.fit_from_covariances(covariances)

    @property
    def epsilon(self):
        r""" Eigenvalue cutoff.

        :getter: Yields current eigenvalue cutoff.
        :setter: Sets new eigenvalue cutoff.
        :type: float
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

    @property
    def lagtime(self) -> Optional[int]:
        r""" The lagtime under which covariances are estimated. Can be `None` in case covariances are provided
        directly instead of estimating them inside this estimator.

        :getter: Yields the current lagtime.
        :setter: Sets a new lagtime, must be positive.
        :type: int or None
        """
        return self._lagtime

    @lagtime.setter
    def lagtime(self, value: Optional[int]):
        if value is not None and value <= 0:
            raise ValueError(f"Lagtime needs to be strictly positive but was {value}")
        self._lagtime = value

    @property
    def scaling(self) -> Optional[str]:
        r""" Scaling parameter to be applied to order parameters upon transformation. Can be one of None, 'kinetic_map',
        or 'km'.

        :getter: Yields currently configured scaling parameter.
        :setter: Sets a new scaling parameter (None, 'kinetic_map', or 'km')
        :type: str or None
        """
        return self._scaling

    @scaling.setter
    def scaling(self, value: Optional[str]):
        valid = (None, 'km', 'kinetic_map')
        if value not in valid:
            raise ValueError("Invalid scaling parameter \"{}\", can only be one of {}".format(value, valid))
        self._scaling = value

    @property
    def var_cutoff(self) -> Optional[float]:
        r""" Variational cutoff which can be used to further restrict the dimension. This takes precedence over the
        :meth:`dim` property.

        :getter: yields the currently set variation cutoff
        :setter: sets a new cutoff
        :type: float or None
        """
        return self._var_cutoff

    @var_cutoff.setter
    def var_cutoff(self, value: Optional[float]):
        if value is not None and (value <= 0. or float(value) > 1.0):
            raise ValueError("VAMP: Invalid var_cutoff parameter, can only be in the interval (0, 1].")
        self._var_cutoff = value

    @property
    def dim(self) -> Optional[int]:
        r""" Dimension attribute. Can either be int or float. In case of

        * :code:`int` it evaluates it as the actual dimension, must be strictly greater 0,
        * :code:`None` all numerically available components are used.

        :getter: yields the dimension
        :setter: sets a new dimension
        :type: int or None
        """
        return self._dim

    @dim.setter
    def dim(self, value: Optional[Union[int, float]]):
        if isinstance(value, Integral):
            if value <= 0:
                # first test against Integral as `isinstance(1, Real)` also evaluates to True
                raise ValueError("VAMP: Invalid dimension parameter, if it is given in terms of the "
                                 "dimension (integer), must be positive.")
        elif value is not None:
            raise ValueError("Invalid type for dimension, got {}".format(value))
        self._dim = value

    def _decompose(self, covariances: CovarianceModel):
        decomposition = self._decomposition(covariances, self.epsilon, self.scaling, self.dim, self.var_cutoff)
        return CovarianceKoopmanModel(
            decomposition.left_singular_vecs, decomposition.singular_values, decomposition.right_singular_vecs,
            rank_0=decomposition.rank0, rank_t=decomposition.rankt, dim=self.dim,
            var_cutoff=self.var_cutoff, cov=covariances, scaling=self.scaling, epsilon=self.epsilon,
            instantaneous_obs=self.observable_transform,
            timelagged_obs=self.observable_transform
        )

    def fit(self, data, *args, **kw):
        r""" Fits a new :class:`CovarianceKoopmanModel` which can be obtained by a
        subsequent call to :meth:`fetch_model`.

        Parameters
        ----------
        data : CovarianceModel or Covariance or timeseries
            Covariance matrices :math:`C_{00}, C_{0t}, C_{tt}` in form of a CovarianceModel instance. If the model
            should be fitted directly from data, please see :meth:`from_data`.
            Optionally, this can also be timeseries data directly, in which case a 'lagtime' must be provided.
        *args
            Optional arguments
        **kw
            Ignored keyword arguments for scikit-learn compatibility.

        Returns
        -------
        self : VAMP
            Reference to self.
        """
        if isinstance(data, (Covariance, CovarianceModel)):
            self.fit_from_covariances(data)
        else:
            self.fit_from_timeseries(data, weights=kw.pop('weights', None))
        return self

    def transform(self, data, propagate=False):
        r""" Projects given timeseries onto dominant singular functions. This method dispatches to
        :meth:`CovarianceKoopmanModel.transform`.

        Parameters
        ----------
        data : (T, n) ndarray
            Input timeseries data.
        propagate : bool, default=False
            Whether to apply the Koopman operator after data was transformed into the whitened feature space.

        Returns
        -------
        Y : (T, m) ndarray
            The projected data.
            If `right` is True, projection will be on the right singular functions. Otherwise, projection will be on
            the left singular functions.
        """
        return self.fetch_model().transform(data, propagate=propagate)

    def fetch_model(self) -> CovarianceKoopmanModel:
        r""" Finalizes current model and yields new :class:`CovarianceKoopmanModel`.

        Returns
        -------
        model : CovarianceKoopmanModel
            The estimated model.
        """
        if self._covariance_estimator is not None:
            # This can only occur when partial_fit was called.
            # A call to fit, fit_from_timeseries, fit_from_covariances ultimately always leads to a call to
            # fit_from_covariances which sets the self._covariance_estimator to None.
            self._model = self._decompose(self._covariance_estimator.fetch_model())
            self._covariance_estimator = None
        return self._model

    def chapman_kolmogorov_validator(self, mlags, test_model: CovarianceKoopmanModel = None,
                                     n_observables=None, observables='phi', statistics='psi'):
        r"""Returns a Chapman-Kolmogorov validator based on this estimator and a test model.

        Parameters
        ----------
        mlags : int or int-array
            Multiple of lagtimes of the test_model to test against.
        test_model : CovarianceKoopmanModel, optional, default=None
            The model that is tested. If not provided, uses this estimator's encapsulated model.
        n_observables : int, optional, default=None
            Limit the number of default observables (and of default statistics) to this number.
            Only used if `observables` are None or `statistics` are None.
        observables : (input_dimension, n_observables) ndarray
            Coefficients that express one or multiple observables in
            the basis of the input features.
        statistics : (input_dimension, n_statistics) ndarray
            Coefficients that express one or multiple statistics in the basis of the input features.

        Returns
        -------
        validator : KoopmanChapmanKolmogorovValidator
            The validator.

        Notes
        -----
        This method computes two sets of time-lagged covariance matrices

        * estimates at higher lag times :

          .. math::

              \left\langle \mathbf{K}(n\tau)g_{i},f_{j}\right\rangle_{\rho_{0}}

          where :math:`\rho_{0}` is the empirical distribution implicitly defined
          by all data points from time steps 0 to T-tau in all trajectories,
          :math:`\mathbf{K}(n\tau)` is a rank-reduced Koopman matrix estimated
          at the lag-time n*tau and g and f are some functions of the data.
          Rank-reduction of the Koopman matrix is controlled by the `dim`
          parameter of :class:`VAMP <deeptime.decomposition.VAMP>`.

        * predictions at higher lag times :

          .. math::

              \left\langle \mathbf{K}^{n}(\tau)g_{i},f_{j}\right\rangle_{\rho_{0}}

          where :math:`\mathbf{K}^{n}` is the n'th power of the rank-reduced
          Koopman matrix contained in self.

        The Champan-Kolmogorov test is to compare the predictions to the estimates.
        """
        test_model = self.fetch_model() if test_model is None else test_model
        assert test_model is not None, "We need a test model via argument or an estimator which was already" \
                                       "fit to data."
        lagtime = self.lagtime
        if n_observables is not None:
            if n_observables > test_model.dim:
                import warnings
                warnings.warn('Selected singular functions as observables but dimension '
                              'is lower than requested number of observables.')
                n_observables = test_model.dim
        else:
            n_observables = test_model.dim

        if isinstance(observables, str) and observables == 'phi':
            observables = test_model.singular_vectors_right[:, :n_observables]
            observables_mean_free = True
        else:
            observables_mean_free = False

        if isinstance(statistics, str) and statistics == 'psi':
            statistics = test_model.singular_vectors_left[:, :n_observables]
            statistics_mean_free = True
        else:
            statistics_mean_free = False
        return VAMPKoopmanCKValidator(test_model, self, lagtime, mlags, observables, statistics,
                                      observables_mean_free, statistics_mean_free)


def _vamp_estimate_model_for_lag(estimator: VAMP, model, data, lagtime):
    est = VAMP(lagtime=lagtime, dim=estimator.dim, var_cutoff=estimator.var_cutoff, scaling=estimator.scaling,
               epsilon=estimator.epsilon, observable_transform=estimator.observable_transform)
    return est.fit(data).fetch_model()


class VAMPKoopmanCKValidator(KoopmanChapmanKolmogorovValidator):

    def fit(self, data, n_jobs=None, progress=None, **kw):
        return super().fit(data, n_jobs, progress, _vamp_estimate_model_for_lag, **kw)
