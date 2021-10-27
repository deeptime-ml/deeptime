from typing import Optional, Union, List, Callable

import numpy as np

from . import vamp_score
from ..base import Model, Transformer, Estimator
from ..basis import Identity, Concatenation
from ..covariance import CovarianceModel, WhiteningTransform
from ..numeric import is_diagonal_matrix
from ..util import LaggedModelValidator
from ..util.decorators import cached_property


class TransferOperatorModel(Model, Transformer):
    r""" Model which contains a finite-dimensional transfer operator (or approximation thereof).
    It describes the temporal evolution of observable space, i.e.,

    .. math:: \mathbb{E}[g(x_{t+\tau})] = K^\top \mathbb{E}[f(x_t)],

    where :math:`K\in\mathbb{R}^{n\times m}` is the transfer operator, :math:`x_t` the system's state at time :math:`t`,
    and :math:`f` and :math:`g` observables of the system's state.

    Parameters
    ----------
    koopman_matrix : (n, n) ndarray
        Applies the transform :math:`K^\top` in the modified basis.
    instantaneous_obs : Callable, optional, default=identity
        Transforms the current state :math:`x_t` to :math:`f(x_t)`. Defaults to `f(x) = x`.
    timelagged_obs : Callable, optional, default=identity
        Transforms the future state :math:`x_{t+\tau}` to :math:`g(x_{t+\tau})`. Defaults to `f(x) = x`.
    """

    def __init__(self, koopman_matrix: np.ndarray,
                 instantaneous_obs: Callable[[np.ndarray], np.ndarray] = Identity(),
                 timelagged_obs: Callable[[np.ndarray], np.ndarray] = Identity()):
        super().__init__()
        self._koopman_matrix = koopman_matrix
        self._instantaneous_obs = instantaneous_obs
        self._timelagged_obs = timelagged_obs

    @property
    def operator(self) -> np.ndarray:
        r""" The operator :math:`K` so that :math:`\mathbb{E}[g(x_{t+\tau}] = K^\top \mathbb{E}[f(x_t)]` in transformed
        bases.
        """
        return self._koopman_matrix

    @property
    def koopman_matrix(self) -> np.ndarray:
        r""" Same as :attr:`operator`. """
        return self.operator

    @cached_property
    def operator_inverse(self) -> np.ndarray:
        r""" Inverse of the operator :math:`K`, i.e., :math:`K^{-1}`.
        Potentially pseudo-inverse instead of true inverse.
        """
        return np.linalg.pinv(self.operator)

    @property
    def instantaneous_obs(self) -> Callable[[np.ndarray], np.ndarray]:
        r""" Transforms the current state :math:`x_t` to :math:`f(x_t)`. Defaults to `f(x) = x`. """
        return self._instantaneous_obs

    @property
    def timelagged_obs(self) -> Callable[[np.ndarray], np.ndarray]:
        r""" Transforms the future state :math:`x_{t+\tau}` to :math:`g(x_{t+\tau})`. Defaults to `f(x) = x`. """
        return self._timelagged_obs

    @property
    def output_dimension(self):
        r""" The dimension of data after propagation by :math:`K`. """
        return self.operator.shape[1]

    def forward(self, data: np.ndarray, propagate=True):
        r""" Maps data forward in time.

        Parameters
        ----------
        data : (T, n) ndarray
            Input data
        propagate : bool, default=True
            Whether to apply the Koopman operator to the featurized data.

        Returns
        -------
        mapped_data : (T, m) ndarray
            Mapped data.
        """
        out = self.instantaneous_obs(data)
        if propagate:
            out = out @ self.operator
        return out

    def backward(self, data: np.ndarray, propagate=True):
        r""" Maps data backward in time.

        Parameters
        ----------
        data : (T, n) ndarray
            Input data
        propagate : bool, default=True
            Whether to apply the Koopman operator to the featurized data.

        Returns
        -------
        mapped_data : (T, m) ndarray
            Mapped data.
        """
        out = self.timelagged_obs(data)
        if propagate:
            out = out @ self.operator_inverse
        return out

    def transform(self, data: np.ndarray, **kw):
        r""" Transforms data by applying the observables to it and then mapping them onto the modes of :math:`K`.

        Parameters
        ----------
        data : (T,n) ndarray
            Input data.

        Returns
        -------
        transformed_data : (T,m) ndarray
            The transformed data.
        """
        return self.instantaneous_obs(data)


class CovarianceKoopmanModel(TransferOperatorModel):
    r"""A type of Koopman model :math:`\mathbb{E}[g(x_{t+\tau})] = K^\top \mathbb{E}[f(x_{t})]`
    which was obtained through diagonalization of covariance matrices. This leads to
    a Koopman operator which is a diagonal matrix and can be used to project onto specific processes of the system.

    In particular, this model expects matrices :math:`U` and :math:`V` as well as singular values :math:`\sigma_i`,
    such that

    .. math::
        \mathbb{E}[V^\top\chi_1 (x_{t+\tau})]=\mathbb{E}[g(x_{t+\tau})] \approx K^\top \mathbb{E}[f(x_{t})] = \mathrm{diag}(\sigma_i) \mathbb{E}[U^\top\chi_0(x_{t})],

    where :math:`\chi_0,\chi_1` are basis transformations of the full state :math:`x_t`.

    The estimators which produce this kind of model are :class:`VAMP <deeptime.decomposition.VAMP>` and
    :class:`TICA <deeptime.decomposition.TICA>`.

    For a description of parameters `operator`, `basis_transform_forward`, `basis_transform_backward`,
    and `output_dimension`: please see :class:`TransferOperatorModel`.

    Parameters
    ----------
    instantaneous_coefficients : (n, k) ndarray
        The coefficient matrix :math:`U`.
    singular_values : (k,) ndarray
        Singular values :math:`\sigma_i`.
    instantaneous_coefficients : (n, k) ndarray
        The coefficient matrix :math:`V`.
    cov : CovarianceModel
        Covariances :math:`C_{00}`, :math:`C_{0t}`, and :math:`C_{tt}`.
    rank_0 : int
        Rank of the instantaneous whitening transformation :math:`C_{00}^{-1/2}`.
    rank_t : int
        Rank of the time-lagged whitening transformation :math:`C_{tt}^{-1/2}`.
    scaling : str or None, default=None
        Scaling parameter which was applied to singular values for additional structure in the projected space.
        See the respective estimator for details.
    epsilon : float, default=1e-6
        Eigenvalue / singular value cutoff. Eigenvalues (or singular values) of :math:`C_{00}` and :math:`C_{11}`
        with norms <= epsilon were cut off. The remaining number of eigenvalues together with the value of `dim`
        define the effective output dimension.
    instantaneous_obs : Callable, optional, default=identity
        Transforms the current state :math:`x_t` to :math:`\chi_0(x_t)`. Defaults to `\chi_0(x) = x`.
    timelagged_obs : Callable, optional, default=identity
        Transforms the future state :math:`x_{t+\tau}` to :math:`\chi_1(x_{t+\tau})`. Defaults to `\chi_1(x) = x`.
    """

    def __init__(self, instantaneous_coefficients, singular_values, timelagged_coefficients, cov,
                 rank_0: int, rank_t: int, dim=None, var_cutoff=None, scaling=None, epsilon=1e-10,
                 instantaneous_obs: Callable[[np.ndarray], np.ndarray] = Identity(),
                 timelagged_obs: Callable[[np.ndarray], np.ndarray] = Identity()):
        if not singular_values.ndim == 1:
            assert is_diagonal_matrix(singular_values)
            singular_values = np.diag(singular_values)
        self._whitening_instantaneous = WhiteningTransform(instantaneous_coefficients,
                                                           cov.mean_0 if cov.data_mean_removed else None)
        self._whitening_timelagged = WhiteningTransform(timelagged_coefficients,
                                                        cov.mean_t if cov.data_mean_removed else None)
        super().__init__(np.diag(singular_values),
                         Concatenation(self._whitening_instantaneous, instantaneous_obs),
                         Concatenation(self._whitening_timelagged, timelagged_obs))
        self._instantaneous_coefficients = instantaneous_coefficients
        self._timelagged_coefficients = timelagged_coefficients
        self._singular_values = singular_values
        self._cov = cov

        self._scaling = scaling
        self._epsilon = epsilon
        self._rank_0 = rank_0
        self._rank_t = rank_t
        self._dim = dim
        self._var_cutoff = var_cutoff
        self._update_output_dimension()

    @property
    def output_dimension(self):
        return self._output_dimension

    @property
    def instantaneous_coefficients(self) -> np.ndarray:
        r""" Coefficient matrix :math:`U`. """
        return self._instantaneous_coefficients

    @property
    def timelagged_coefficients(self) -> np.ndarray:
        r""" Coefficient matrix :math:`V`. """
        return self._timelagged_coefficients

    @property
    def scaling(self) -> Optional[str]:
        """Scaling of projection. Can be :code:`None`, 'kinetic map', or 'km' """
        return self._scaling

    @property
    def singular_vectors_left(self) -> np.ndarray:
        """Transformation matrix that represents the linear map from mean-free feature space
        to the space of left singular functions."""
        return self.instantaneous_coefficients

    @property
    def singular_vectors_right(self) -> np.ndarray:
        """Transformation matrix that represents the linear map from mean-free feature space
        to the space of right singular functions."""
        return self.timelagged_coefficients

    @property
    def singular_values(self) -> np.ndarray:
        """ The singular values of the half-weighted Koopman matrix. """
        return np.diag(self.operator)

    @property
    def cov(self) -> CovarianceModel:
        r""" Estimated covariances. """
        return self._cov

    @property
    def mean_0(self) -> np.ndarray:
        r""" Shortcut to :attr:`mean_0 <deeptime.covariance.CovarianceModel.mean_0>`. """
        return self.cov.mean_0

    @property
    def mean_t(self) -> np.ndarray:
        r""" Shortcut to :attr:`mean_t <deeptime.covariance.CovarianceModel.mean_t>`. """
        return self.cov.mean_t

    @property
    def cov_00(self) -> np.ndarray:
        r""" Shortcut to :attr:`cov_00 <deeptime.covariance.CovarianceModel.cov_00>`. """
        return self.cov.cov_00

    @property
    def cov_0t(self) -> np.ndarray:
        r""" Shortcut to :attr:`cov_0t <deeptime.covariance.CovarianceModel.cov_0t>`. """
        return self.cov.cov_0t

    @property
    def cov_tt(self) -> np.ndarray:
        r""" Shortcut to :attr:`cov_tt <deeptime.covariance.CovarianceModel.cov_tt>`. """
        return self.cov.cov_tt

    @property
    def whitening_rank_0(self) -> int:
        r""" Rank of the instantaneous whitening transformation :math:`C_{00}^{-1/2}`. """
        return self._rank_0

    @property
    def whitening_rank_t(self) -> int:
        r""" Rank of the time-lagged whitening transformation :math:`C_{tt}^{-1/2}`. """
        return self._rank_t

    @property
    def epsilon(self) -> float:
        r""" Singular value cutoff. """
        return self._epsilon

    @staticmethod
    def _cumvar(singular_values) -> np.ndarray:
        cumvar = np.cumsum(singular_values ** 2)
        cumvar /= cumvar[-1]
        return cumvar

    @staticmethod
    def effective_output_dimension(rank0, rankt, dim, var_cutoff, singular_values) -> int:
        r""" Computes effective output dimension. """
        if (dim is None and var_cutoff is None) or (var_cutoff is not None and var_cutoff == 1.0):
            return min(rank0, rankt)
        if var_cutoff is not None:
            return np.searchsorted(CovarianceKoopmanModel._cumvar(singular_values), var_cutoff) + 1
        else:
            if dim is None:
                return min(rank0, rankt)
            else:
                return np.min([rank0, rankt, dim])

    def _update_output_dimension(self):
        self._output_dimension = CovarianceKoopmanModel.effective_output_dimension(
            self.whitening_rank_0, self.whitening_rank_t, self.dim, self.var_cutoff, self.singular_values
        )
        self._whitening_instantaneous.dim = self.output_dimension
        self._whitening_timelagged.dim = self.output_dimension

    def backward(self, data: np.ndarray, propagate=True):
        out = self.timelagged_obs(data)
        if propagate:
            op = self.operator_inverse[:self.output_dimension, :self.output_dimension]
            out = out @ op
        return out

    def forward(self, data: np.ndarray, propagate=True):
        out = self.instantaneous_obs(data)
        if propagate:
            op = self.operator[:self.output_dimension, :self.output_dimension]
            out = out @ op
        return out

    def transform(self, data: np.ndarray, **kw):
        r"""Projects data onto the Koopman modes :math:`f(x) = U^\top \chi_0 (x)`, where :math:`U` are the
        coefficients of the basis :math:`\chi_0`.

        Parameters
        ----------
        data : (T, n) ndarray
            Input data.

        Returns
        -------
        transformed_data : (T, k) ndarray
            Data projected onto the Koopman modes.
        """
        return self.instantaneous_obs(data)

    @property
    def var_cutoff(self) -> Optional[float]:
        r""" Variance cutoff parameter. Can be set to include dimensions up to a certain threshold. Takes
        precedence over the :meth:`dim` parameter.

        :getter: Yields the current variance cutoff.
        :setter: Sets a new variance cutoff
        :type: float or None
        """
        return self._var_cutoff

    @var_cutoff.setter
    def var_cutoff(self, value):
        assert 0 < value <= 1., "Invalid dimension parameter, if it is given in terms of a variance cutoff, " \
                                "it can only be in the interval (0, 1]."
        self._var_cutoff = value
        self._update_output_dimension()

    @property
    def dim(self) -> Optional[int]:
        r""" Dimension attribute. Can either be int or None. In case of

        * :code:`int` it evaluates it as the actual dimension, must be strictly greater 0,
        * :code:`None` all numerically available components are used.

        :getter: yields the dimension
        :setter: sets a new dimension
        :type: int or None
        """
        return self._dim

    @dim.setter
    def dim(self, value: Optional[int]):
        if isinstance(value, int) and value <= 0:
            # first test against Integral as `isinstance(1, Real)` also evaluates to True
            raise ValueError("VAMP: Invalid dimension parameter, if it is given in terms of the "
                             "dimension (integer), must be positive.")
        self._dim = value
        self._update_output_dimension()

    @property
    def cumulative_kinetic_variance(self) -> np.ndarray:
        r""" Yields the cumulative kinetic variance. """
        return CovarianceKoopmanModel._cumvar(self.singular_values)

    @cached_property
    def _instantaneous_whitening_backwards(self):
        inv = np.linalg.pinv(self._whitening_instantaneous.sqrt_inv_cov)
        mean = self._whitening_instantaneous.mean
        if mean is None:
            mean = 0
        return lambda x: x @ inv + mean

    def propagate(self, trajectory: np.ndarray, components: Optional[Union[int, List[int]]] = None) -> np.ndarray:
        r""" Applies the forward transform to the trajectory in non-transformed space. Given the Koopman operator
        :math:`\Sigma`, transformations  :math:`V^\top - \mu_t` and :math:`U^\top -\mu_0` for
        bases :math:`f` and :math:`g`, respectively, this is achieved by transforming each frame :math:`X_t` with

        .. math::
            \hat{X}_{t+\tau} = (V^\top)^{-1} \Sigma U^\top (X_t - \mu_0) + \mu_t.

        If the model stems from a :class:`VAMP <deeptime.decomposition.VAMP>` estimator, :math:`V` are the left
        singular vectors, :math:`\Sigma` the singular values, and :math:`U` the right singular vectors.

        Parameters
        ----------
        trajectory : (T, n) ndarray
            The input trajectory
        components : int or list of int or None, default=None
            Optional arguments for the Koopman operator if appropriate. If the model stems from
            a :class:`VAMP <deeptime.decomposition.VAMP>` estimator, these are the component(s) to project onto.
            If None, all processes are taken into account, if list of integer, this sets all singular values
            to zero but the "components"th ones.

        Returns
        -------
        predictions : (T, n) ndarray
            The predicted trajectory.
        """
        if components is not None:
            operator = np.zeros_like(self.operator)
            if not isinstance(components, (list, tuple)):
                components = [components]
            for ii in components:
                operator[ii, ii] = self.operator[ii, ii]
        else:
            operator = self.operator
        out = self.instantaneous_obs(trajectory)  # (T, N) -> (T, n) project into whitened space
        out = out @ operator  # (T, n) -> (T, n) propagation in whitened space
        out = self._instantaneous_whitening_backwards(out)  # (T, n) -> (T, N) back into original space
        return out

    def score(self, r: Union[float, str], test_model=None, epsilon=1e-6, dim=None):
        """Compute the VAMP score between a this model and potentially a test model for cross-validation.

        Parameters
        ----------
        r : float or str
            The type of score to evaluate. Can by an floating point value greater or equal to 1 or 'E', yielding the
            VAMP-r score or the VAMP-E score, respectively. :footcite:`wu2020variational`
            Typical choices are:

            *  'VAMP1'  Sum of singular values of the half-weighted Koopman matrix.
                        If the model is reversible, this is equal to the sum of
                        Koopman matrix eigenvalues, also called Rayleigh quotient :footcite:`wu2020variational`.
            *  'VAMP2'  Sum of squared singular values of the half-weighted Koopman
                        matrix :footcite:`wu2020variational`. If the model is reversible, this is
                        equal to the kinetic variance :footcite:`noe2015kinetic`.
            *  'VAMPE'  Approximation error of the estimated Koopman operator with respect to
                        the true Koopman operator up to an additive constant :footcite:`wu2020variational` .

        test_model : CovarianceKoopmanModel, optional, default=None

            If `test_model` is not None, this method computes the cross-validation score
            between self and `covariances_test`. It is assumed that self was estimated from
            the "training" data and `test_model` was estimated from the "test" data. The
            score is computed for one realization of self and `test_model`. Estimation
            of the average cross-validation score and partitioning of data into test and
            training part is not performed by this method.

            If `covariances_test` is None, this method computes the VAMP score for the model
            contained in self.

        epsilon : float, default=1e-6
            Regularization parameter for computing sqrt-inverses of spd matrices.
        dim : int, optional, default=None
            How many components to use for scoring.

        Returns
        -------
        score : float
            If `test_model` is not None, returns the cross-validation VAMP score between
            self and `test_model`. Otherwise return the selected VAMP-score of self.

        Notes
        -----
        The VAMP-:math:`r` and VAMP-E scores are computed according to :footcite:`wu2020variational`,
        Equation (33) and Equation (30), respectively.

        References
        ----------
        .. footbibliography::
        """
        test_cov = test_model.cov if test_model is not None else None
        dim = self.output_dimension if dim is None else dim
        return vamp_score(self, r, test_cov, dim, epsilon)

    def expectation(self, observables, statistics, lag_multiple=1, observables_mean_free=False,
                    statistics_mean_free=False):
        r"""Compute future expectation of observable or covariance using the approximated Koopman operator.

        Parameters
        ----------
        observables : np.ndarray((input_dimension, n_observables))
            Coefficients that express one or multiple observables in
            the basis of the input features.

        statistics : np.ndarray((input_dimension, n_statistics)), optional
            Coefficients that express one or multiple statistics in
            the basis of the input features.
            This parameter can be None. In that case, this method
            returns the future expectation value of the observable(s).

        lag_multiple : int
            If > 1, extrapolate to a multiple of the estimator's lag
            time by assuming Markovianity of the approximated Koopman
            operator.

        observables_mean_free : bool, default=False
            If true, coefficients in `observables` refer to the input
            features with feature means removed.
            If false, coefficients in `observables` refer to the
            unmodified input features.

        statistics_mean_free : bool, default=False
            If true, coefficients in `statistics` refer to the input
            features with feature means removed.
            If false, coefficients in `statistics` refer to the
            unmodified input features.

        Returns
        -------
        expectation : ndarray
            The equilibrium expectation of observables or covariance if statistics is not None.

        Notes
        -----
        A "future expectation" of a observable g is the average of g computed
        over a time window that has the same total length as the input data
        from which the Koopman operator was estimated but is shifted
        by ``lag_multiple*tau`` time steps into the future (where tau is the lag
        time).

        It is computed with the equation:

        .. math::

            \mathbb{E}[g]_{\rho_{n}}=\mathbf{q}^{T}\mathbf{P}^{n-1}\mathbf{e}_{1}

        where

        .. math::

            P_{ij}=\sigma_{i}\langle\psi_{i},\phi_{j}\rangle_{\rho_{1}}

        and

        .. math::

            q_{i}=\langle g,\phi_{i}\rangle_{\rho_{1}}

        and :math:`\mathbf{e}_{1}` is the first canonical unit vector.


        A model prediction of time-lagged covariances between the
        observable f and the statistic g at a lag-time of ``lag_multiple*tau``
        is computed with the equation:

        .. math::

            \mathrm{cov}[g,\,f;n\tau]=\mathbf{q}^{T}\mathbf{P}^{n-1}\boldsymbol{\Sigma}\mathbf{r}

        where :math:`r_{i}=\langle\psi_{i},f\rangle_{\rho_{0}}` and
        :math:`\boldsymbol{\Sigma}=\mathrm{diag(\boldsymbol{\sigma})}` .
        """
        if lag_multiple <= 0:
            raise ValueError("lag_multiple <= 0 not implemented")

        dim = self.output_dimension

        S = np.diag(np.concatenate(([1.0], self.singular_values[0:dim])))
        V = self.singular_vectors_right[:, 0:dim]
        U = self.singular_vectors_left[:, 0:dim]
        m_0 = self.mean_0
        m_t = self.mean_t

        if lag_multiple == 1:
            P = S
        else:
            p = np.zeros((dim + 1, dim + 1))
            p[0, 0] = 1.0
            p[1:, 0] = U.T.dot(m_t - m_0)
            p[1:, 1:] = U.T.dot(self.cov_tt).dot(V)
            P = np.linalg.matrix_power(S.dot(p), lag_multiple - 1).dot(S)

        Q = np.zeros((observables.shape[1], dim + 1))
        if not observables_mean_free:
            Q[:, 0] = observables.T.dot(m_t)
        Q[:, 1:] = observables.T.dot(self.cov_tt).dot(V)

        if statistics is not None:
            # compute covariance
            R = np.zeros((statistics.shape[1], dim + 1))
            if not statistics_mean_free:
                R[:, 0] = statistics.T.dot(m_0)
            R[:, 1:] = statistics.T.dot(self.cov_00).dot(U)

            # compute lagged covariance
            return Q.dot(P).dot(R.T)
            # TODO: discuss whether we want to return this or the transpose
            # TODO: from MSMs one might expect to first index to refer to the statistics, here it is the other way round
        else:
            # compute future expectation
            return Q.dot(P)[:, 0]

    def timescales(self, k=None, lagtime: Optional[int] = None) -> np.ndarray:
        r"""Implied timescales of the TICA transformation

        For each :math:`i`-th eigenvalue, this returns

        .. math::

            t_i = -\frac{\tau}{\log(|\lambda_i|)}

        where :math:`\tau` is the :attr:`lagtime` of the TICA object and :math:`\lambda_i` is the `i`-th
        :attr:`eigenvalue <eigenvalues>` of the TICA object.

        Parameters
        ----------
        k : int, optional, default=None
            Number of timescales to be returned. By default with respect to all available singular values.
        lagtime : int, optional, default=None
            The lagtime with respect to which to compute the timescale. If :code:`None`, this defaults to the
            lagtime under which the covariances were estimated.

        Returns
        -------
        timescales: (n,) np.array
            numpy array with the implied timescales. In principle, one should expect as many timescales as
            input coordinates were available. However, less eigenvalues will be returned if the TICA matrices
            were not full rank or :attr:`dim` contained a floating point percentage, i.e., was interpreted as
            variance cutoff.

        Raises
        ------
        ValueError
            If any of the singular values not real, i.e., has a non-zero imaginary component.
        """
        if not np.all(np.isreal(self.singular_values)):
            raise ValueError("This is only meaningful for real singular values.")
        if lagtime is None:
            lagtime = self._cov.lagtime
        return - lagtime / np.log(np.abs(self.singular_values[:k]))

    @property
    def feature_component_correlation(self):
        r"""Instantaneous correlation matrix between mean-free input features and projection components.

        Denoting the input features as :math:`X_i` and the projection components as :math:`\theta_j`, the
        instantaneous, linear correlation between them can be written as

        .. math::
            \mathbf{Corr}(X_i - \mu_i, \mathbf{\theta}_j) = \frac{1}{\sigma_{X_i - \mu_i}}\sum_l \sigma_{(X_i - \mu_i)(X_l - \mu_l)} \mathbf{U}_{li}

        The matrix :math:`\mathbf{U}` is the matrix containing the eigenvectors of the generalized
        eigenvalue problem as column vectors.

        Returns
        -------
        corr : ndarray(n,m)
            Correlation matrix between input features and projection components. There is a row for each
            feature and a column for each component.
        """
        feature_sigma = np.sqrt(np.diag(self.cov_00))
        return np.dot(self.cov_00, self.singular_vectors_left[:, : self.output_dimension]) / feature_sigma[:, None]


class KoopmanChapmanKolmogorovValidator(LaggedModelValidator):
    r""" Validates a Koopman model

    """

    def __init__(self, test_model: Model, test_estimator: Estimator, test_model_lagtime: int, mlags,
                 observables, statistics, observables_mean_free, statistics_mean_free):
        super().__init__(test_model, test_estimator, test_model_lagtime, mlags)
        self.observables = observables
        self.statistics = statistics
        self.observables_mean_free = observables_mean_free
        self.statistics_mean_free = statistics_mean_free

    @property
    def statistics(self):
        return self._statistics

    @statistics.setter
    def statistics(self, value):
        self._statistics = value
        if self._statistics is not None:
            self.nsets = min(self.observables.shape[1], self._statistics.shape[1])

    def _compute_observables(self, model: CovarianceKoopmanModel, mlag):
        # todo for lag time 0 we return a matrix of nan, until the correct solution is implemented
        if mlag == 0 or model is None:
            if self.statistics is None:
                return np.zeros(self.observables.shape[1]) + np.nan
            else:
                return np.zeros((self.observables.shape[1], self.statistics.shape[1])) + np.nan
        else:
            return model.expectation(statistics=self.statistics, observables=self.observables, lag_multiple=mlag,
                                     statistics_mean_free=self.statistics_mean_free,
                                     observables_mean_free=self.observables_mean_free)

    def _compute_observables_conf(self, model, mlag, conf=0.95):
        raise NotImplementedError('estimation of confidence intervals not yet implemented for Koopman models')
