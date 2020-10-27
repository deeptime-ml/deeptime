from typing import Optional, Union, List

import numpy as np

from ..base import Model, Transformer
from ..covariance import CovarianceModel
from ..numeric import is_diagonal_matrix, spd_inv_sqrt
from ..util.decorators import cached_property


class KoopmanBasisTransform(object):
    r""" Transforms a system's observable

    .. math:: f(x_t) = (\chi^{(1)}(x_t),\ldots,\chi^{(n)})^\top

    to another basis

    .. math:: \tilde f(x_t) = T (f(x_t) - \mu ),

    where :math:`T` is the transformation matrix and :math:`\mu` a constant mean value that is subtracted.
    """

    def __init__(self, mean: Union[float, np.ndarray], transformation_matrix):
        r""" Creates a new Koopman basis transform instance.

        Parameters
        ----------
        mean : (n,) ndarray or float
            Mean value which gets subtracted from each frame prior to transformation.
        transformation_matrix : (n, m) ndarray
            Transformation matrix :math:`T`.
        """
        self._mean = mean
        self._transformation_matrix = transformation_matrix

    @property
    def transformation_matrix(self) -> np.ndarray:
        r""" The transformation matrix :math:`T`. """
        return self._transformation_matrix

    @property
    def mean(self) -> np.ndarray:
        r""" The mean :math:`\mu`. """
        return self._mean

    @cached_property
    def backward_transformation_matrix(self):
        r""" Yields the (pseudo) inverse :math:`T^{-1}`. """
        return np.linalg.pinv(self._transformation_matrix)

    def __call__(self, data, inverse=False, dim=None):
        r""" Applies the basis transform to data.

        Parameters
        ----------
        data : (T, n) ndarray
            Data consisting of `T` frames in `n` dimensions.
        inverse : bool, default=False
            Whether to apply the forward or backward operation, i.e., :math:`T (f(x_t) - \mu )` or
            :math:`T^{-1} f(x_t) + \mu`, respectively.
        dim : int or None
            Number of dimensions to restrict to, removes the all but the first :math:`n` basis transformation vectors.
            Can only be not `None` if `inverse` is False.

        Returns
        -------
        transformed_data : (T, k) ndarray
            Transformed data. If :math:`T\in\mathbb{R}^{n\times m}`, we get :math:`k=\min\{m, \mathrm{dim}\}`.
        """
        if not inverse:
            return (data - self._mean) @ self._transformation_matrix[:, :dim]
        else:
            if dim is not None:
                raise ValueError("This currently only works for the forward transform.")
            return data @ self.backward_transformation_matrix + self._mean


class IdentityKoopmanBasisTransform(KoopmanBasisTransform):
    r""" Explicit specialization of :class:`KoopmanBasisTransform` which behaves like the identity. """

    def __init__(self):
        super().__init__(0., 1.)

    def backward_transformation_matrix(self):
        return 1.

    def __call__(self, data, inverse=False, dim=None):
        return data


class KoopmanModel(Model, Transformer):
    r""" Model which contains a finite-dimensional Koopman operator (or approximation thereof).
    It describes the temporal evolution of observable space, i.e.,

    .. math:: \mathbb{E}[g(x_{t+\tau})] = K^\top \mathbb{E}[f(x_t)],

    where :math:`K\in\mathbb{R}^{n\times m}` is the Koopman operator, :math:`x_t` the system's state at time :math:`t`,
    and :math:`f` and :math:`g` observables of the system's state.
    """

    def __init__(self, operator: np.ndarray,
                 basis_transform_forward: Optional[KoopmanBasisTransform],
                 basis_transform_backward: Optional[KoopmanBasisTransform],
                 output_dimension=None):
        r""" Creates a new Koopman model.

        Parameters
        ----------
        operator : (n, n) ndarray
            Applies the transform :math:`K^\top` in the modified basis.
        basis_transform_forward : KoopmanBasisTransform or None
            Transforms the current state :math:`f(x_t)` to the basis in which the Koopman operator is defined.
            If `None`, this defaults to the identity operation.
        basis_transform_backward : KoopmanBasisTransform or None
            Transforms the future state :math:`g(x_t)` to the basis in which the Koopman operator is defined.
            If `None`, this defaults to the identity operation
        """
        super().__init__()
        if basis_transform_forward is None:
            basis_transform_forward = IdentityKoopmanBasisTransform()
        if basis_transform_backward is None:
            basis_transform_backward = IdentityKoopmanBasisTransform()
        self._operator = np.asarray_chkfinite(operator)
        self._basis_transform_forward = basis_transform_forward
        self._basis_transform_backward = basis_transform_backward
        if output_dimension is not None and not is_diagonal_matrix(self.operator):
            raise ValueError("Output dimension can only be set if the Koopman operator is a diagonal matrix. This "
                             "can be achieved through the VAMP estimator.")
        self._output_dimension = output_dimension

    @property
    def operator(self) -> np.ndarray:
        r""" The operator :math:`K` so that :math:`\mathbb{E}[g(x_{t+\tau}] = K^\top \mathbb{E}[f(x_t)]` in transformed
        bases.
        """
        return self._operator

    @cached_property
    def operator_inverse(self) -> np.ndarray:
        r""" Inverse of the operator :math:`K`, i.e., :math:`K^{-1}`. Potentially also pseudo-inverse. """
        return np.linalg.pinv(self.operator)

    @property
    def basis_transform_forward(self) -> KoopmanBasisTransform:
        r"""Transforms the basis of :math:`\mathbb{E}[f(x_t)]` to the one in which the Koopman operator is defined. """
        return self._basis_transform_forward

    @property
    def basis_transform_backward(self) -> KoopmanBasisTransform:
        r"""Transforms the basis of :math:`\mathbb{E}[g(x_{t+\tau})]` to the one in which
        the Koopman operator is defined. """
        return self._basis_transform_backward

    @property
    def output_dimension(self):
        r"""The output dimension of the :meth:`transform` pass."""
        return self._output_dimension

    def forward(self, trajectory: np.ndarray, components: Optional[Union[int, List[int]]] = None) -> np.ndarray:
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
            if not is_diagonal_matrix(self.operator):
                raise ValueError("A subselection of components is only possible if the koopman operator is a diagonal"
                                 "matrix! This can be achieved by using the VAMP estimator, yielding appropriate"
                                 "basis transforms from covariance matrices.")
            operator = np.zeros_like(self.operator)
            if not isinstance(components, (list, tuple)):
                components = [components]
            for ii in components:
                operator[ii, ii] = self.operator[ii, ii]
        else:
            operator = self.operator
        out = self.basis_transform_forward(trajectory)  # (T, N) -> (T, n) project into whitened space
        out = out @ operator  # (T, n) -> (T, n) propagation in whitened space
        out = self.basis_transform_backward(out, inverse=True)  # (T, n) -> (T, N) back into original space
        return out

    def transform(self, data, forward=True, propagate=False, **kwargs):
        r"""Projects the data into the Koopman operator basis, possibly discarding non-relevant dimensions.

        Beware that only `frames[tau:]` of each trajectory returned by this method contain valid values
        of the "future" functions :math:`g(x_{t+\tau})`. Conversely, only `frames[:-tau]` of each
        trajectory returned by this method contain valid values of the "instantaneous" functions :math:`f(x_t)`.
        The remaining frames could possibly be interpreted as some extrapolation.

        Parameters
        ----------
        data : ndarray(T, n)
            the input data
        forward : bool, default=True
            Whether to use forward or backward transform for projection.
        propagate : bool, default=False
            Whether to propagate the projection with :math:`K^\top` (or :math:`(K^\top)^{-1}` in the backward case).

        Returns
        -------
        projection : ndarray(T, m)
            The projected data.
            In case of VAMP, if `forward` is True, projection will be on the right singular
            functions. Otherwise, projection will be on the left singular functions.
        """
        if forward:
            transform = self.basis_transform_forward(data, dim=self.output_dimension)
            return transform @ self.operator if propagate else transform
        else:
            transform = self.basis_transform_backward(data, dim=self.output_dimension)
            return transform @ self.operator_inverse if propagate else transform


class CovarianceKoopmanModel(KoopmanModel):
    r"""A type of Koopman model which was obtained through diagonalization of covariance matrices. This leads to
    a Koopman operator which is a diagonal matrix and can be used to project onto specific processes of the system.

    The estimators which produce this kind of model are :class:`VAMP <deeptime.decomposition.VAMP>` and
    :class:`TICA <deeptime.decomposition.TICA>`."""

    def __init__(self, operator: np.ndarray, basis_transform_forward: Optional[KoopmanBasisTransform],
                 basis_transform_backward: Optional[KoopmanBasisTransform], cov: CovarianceModel,
                 rank_0: int, rank_t: int, dim=None, var_cutoff=None, scaling=None, epsilon=1e-6):
        r""" For a description of parameters `operator`, `basis_transform_forward`, `basis_transform_backward`,
        and `output_dimension`: please see :meth:`KoopmanModel.__init__`.

        Parameters
        ----------
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
        """
        if not is_diagonal_matrix(operator):
            raise ValueError("Koopman operator must be diagonal matrix!")
        output_dim = CovarianceKoopmanModel.effective_output_dimension(rank_0, rank_t, dim, var_cutoff,
                                                                       np.diag(operator))
        super().__init__(operator, basis_transform_forward, basis_transform_backward, output_dimension=output_dim)
        self._cov = cov
        self._scaling = scaling
        self._epsilon = epsilon
        self._dim = dim
        self._var_cutoff = var_cutoff
        self._rank_0 = rank_0
        self._rank_t = rank_t

    @property
    def scaling(self) -> Optional[str]:
        """Scaling of projection. Can be :code:`None`, 'kinetic map', or 'km' """
        return self._scaling

    @property
    def singular_vectors_left(self) -> np.ndarray:
        """Transformation matrix that represents the linear map from mean-free feature space
        to the space of left singular functions."""
        return self.basis_transform_forward.transformation_matrix

    @property
    def singular_vectors_right(self) -> np.ndarray:
        """Transformation matrix that represents the linear map from mean-free feature space
        to the space of right singular functions."""
        return self.basis_transform_backward.transformation_matrix

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
        if value <= 0. or float(value) > 1.0:
            raise ValueError("VAMP: Invalid dimension parameter, if it is given in terms of a floating point, "
                             "can only be in the interval (0, 1].")
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
        if isinstance(value, int):
            if value <= 0:
                # first test against Integral as `isinstance(1, Real)` also evaluates to True
                raise ValueError("VAMP: Invalid dimension parameter, if it is given in terms of the "
                                 "dimension (integer), must be positive.")
        self._dim = value
        self._update_output_dimension()

    @property
    def cumulative_kinetic_variance(self) -> np.ndarray:
        r""" Yields the cumulative kinetic variance. """
        return CovarianceKoopmanModel._cumvar(self.singular_values)

    def score(self, test_model=None, score_method='VAMP2'):
        """Compute the VAMP score for this model or the cross-validation score between self and a second model.

        Parameters
        ----------
        test_model : CovarianceKoopmanModel, optional, default=None

            If `test_model` is not None, this method computes the cross-validation score
            between self and `test_model`. It is assumed that self was estimated from
            the "training" data and `test_model` was estimated from the "test" data. The
            score is computed for one realization of self and `test_model`. Estimation
            of the average cross-validation score and partitioning of data into test and
            training part is not performed by this method.

            If `test_model` is None, this method computes the VAMP score for the model
            contained in self.

        score_method : str, optional, default='VAMP2'
            Available scores are based on the variational approach
            for Markov processes :cite:`vampscore-wu2020variational`:

            *  'VAMP1'  Sum of singular values of the half-weighted Koopman matrix :cite:`vampscore-wu2020variational`.
                        If the model is reversible, this is equal to the sum of
                        Koopman matrix eigenvalues, also called Rayleigh quotient :cite:`vampscore-wu2020variational`.
            *  'VAMP2'  Sum of squared singular values of the half-weighted Koopman
                        matrix :cite:`vampscore-wu2020variational`. If the model is reversible, this is
                        equal to the kinetic variance :cite:`vampscore-noe2015kinetic`.
            *  'VAMPE'  Approximation error of the estimated Koopman operator with respect to
                        the true Koopman operator up to an additive constant :cite:`vampscore-wu2020variational` .

        Returns
        -------
        score : float
            If `test_model` is not None, returns the cross-validation VAMP score between
            self and `test_model`. Otherwise return the selected VAMP-score of self.

        Notes
        -----
        The VAMP-:math:`r` and VAMP-E scores are computed according to :cite:`vampscore-wu2020variational`,
        Equation (33) and Equation (30), respectively.

        References
        ----------
        .. bibliography:: /references.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: vampscore-
        """
        if test_model is None:
            test_model = self
        Uk = self.singular_vectors_left[:, 0:self.output_dimension]
        Vk = self.singular_vectors_right[:, 0:self.output_dimension]
        res = None
        if score_method == 'VAMP1' or score_method == 'VAMP2':
            # see https://arxiv.org/pdf/1707.04659.pdf eqn. (33)
            A = np.atleast_2d(spd_inv_sqrt(Uk.T.dot(test_model.cov_00).dot(Uk), epsilon=self.epsilon))
            B = np.atleast_2d(Uk.T.dot(test_model.cov_0t).dot(Vk))
            C = np.atleast_2d(spd_inv_sqrt(Vk.T.dot(test_model.cov_tt).dot(Vk), epsilon=self.epsilon))
            ABC = np.linalg.multi_dot([A, B, C])
            if score_method == 'VAMP1':
                res = np.linalg.norm(ABC, ord='nuc')
            elif score_method == 'VAMP2':
                res = np.linalg.norm(ABC, ord='fro')**2
        elif score_method == 'VAMPE':
            K = np.diag(self.singular_values[0:self.output_dimension])
            # see https://arxiv.org/pdf/1707.04659.pdf eqn. (30)
            res = np.trace(2.0 * np.linalg.multi_dot([K, Uk.T, test_model.cov_0t, Vk])
                           - np.linalg.multi_dot([K, Uk.T, test_model.cov_00, Uk, K, Vk.T, test_model.cov_tt, Vk]))
        else:
            raise ValueError('"score" should be one of VAMP1, VAMP2 or VAMPE')
        assert res is not None
        # add the contribution (+1) of the constant singular functions to the result
        return res + 1

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

        Notes
        -----
        A "future expectation" of a observable g is the average of g computed
        over a time window that has the same total length as the input data
        from which the Koopman operator was estimated but is shifted
        by lag_multiple*tau time steps into the future (where tau is the lag
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
        observable f and the statistic g at a lag-time of lag_multiple*tau
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

    def timescales(self, lagtime: Optional[int] = None) -> np.ndarray:
        r"""Implied timescales of the TICA transformation

        For each :math:`i`-th eigenvalue, this returns

        .. math::

            t_i = -\frac{\tau}{\log(|\lambda_i|)}

        where :math:`\tau` is the :attr:`lagtime` of the TICA object and :math:`\lambda_i` is the `i`-th
        :attr:`eigenvalue <eigenvalues>` of the TICA object.

        Parameters
        ----------
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
        return - lagtime / np.log(np.abs(self.singular_values))

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
