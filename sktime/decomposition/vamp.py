# This file is part of PyEMMA.
#
# Copyright (c) 2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
@author: paul, marscher, wu, noe
"""


from collections import namedtuple
from numbers import Real, Integral
from typing import Optional, Union

import numpy as np

from ..base import Model, Estimator, Transformer
from ..covariance.covariance import Covariance, CovarianceModel
from ..numeric import mdot
from ..numeric.eigen import spd_inv_split, spd_inv_sqrt
from ..util import cached_property

__all__ = ['VAMP', 'VAMPModel']


class VAMPModel(Model, Transformer):
    r""" Model which was estimated from a :class:`VAMP` estimator.

    See Also
    --------
    VAMP : vamp estimator
    """

    _DiagonalizationResults = namedtuple("DiagonalizationResults", ['rank0', 'rankt', 'singular_values',
                                                                    'left_singular_vecs', 'right_singular_vecs'])

    def __init__(self, cov: CovarianceModel, dim=None, epsilon=1e-6, scaling=None, right=False):
        r""" Creates a new model instance.

        Parameters
        ----------
        cov : CovarianceModel
            Estimated covariances.
        dim : int or float or None, optional, default=None
            Dimension parameter, see :attr:`VAMP.dim` for a more detailed description. The effective dimension can be
            obtained from :attr:`output_dimension`.
        epsilon : float, optional, default=1e-6
            Singular value cutoff parameter, see :attr:`VAMP.epsilon`.
        scaling : str or None, optional, default=None
            Scaling parameter, see :attr:`VAMP.scaling`.
        right : bool, optional, default=False
            Whether right or left eigenvectors should be used for projection.
        """
        super().__init__()
        self._cov = cov
        self._dim = dim
        self._epsilon = epsilon
        self._scaling = scaling
        self._right = right

    def _invalidate_caches(self):
        r""" Invalidates all cached properties and causes them to be re-evaluated """
        for member in self.__class__.__dict__.values():
            if isinstance(member, cached_property):
                member.invalidate()
        self._eigenvalues = None
        self._stationary_distribution = None

    @property
    def right(self) -> bool:
        r""" Whether right or left eigenvectors should be used for projection. """
        return self._right

    @right.setter
    def right(self, value: bool):
        self._right = value

    @property
    def epsilon(self) -> float:
        r""" Singular value cutoff. """
        return self._epsilon

    @property
    def dim(self) -> Optional[Real]:
        r""" Dimension parameter used for estimating this model. Affects the effective :attr:`output_dimension`.

        :getter: Yields the currently configured dim parameter.
        :setter: Sets a new dimension cutoff, triggers re-evaluation of decomposition.
        :type: int or float or None
        """
        return self._dim

    @dim.setter
    def dim(self, value: Optional[Real]):
        if isinstance(value, Integral) and value <= 0:
            # first test against Integral as `isinstance(1, Real)` also evaluates to True
            raise ValueError("Invalid dimension parameter, if it is given in terms of the dimension (integer), "
                             "must be positive.")
        elif isinstance(value, Real) and (value <= 0. or float(value) > 1.0):
            raise ValueError("Invalid dimension parameter, if it is given in terms of a floating point, "
                             "can only be in the interval (0, 1].")
        elif value is not None and not isinstance(value, (Integral, Real)):
            raise ValueError("Invalid type for dimension, got {}".format(value))
        self._dim = value
        self._invalidate_caches()

    @property
    def scaling(self) -> Optional[str]:
        """Scaling of projection. Can be :code:`None`, 'kinetic map', or 'km' """
        return self._scaling

    @property
    def singular_vectors_left(self) -> np.ndarray:
        """Tranformation matrix that represents the linear map from mean-free feature space
        to the space of left singular functions."""
        return self._decomposition.left_singular_vecs

    @property
    def singular_vectors_right(self) -> np.ndarray:
        """Tranformation matrix that represents the linear map from mean-free feature space
        to the space of right singular functions."""
        return self._decomposition.right_singular_vecs

    @property
    def singular_values(self) -> np.ndarray:
        """ The singular values of the half-weighted Koopman matrix. """
        return self._decomposition.singular_values

    @property
    def cov(self) -> CovarianceModel:
        r""" Estimated covariances. """
        return self._cov

    @property
    def mean_0(self) -> np.ndarray:
        r""" Shortcut to :attr:`mean_0 <sktime.covariance.CovarianceModel.mean_0>`. """
        return self.cov.mean_0

    @property
    def mean_t(self) -> np.ndarray:
        r""" Shortcut to :attr:`mean_t <sktime.covariance.CovarianceModel.mean_t>`. """
        return self.cov.mean_t

    @property
    def cov_00(self) -> np.ndarray:
        r""" Shortcut to :attr:`cov_00 <sktime.covariance.CovarianceModel.cov_00>`. """
        return self.cov.cov_00

    @property
    def cov_0t(self) -> np.ndarray:
        r""" Shortcut to :attr:`cov_0t <sktime.covariance.CovarianceModel.cov_0t>`. """
        return self.cov.cov_0t

    @property
    def cov_tt(self) -> np.ndarray:
        r""" Shortcut to :attr:`cov_tt <sktime.covariance.CovarianceModel.cov_tt>`. """
        return self.cov.cov_tt

    @staticmethod
    def _cumvar(singular_values) -> np.ndarray:
        cumvar = np.cumsum(singular_values ** 2)
        cumvar /= cumvar[-1]
        return cumvar

    @property
    def cumvar(self) -> np.ndarray:
        """ Cumulative kinetic variance. """
        return VAMPModel._cumvar(self.singular_values)

    @staticmethod
    def _dimension(rank0, rankt, dim, singular_values) -> int:
        """ output dimension """
        if dim is None or (isinstance(dim, float) and dim == 1.0):
            return min(rank0, rankt)
        if isinstance(dim, float):
            return np.searchsorted(VAMPModel._cumvar(singular_values), dim) + 1
        else:
            return np.min([rank0, rankt, dim])

    @property
    def output_dimension(self) -> int:
        """ Effective Output dimension. """
        rank0 = self._decomposition.rank0
        rankt = self._decomposition.rankt
        return self._dimension(rank0, rankt, self.dim, self.singular_values)

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

        if statistics is not None:
            # compute lagged covariance
            return Q.dot(P).dot(R.T)
            # TODO: discuss whether we want to return this or the transpose
            # TODO: from MSMs one might expect to first index to refer to the statistics, here it is the other way round
        else:
            # compute future expectation
            return Q.dot(P)[:, 0]

    @cached_property
    def _decomposition(self) -> _DiagonalizationResults:
        """Performs SVD on covariance matrices and save left, right singular vectors and values in the model."""
        L0 = spd_inv_split(self.cov_00, epsilon=self.epsilon)
        rank0 = L0.shape[1] if L0.ndim == 2 else 1
        Lt = spd_inv_split(self.cov_tt, epsilon=self.epsilon)
        rankt = Lt.shape[1] if Lt.ndim == 2 else 1

        W = np.dot(L0.T, self.cov_0t).dot(Lt)
        from scipy.linalg import svd
        A, s, BT = svd(W, compute_uv=True, lapack_driver='gesvd')

        singular_values = s

        # don't pass any values in the argument list that call _diagonalize again!!!
        m = VAMPModel._dimension(rank0, rankt, self.dim, singular_values)

        U = np.dot(L0, A[:, :m])
        V = np.dot(Lt, BT[:m, :].T)

        # scale vectors
        if self.scaling is not None:
            U *= s[np.newaxis, 0:m]  # scaled left singular functions induce a kinetic map
            V *= s[np.newaxis, 0:m]  # scaled right singular functions induce a kinetic map wrt. backward propagator

        return VAMPModel._DiagonalizationResults(
            rank0=rank0, rankt=rankt, singular_values=singular_values, left_singular_vecs=U, right_singular_vecs=V
        )

    def forward(self, trajectory: np.ndarray, component: Optional[int] = None):
        r"""Applies the forward transform to the trajectory in non-whitened space. This is achieved by
        transforming each frame :math:`X_t` with

        .. math::
            \hat{X}_{t+\tau} = (V^\top)^{-1} \Sigma U^\top (X_t - \mu_0) + \mu_t,

        where :math:`V` are the left singular vectors, :math:`\Sigma` the singular values, and :math:`U` the right
        singular vectors.

        Parameters
        ----------
        trajectory : (T, n) ndarray
            The input trajectory
        component : int or None
            The component to project onto. If None, all processes are taken into account, if integer sets all singular
            values to zero but the "component"th one.

        Returns
        -------
        predictions : (T, n) ndarray
            The predicted trajectory.
        """
        if component is not None:
            singval = np.zeros((len(self.singular_values), len(self.singular_values)))
            singval[component, component] = self.singular_values[component]
        else:
            singval = np.diag(self.singular_values)

        output_trajectory = np.empty_like(trajectory)
        VT_inv = np.linalg.pinv(self.singular_vectors_right.T)
        for t, frame in enumerate(trajectory):
            output_trajectory[t] = VT_inv @ singval @ np.dot(self.singular_vectors_left.T, frame - self.mean_0) + self.mean_t

        return output_trajectory

    def transform(self, data, right=None):
        r"""Projects the data onto the dominant singular functions.

        Parameters
        ----------
        data : ndarray(n, m)
            the input data
        right : bool or None, optional, default=None
            Whether to use left or right eigenvectors for projection, overrides :attr:`right` if not None.

        Returns
        -------
        Y : ndarray(n,)
            the projected data
            If `right` is True, projection will be on the right singular
            functions. Otherwise, projection will be on the left singular
            functions.
        """
        right = self.right if right is None else right
        if right:
            X_meanfree = data - self.mean_t
            Y = np.dot(X_meanfree, self.singular_vectors_right[:, :self.output_dimension])
        else:
            X_meanfree = data - self.mean_0
            Y = np.dot(X_meanfree, self.singular_vectors_left[:, :self.output_dimension])
        return Y

    def score(self, test_model=None, score_method='VAMP2'):
        """Compute the VAMP score for this model or the cross-validation score between self and a second model.

        Parameters
        ----------
        test_model : VAMPModel, optional, default=None

            If `test_model` is not None, this method computes the cross-validation score
            between self and `test_model`. It is assumed that self was estimated from
            the "training" data and `test_model` was estimated from the "test" data. The
            score is computed for one realization of self and `test_model`. Estimation
            of the average cross-validation score and partitioning of data into test and
            training part is not performed by this method.

            If `test_model` is None, this method computes the VAMP score for the model
            contained in self.

        score_method : str, optional, default='VAMP2'
            Available scores are based on the variational approach for Markov processes [1]_:

            *  'VAMP1'  Sum of singular values of the half-weighted Koopman matrix [1]_ .
                        If the model is reversible, this is equal to the sum of
                        Koopman matrix eigenvalues, also called Rayleigh quotient [1]_.
            *  'VAMP2'  Sum of squared singular values of the half-weighted Koopman matrix [1]_ .
                        If the model is reversible, this is equal to the kinetic variance [2]_ .
            *  'VAMPE'  Approximation error of the estimated Koopman operator with respect to
                        the true Koopman operator up to an additive constant [1]_ .

        Returns
        -------
        score : float
            If `test_model` is not None, returns the cross-validation VAMP score between
            self and `test_model`. Otherwise return the selected VAMP-score of self.

        References
        ----------
        .. [1] Wu, H. and Noe, F. 2017. Variational approach for learning Markov processes from time series data.
            arXiv:1707.04659v1
        .. [2] Noe, F. and Clementi, C. 2015. Kinetic distance and kinetic maps from molecular dynamics simulation.
            J. Chem. Theory. Comput. doi:10.1021/acs.jctc.5b00553
        """
        if test_model is None:
            test_model = self
        Uk = self.singular_vectors_left[:, 0:self.output_dimension]
        Vk = self.singular_vectors_right[:, 0:self.output_dimension]
        res = None
        if score_method == 'VAMP1' or score_method == 'VAMP2':
            A = spd_inv_sqrt(Uk.T.dot(test_model.cov_00).dot(Uk), epsilon=self.epsilon)
            B = Uk.T.dot(test_model.cov_0t).dot(Vk)
            C = spd_inv_sqrt(Vk.T.dot(test_model.cov_tt).dot(Vk), epsilon=self.epsilon)
            ABC = mdot(A, B, C)
            if score_method == 'VAMP1':
                res = np.linalg.norm(ABC, ord='nuc')
            elif score_method == 'VAMP2':
                res = np.linalg.norm(ABC, ord='fro')**2
        elif score_method == 'VAMPE':
            Sk = np.diag(self.singular_values[0:self.output_dimension])
            res = np.trace(2.0 * mdot(Vk, Sk, Uk.T, test_model.cov_0t)
                           - mdot(Vk, Sk, Uk.T, test_model.cov_00, Uk, Sk, Vk.T, test_model.cov_tt))
        else:
            raise ValueError('"score" should be one of VAMP1, VAMP2 or VAMPE')
        assert res is not None
        # add the contribution (+1) of the constant singular functions to the result
        return res + 1


class VAMP(Estimator, Transformer):
    r"""Variational approach for Markov processes (VAMP).

    The implementation is based on [1]_, [2]_.

    See Also
    --------
    VAMPModel : type of model produced by this estimator

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

      \mathbf{C}_{00}	:=\frac{1}{T-\tau}\sum_{t=0}^{T-\tau}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{0}\right]\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{0}\right]

      \mathbf{C}_{11}	:=\frac{1}{T-\tau}\sum_{t=\tau}^{T}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{1}\right]\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{1}\right]

      \mathbf{C}_{01}	:=\frac{1}{T-\tau}\sum_{t=0}^{T-\tau}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{0}\right]\left[\boldsymbol{\chi}(t+\tau)-\boldsymbol{\mu}_{1}\right]

    The Koopman matrix is then computed as follows:

    .. math::

      \mathbf{K}=\mathbf{C}_{00}^{-1}\mathbf{C}_{01}

    It can be shown [1]_ that the leading singular functions of the
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

      \boldsymbol{\psi}(t):=\mathbf{U}^{\prime\top}\mathbf{C}_{00}^{-\frac{1}{2}}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{0}\right]

      \boldsymbol{\phi}(t):=\mathbf{V}^{\prime\top}\mathbf{C}_{11}^{-\frac{1}{2}}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{1}\right]


    References
    ----------
    .. [1] Wu, H. and Noe, F. 2017. Variational approach for learning Markov processes from time series data.
      arXiv:1707.04659v1
    .. [2] Noe, F. and Clementi, C. 2015. Kinetic distance and kinetic maps from molecular dynamics simulation.
      J. Chem. Theory. Comput. doi:10.1021/acs.jctc.5b00553
    .. [3] Chan, T. F., Golub G. H., LeVeque R. J. 1979. Updating formulae and pairwise algorithms for
     computing sample variances. Technical Report STAN-CS-79-773, Department of Computer Science, Stanford University.
    """

    def __init__(self, lagtime: int, dim: Optional[Real] = None, scaling: Optional[str] = None, right: bool = False,
                 epsilon: float = 1e-6, ncov: Union[int] = float('inf')):
        r""" Creates a new VAMP estimator.

        Parameters
        ----------
        lagtime : int
            The lagtime to be used for estimating covariances.
        dim : float or int, optional, default=None
            Number of dimensions to keep:

            * if dim is not set (None) all available ranks are kept:
              :code:`n_components == min(n_samples, n_uncorrelated_features)`
            * if dim is an integer >= 1, this number specifies the number
              of dimensions to keep.
            * if dim is a float with ``0 < dim <= 1``, select the number
              of dimensions such that the amount of kinetic variance
              that needs to be explained is greater than the percentage
              specified by dim.
        scaling : str, optional, default=None
            Scaling to be applied to the VAMP order parameters upon transformation

            * None: no scaling will be applied, variance of the order parameters is 1
            * 'kinetic_map' or 'km': order parameters are scaled by singular value.
              Only the left singular functions induce a kinetic map wrt the
              conventional forward propagator. The right singular functions induce
              a kinetic map wrt the backward propagator.
        right: bool, optional, default=None
            Whether to compute the right singular functions.

            If :code:`right==True`, :meth:`transform` and :meth:`VAMPModel.transform` will use the right singular
            functions.

            Beware that only `frames[tau:, :]` of each trajectory returned
            by :meth:`transform` contain valid values of the right singular
            functions. Conversely, only `frames[0:-tau, :]` of each
            trajectory returned by :meth:`transform` contain valid values of
            the left singular functions. The remaining frames might
            possibly be interpreted as some extrapolation.
        epsilon : float, optional, default=1e-6
            Eigenvalue cutoff. Eigenvalues of :math:`C_{00}` and :math:`C_{11}`
            with norms <= epsilon will be cut off. The remaining number of
            eigenvalues together with the value of `dim` define the size of the output.
        ncov : int or float('inf'), optional, default=float('inf')
            Limit the memory usage of the algorithm from [3]_ to an amount that corresponds
            to ncov additional copies of each correlation matrix.
        """
        super(VAMP, self).__init__()
        self.dim = dim
        self.scaling = scaling
        self.right = right
        self.epsilon = epsilon
        self.ncov = ncov
        self._covar = Covariance(lagtime=lagtime, compute_c00=True, compute_c0t=True, compute_ctt=True,
                                 remove_data_mean=True, reversible=False, bessels_correction=False,
                                 ncov=self.ncov)
        self.lagtime = lagtime

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
    def right(self) -> bool:
        r""" Whether to use right singular functions instead of left singular functions.

        :type: bool
        """
        return self._right

    @right.setter
    def right(self, value: bool):
        self._right = value

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
    def dim(self) -> Optional[Real]:
        r""" Dimension attribute. Can either be int or float. In case of

        * :code:`int` it evaluates it as the actual dimension, must be strictly greater 0,
        * :code:`float` it evaluates it as percentage of the captured kinetic variance, i.e., must be in :math:`(0,1]`,
        * :code:`None` all components are used.

        :getter: yields the dimension
        :setter: sets a new dimension
        :type: int or float
        """
        return self._dim

    @dim.setter
    def dim(self, value: Optional[Real]):
        if isinstance(value, Integral):
            if value <= 0:
                # first test against Integral as `isinstance(1, Real)` also evaluates to True
                raise ValueError("VAMP: Invalid dimension parameter, if it is given in terms of the "
                                 "dimension (integer), must be positive.")
        elif isinstance(value, Real) and (value <= 0. or float(value) > 1.0):
            raise ValueError("VAMP: Invalid dimension parameter, if it is given in terms of a floating point, "
                             "can only be in the interval (0, 1].")
        elif value is not None and not isinstance(value, (Integral, Real)):
            raise ValueError("Invalid type for dimension, got {}".format(value))
        self._dim = value

    def fit(self, data, **kw):
        r""" Fits a new :class:`VAMPModel` which can be obtained by a subsequent call to :meth:`fetch_model`.

        Parameters
        ----------
        data : (T, d) ndarray
            Input timeseries. If presented with multiple trajectories, :meth:`partial_fit` should be used.
        **kw
            Ignored keyword arguments for scikit-learn compatibility.

        Returns
        -------
        self : VAMP
            Reference to self.
        """
        self._covar.fit(data, **kw)
        return self

    def transform(self, data, right=None):
        r""" Projects given timeseries onto dominant singular functions. This method dispatches to
        :meth:`VAMPModel.transform`.

        Parameters
        ----------
        data : (T, n) ndarray
            Input timeseries data.
        right : bool or None, optional, default=None
            Whether to use left or right eigenvectors for projection, overrides :attr:`right` if not None.

        Returns
        -------
        Y : (T, m) ndarray
            The projected data.
            If `right` is True, projection will be on the right singular functions. Otherwise, projection will be on
            the left singular functions.
        """
        return self.fetch_model().transform(data, right=right)

    def partial_fit(self, data):
        """ Incrementally update the covariances and mean.

        Parameters
        ----------
        data: (T, d) ndarray
            input data.

        Returns
        -------
        self : VAMP
            Reference to self.
        """
        self._covar.partial_fit(data)
        return self

    def fetch_model(self) -> VAMPModel:
        r""" Finalizes current model and yields new :class:`VAMPModel`.

        Returns
        -------
        model : VAMPModel
            The estimated model.
        """
        covar_model = self._covar.fetch_model()
        return VAMPModel(cov=covar_model, dim=self.dim, epsilon=self.epsilon, scaling=self.scaling, right=self.right)

    @property
    def lagtime(self) -> int:
        r""" Lagtime at which to compute covariances.

        :getter: Yields currently configured lagtime.
        :setter: Configures a new lagtime.
        :type: int
        """
        return self._covar.lagtime

    @lagtime.setter
    def lagtime(self, value: int):
        self._covar.lagtime = value
