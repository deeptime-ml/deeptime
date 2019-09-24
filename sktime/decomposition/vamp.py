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
'''
@author: paul, marscher, wu, noe
'''


import warnings

import numpy as np

from sktime.base import Model, Estimator
from sktime.covariance.online_covariance import OnlineCovariance
from sktime.lagged_model_validator import LaggedModelValidator
from sktime.numeric import mdot
from sktime.numeric.eigen import spd_inv_split, spd_inv_sqrt

__all__ = ['VAMP', 'VAMPModel', 'VAMPChapmanKolmogorovValidator']


class VAMPModel(Model):

    def __init__(self, mean_0=None, mean_t=None, cov_00=None, cov_tt=None, cov_0t=None, dim=None, epsilon=1e-6,
                 scaling=None, right=True):
        self.mean_0 = mean_0
        self.mean_t = mean_t
        self.cov_00 = cov_00
        self.cov_tt = cov_tt
        self.cov_0t = cov_0t
        self._svd_performed = False
        self.dim = dim
        self.epsilon = epsilon
        self.scaling = scaling
        self.right = right

    @property
    def scaling(self):
        """Scaling of projection. Can be None or 'kinetic map', 'km' """
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        if value not in (None, 'km', 'kinetic map'):
            raise ValueError('unexpected value (%s) of "scaling". Must be one of ("km", "kinetic map", None).' % value)
        self._scaling = value

    @property
    def singular_vectors_left(self):
        """Tranformation matrix that represents the linear map from mean-free feature space
        to the space of left singular functions."""
        if not self._svd_performed:
            self._diagonalize()
        return self._U

    @property
    def singular_vectors_right(self):
        """Tranformation matrix that represents the linear map from mean-free feature space
        to the space of right singular functions."""
        if not self._svd_performed:
            self._diagonalize()
        return self._V

    @property
    def singular_values(self):
        """The singular values of the half-weighted Koopman matrix"""
        if not self._svd_performed:
            self._diagonalize()
        return self._singular_values

    @property
    def cov_00(self):
        return self._cov_00

    @cov_00.setter
    def cov_00(self, val):
        self._svd_performed = False
        self._cov_00 = val

    @property
    def cov_0t(self):
        return self._cov_0t

    @cov_0t.setter
    def cov_0t(self, val):
        self._svd_performed = False
        self._cov_0t = val

    @property
    def cov_tt(self):
        return self._cov_tt

    @cov_tt.setter
    def cov_tt(self, val):
        self._svd_performed = False
        self._cov_tt = val

    @staticmethod
    def _cumvar(singular_values):
        cumvar = np.cumsum(singular_values ** 2)
        cumvar /= cumvar[-1]
        return cumvar

    @property
    def cumvar(self):
        """ cumulative kinetic variance """
        return VAMPModel._cumvar(self.singular_values)

    @staticmethod
    def _dimension(rank0, rankt, dim, singular_values):
        """ output dimension """
        if dim is None or (isinstance(dim, float) and dim == 1.0):
            return min(rank0, rankt)
        if isinstance(dim, float):
            return np.searchsorted(VAMPModel._cumvar(singular_values), dim) + 1
        else:
            return np.min([rank0, rankt, dim])

    def dimension(self):
        """ output dimension """
        if self.cov_00 is None:  # no data yet
            if isinstance(self.dim, int):  # return user choice
                warnings.warn('Returning user-input for dimension, since this model has not yet been estimated.')
                return self.dim
            raise RuntimeError('This model has not been initialized yet (cov_00 etc.).')

        if not self._svd_performed:
            self._diagonalize()
        return self._dimension(self._rank0, self._rankt, self.dim, self.singular_values)

    def expectation(self, observables, statistics, lag_multiple=1, observables_mean_free=False, statistics_mean_free=False):
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
        dim = self.dimension()

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

    def _diagonalize(self):
        """Performs SVD on covariance matrices and save left, right singular vectors and values in the model.

        Parameters
        ----------
        scaling : None or string, default=None
            Scaling to be applied to the VAMP modes upon transformation
            * None: no scaling will be applied, variance of the singular
              functions is 1
            * 'kinetic map' or 'km': singular functions are scaled by
              singular value. Note that only the left singular functions
              induce a kinetic map.
        """
        L0 = spd_inv_split(self.cov_00, epsilon=self.epsilon)
        self._rank0 = L0.shape[1] if L0.ndim == 2 else 1
        Lt = spd_inv_split(self.cov_tt, epsilon=self.epsilon)
        self._rankt = Lt.shape[1] if Lt.ndim == 2 else 1

        W = np.dot(L0.T, self.cov_0t).dot(Lt)
        from scipy.linalg import svd
        A, s, BT = svd(W, compute_uv=True, lapack_driver='gesvd')

        self._singular_values = s

        # don't pass any values in the argument list that call _diagonalize again!!!
        m = VAMPModel._dimension(self._rank0, self._rankt, self.dim, self._singular_values)

        U = np.dot(L0, A[:, :m])
        V = np.dot(Lt, BT[:m, :].T)

        # scale vectors
        if self.scaling is not None:
            U *= s[np.newaxis, 0:m]  # scaled left singular functions induce a kinetic map
            V *= s[np.newaxis, 0:m]  # scaled right singular functions induce a kinetic map wrt. backward propagator

        self._U = U
        self._V = V
        self._svd_performed = True

    def transform(self, X):
        r"""Projects the data onto the dominant singular functions.

        Parameters
        ----------
        X : ndarray(n, m)
            the input data

        Returns
        -------
        Y : ndarray(n,)
            the projected data
            If `right` is True, projection will be on the right singular
            functions. Otherwise, projection will be on the left singular
            functions.
        """
        # TODO: in principle get_output should not return data for *all* frames!
        if self.right:
            X_meanfree = X - self.mean_t
            Y = np.dot(X_meanfree, self.singular_vectors_right[:, 0:self.dimension()])
        else:
            X_meanfree = X - self.mean_0
            Y = np.dot(X_meanfree, self.singular_vectors_left[:, 0:self.dimension()])

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
        # TODO: implement for TICA too
        if test_model is None:
            test_model = self
        Uk = self.singular_vectors_left[:, 0:self.dimension()]
        Vk = self.singular_vectors_right[:, 0:self.dimension()]
        res = None
        if score_method == 'VAMP1' or score_method == 'VAMP2':
            A = spd_inv_sqrt(Uk.T.dot(test_model.cov_00).dot(Uk))
            B = Uk.T.dot(test_model.cov_0t).dot(Vk)
            C = spd_inv_sqrt(Vk.T.dot(test_model.cov_tt).dot(Vk))
            ABC = mdot(A, B, C)
            if score_method == 'VAMP1':
                res = np.linalg.norm(ABC, ord='nuc')
            elif score_method == 'VAMP2':
                res = np.linalg.norm(ABC, ord='fro')**2
        elif score_method == 'VAMPE':
            Sk = np.diag(self.singular_values[0:self.dimension()])
            res = np.trace(2.0 * mdot(Vk, Sk, Uk.T, test_model.cov_0t) - mdot(Vk, Sk, Uk.T, test_model.cov_00, Uk, Sk, Vk.T, test_model.cov_tt))
        else:
            raise ValueError('"score" should be one of VAMP1, VAMP2 or VAMPE')
        # add the contribution (+1) of the constant singular functions to the result
        assert res
        return res + 1


class VAMP(Estimator):
    r"""Variational approach for Markov processes (VAMP)"""

    def __init__(self, lagtime=1, dim=None, scaling=None, right=False, epsilon=1e-6,
                 ncov=float('inf')):
        r""" Variational approach for Markov processes (VAMP) [1]_.

          Parameters
          ----------
          dim : float or int, default=None
              Number of dimensions to keep:

              * if dim is not set (None) all available ranks are kept:
                  `n_components == min(n_samples, n_uncorrelated_features)`
              * if dim is an integer >= 1, this number specifies the number
                of dimensions to keep.
              * if dim is a float with ``0 < dim < 1``, select the number
                of dimensions such that the amount of kinetic variance
                that needs to be explained is greater than the percentage
                specified by dim.
          scaling : None or string
              Scaling to be applied to the VAMP order parameters upon transformation

              * None: no scaling will be applied, variance of the order parameters is 1
              * 'kinetic map' or 'km': order parameters are scaled by singular value.
                Only the left singular functions induce a kinetic map wrt the
                conventional forward propagator. The right singular functions induce
                a kinetic map wrt the backward propagator.
          right : boolean
              Whether to compute the right singular functions.
              If `right==True`, `get_output()` will return the right singular
              functions. Otherwise, `get_output()` will return the left singular
              functions.
              Beware that only `frames[tau:, :]` of each trajectory returned
              by `get_output()` contain valid values of the right singular
              functions. Conversely, only `frames[0:-tau, :]` of each
              trajectory returned by `get_output()` contain valid values of
              the left singular functions. The remaining frames might
              possibly be interpreted as some extrapolation.
          epsilon : float
              eigenvalue cutoff. Eigenvalues of :math:`C_{00}` and :math:`C_{11}`
              with norms <= epsilon will be cut off. The remaining number of
              eigenvalues together with the value of `dim` define the size of the output.
          ncov : int, default=infinity
              limit the memory usage of the algorithm from [3]_ to an amount that corresponds
              to ncov additional copies of each correlation matrix

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
        self.dim = dim
        self.scaling = scaling
        self.right = right
        self.epsilon = epsilon
        self.ncov = ncov
        self._covar = OnlineCovariance(lagtime=lagtime, compute_c00=True, compute_c0t=True, compute_ctt=True, remove_data_mean=True,
                                       reversible=False, bessels_correction=False, ncov=self.ncov)
        self.lagtime = lagtime
        super(VAMP, self).__init__()

    def _create_model(self) -> VAMPModel:
        return VAMPModel(dim=self.dim, epsilon=self.epsilon, scaling=self.scaling, right=self.right)

    def fit(self, data, **kw):
        self._covar.fit(data, **kw)
        self.fetch_model()
        return self

    def partial_fit(self, X):
        """ incrementally update the covariances and mean.

        Parameters
        ----------
        X: array, list of arrays, PyEMMA reader
            input data.

        Notes
        -----
        The projection matrix is first being calculated upon its first access.
        """
        self._covar.partial_fit(X)
        return self

    def fetch_model(self) -> VAMPModel:
        covar_model = self._covar.fetch_model()

        self._model.cov_00 = covar_model.cov_00
        self._model.cov_0t = covar_model.cov_0t
        self._model.cov_tt = covar_model.cov_tt

        self._model.mean_0 = covar_model.mean_0
        self._model.mean_t = covar_model.mean_t
        self._model._diagonalize()
        return self._model

    @property
    def lagtime(self):
        return self._covar.lagtime

    @lagtime.setter
    def lagtime(self, value):
        self._covar.lagtime = value


class VAMPChapmanKolmogorovValidator(LaggedModelValidator):

    def __init__(self, test_model, test_estimator, observables, statistics,
                 observables_mean_free, statistics_mean_free, mlags=10):
        r"""
        Note
        ----
        It is recommended that you create this object by calling the
        `cktest` method of a VAMP object created with
        :func:`vamp <pyemma.coordinates.vamp>`.

        Parameters
        ----------
        test_model : Model
         Model with the smallest lag time. Is used to make predictions
         for larger lag times.

        test_estimator : Estimator
         Parametrized Estimator that has produced the model.
         Is used as a prototype for estimating models at higher lag times.

        observables : np.ndarray((input_dimension, n_observables))
         Coefficients that express one or multiple observables in
         the basis of the input features.

        statistics : np.ndarray((input_dimension, n_statistics))
         Coefficients that express one or multiple statistics in
         the basis of the input features.

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

        mlags : int or int-array, default=10
         multiples of lag times for testing the Model, e.g. range(10).
         A single int will trigger a range, i.e. mlags=10 maps to
         mlags=range(10).
         Note that you need to be able to do a model prediction for each
         of these lag time multiples, e.g. the value 0 only make sense
         if model.expectation(lag_multiple=0) will work.

        Notes
        -----
        The object can be plotted with :func:`plot_cktest <pyemma.plots.plot_cktest>`
        with the option `y01=False`.
        """
        super(VAMPChapmanKolmogorovValidator, self).__init__(test_model, test_estimator, mlags=mlags)
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

    def _compute_observables(self, model: VAMPModel, mlag=1):
        return model.expectation(statistics=self.statistics, observables=self.observables, lag_multiple=mlag,
                                 statistics_mean_free=self.statistics_mean_free,
                                 observables_mean_free=self.observables_mean_free)

    def _compute_observables_conf(self, model, mlag=1, conf=0.95):
        raise NotImplementedError('estimation of confidence intervals not yet implemented for VAMP')


def vamp_cktest(test_estimator, model, n_observables=None, observables='phi', statistics='psi', mlags=10, data=None):
    r"""Do the Chapman-Kolmogorov test by computing predictions for higher lag times and by performing estimations at higher lag times.

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
      parameter of :func:`vamp <pyemma.coordinates.vamp>`.

    * predictions at higher lag times :

      .. math::

          \left\langle \mathbf{K}^{n}(\tau)g_{i},f_{j}\right\rangle_{\rho_{0}}

      where :math:`\mathbf{K}^{n}` is the n'th power of the rank-reduced
      Koopman matrix contained in self.


    The Champan-Kolmogorov test is to compare the predictions to the
    estimates.

    Parameters
    ----------
    n_observables : int, optional, default=None
        Limit the number of default observables (and of default statistics)
        to this number.
        Only used if `observables` are None or `statistics` are None.

    observables : np.ndarray((input_dimension, n_observables)) or 'phi'
        Coefficients that express one or multiple observables :math:`g`
        in the basis of the input features.
        This parameter can be 'phi'. In that case, the dominant
        right singular functions of the Koopman operator estimated
        at the smallest lag time are used as default observables.

    statistics : np.ndarray((input_dimension, n_statistics)) or 'psi'
        Coefficients that express one or multiple statistics :math:`f`
        in the basis of the input features.
        This parameter can be 'psi'. In that case, the dominant
        left singular functions of the Koopman operator estimated
        at the smallest lag time are used as default statistics.

    mlags : int or int-array, default=10
        multiples of lag times for testing the Model, e.g. range(10).
        A single int will trigger a range, i.e. mlags=10 maps to
        mlags=range(10).
        Note that you need to be able to do a model prediction for each
        of these lag time multiples, e.g. the value 0 only make sense
        if model.expectation(lag_multiple=0) will work.

    Returns
    -------
    vckv : :class:`VAMPChapmanKolmogorovValidator <pyemma.coordinates.transform.VAMPChapmanKolmogorovValidator>`
        Contains the estimated and the predicted covarince matrices.
        The object can be plotted with :func:`plot_cktest <pyemma.plots.plot_cktest>` with the option `y01=False`.
    """
    if n_observables is not None:
        if n_observables > model.dimension():
            warnings.warn('Selected singular functions as observables but dimension '
                          'is lower than requested number of observables.')
            n_observables = model.dimension()
    else:
        n_observables = model.dimension()

    if isinstance(observables, str) and observables == 'phi':
        observables = model.singular_vectors_right[:, 0:n_observables]
        observables_mean_free = True
    else:
        #ensure_ndarray(observables, ndim=2)
        observables_mean_free = False

    if isinstance(statistics, str) and statistics == 'psi':
        statistics = model.singular_vectors_left[:, 0:n_observables]
        statistics_mean_free = True
    else:
        #ensure_ndarray_or_None(statistics, ndim=2)
        statistics_mean_free = False
    ck = VAMPChapmanKolmogorovValidator(model, test_estimator, observables, statistics, observables_mean_free,
                                        statistics_mean_free, mlags=mlags)
    if data is not None:
        ck.fit(data)
    return ck
