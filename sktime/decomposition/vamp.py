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
from typing import Optional, Union, List

import numpy as np

from .koopman import CovarianceKoopmanModel, KoopmanBasisTransform
from ..base import Estimator, Transformer
from ..covariance.covariance import Covariance, CovarianceModel
from ..numeric.eigen import spd_inv_split

__all__ = ['VAMP']


class VAMP(Estimator, Transformer):
    r"""Variational approach for Markov processes (VAMP).

    The implementation is based on :cite:`vamp-wu2020variational`, :cite:`vamp-noe2015kinetic`.

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

      \mathbf{C}_{00}	:=\frac{1}{T-\tau}\sum_{t=0}^{T-\tau}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{0}\right]\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{0}\right]

      \mathbf{C}_{11}	:=\frac{1}{T-\tau}\sum_{t=\tau}^{T}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{1}\right]\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{1}\right]

      \mathbf{C}_{01}	:=\frac{1}{T-\tau}\sum_{t=0}^{T-\tau}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{0}\right]\left[\boldsymbol{\chi}(t+\tau)-\boldsymbol{\mu}_{1}\right]

    The Koopman matrix is then computed as follows:

    .. math::

      \mathbf{K}=\mathbf{C}_{00}^{-1}\mathbf{C}_{01}

    It can be shown :cite:`vamp-wu2020variational` that the leading singular functions of the
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
    .. bibliography:: /references.bib
        :style: unsrt
        :filter: docname in docnames
        :keyprefix: vamp-
    """

    def __init__(self, dim: Optional[Union[int, float]] = None, scaling: Optional[str] = None,
                 epsilon: float = 1e-6):
        r""" Creates a new VAMP estimator.

        Parameters
        ----------
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
        epsilon : float, optional, default=1e-6
            Eigenvalue cutoff. Eigenvalues of :math:`C_{00}` and :math:`C_{11}`
            with norms <= epsilon will be cut off. The remaining number of
            eigenvalues together with the value of `dim` define the size of the output.
        """
        super(VAMP, self).__init__()
        self.dim = dim
        self.scaling = scaling
        self.epsilon = epsilon

    _DiagonalizationResults = namedtuple("DiagonalizationResults", ['rank0', 'rankt', 'singular_values',
                                                                    'left_singular_vecs', 'right_singular_vecs'])

    @staticmethod
    def _cumvar(singular_values) -> np.ndarray:
        cumvar = np.cumsum(singular_values ** 2)
        cumvar /= cumvar[-1]
        return cumvar

    @staticmethod
    def _decomposition(covariances, epsilon, scaling, dim) -> _DiagonalizationResults:
        """Performs SVD on covariance matrices and save left, right singular vectors and values in the model."""
        L0 = spd_inv_split(covariances.cov_00, epsilon=epsilon)
        rank0 = L0.shape[1] if L0.ndim == 2 else 1
        Lt = spd_inv_split(covariances.cov_tt, epsilon=epsilon)
        rankt = Lt.shape[1] if Lt.ndim == 2 else 1

        W = np.dot(L0.T, covariances.cov_0t).dot(Lt)
        from scipy.linalg import svd
        A, s, BT = svd(W, compute_uv=True, lapack_driver='gesvd')

        singular_values = s

        m = CovarianceKoopmanModel.effective_output_dimension(rank0, rankt, dim, singular_values)

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
            Limit the memory usage of the algorithm from :cite:`vamp-chan1982updating` to an amount that corresponds
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
        covariances = self._to_covariance_model(covariances)
        self._model = self._decompose(covariances)
        return self

    def fit_from_timeseries(self, data: Union[np.ndarray, List[np.ndarray]], lagtime: int,
                            weights=None):
        r""" Estimates a :class:`CovarianceKoopmanModel` directly from time-series data using the :class:`Covariance`
        estimator. For parameters `dim`, `scaling`, `epsilon`.

        Parameters
        ----------
        data : (T, n) ndarray or list thereof
            Input time-series.
        lagtime : int
            Lagtime for covariance matrix estimation, must be positive.
        weights
            See the :class:`Covariance <sktime.covariance.Covariance>` estimator.

        Returns
        -------
        self : VAMP
            Reference to self.
        """
        covariance_estimator = self.covariance_estimator(lagtime=lagtime)
        covariances = covariance_estimator.fit(data, weights=weights).fetch_model()
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
    def dim(self) -> Optional[Union[int, float]]:
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
    def dim(self, value: Optional[Union[int, float]]):
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

    def _decompose(self, covariances: CovarianceModel):
        decomposition = self._decomposition(covariances, self.epsilon, self.scaling, self.dim)
        return CovarianceKoopmanModel(
            operator=np.diag(decomposition.singular_values),
            basis_transform_forward=KoopmanBasisTransform(covariances.mean_0, decomposition.left_singular_vecs),
            basis_transform_backward=KoopmanBasisTransform(covariances.mean_t, decomposition.right_singular_vecs),
            rank_0=decomposition.rank0, rank_t=decomposition.rankt,
            dim=self.dim, cov=covariances, scaling=self.scaling, epsilon=self.epsilon
        )

    def fit(self, data, *args, **kw):
        r""" Fits a new :class:`CovarianceKoopmanModel` which can be obtained by a
        subsequent call to :meth:`fetch_model`.

        Parameters
        ----------
        data : CovarianceModel or Covariance or timeseries
            Covariance matrices :math:`C_{00}, C_{0t}, C_{tt}` in form of a CovarianceModel instance. If the model
            should be fitted directly from data, please see :meth:`from_data`.
            Optionally, this can also be timeseries data directly, in which case the keyword argument 'lagtime'
            must be provided.
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
            if 'lagtime' not in kw.keys():
                raise ValueError("Cannot fit on timeseries data without a lagtime!")
            self.fit_from_timeseries(data, lagtime=kw.pop('lagtime'), weights=kw.pop('weights', None))
        return self

    def transform(self, data, forward=True):
        r""" Projects given timeseries onto dominant singular functions. This method dispatches to
        :meth:`CovarianceKoopmanModel.transform`.

        Parameters
        ----------
        data : (T, n) ndarray
            Input timeseries data.
        forward : bool, default=True
            Whether to use left or right eigenvectors for projection. Left corresponds to `forward == True`, right
            corresponds to `forward == False`.

        Returns
        -------
        Y : (T, m) ndarray
            The projected data.
            If `right` is True, projection will be on the right singular functions. Otherwise, projection will be on
            the left singular functions.
        """
        return self.fetch_model().transform(data, forward=forward)

    def fetch_model(self) -> CovarianceKoopmanModel:
        r""" Finalizes current model and yields new :class:`CovarianceKoopmanModel`.

        Returns
        -------
        model : CovarianceKoopmanModel
            The estimated model.
        """
        return self._model
