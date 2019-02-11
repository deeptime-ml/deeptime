from sktime.base import Model, Estimator
from sktime.covariance.online_covariance import OnlineCovarianceModel, OnlineCovariance
from sktime.numeric.eigen import eig_corr

import numpy as np

__author__ = 'marscher'


# TODO: would it make sense to extend the Covariance model or just make a composition?
class TICAModelBase(Model):

    def __init__(self, mean=None, cov=None, cov_tau=None, dim=None, epsilon=1e-6, scaling=None, lag=0):
        self.set_model_params(mean=mean, cov=cov, cov_tau=cov_tau, dim=dim, epsilon=epsilon, scaling=scaling, lag=lag)

    # TODO: do we want to keep constructor specification of model parameters?
    def set_model_params(self, mean=None, cov_tau=None, cov=None,
                         dim=0.95,
                         epsilon=1e-6,
                         scaling='kinetic_map',
                         lag=0,
                         ):
        self.cov = cov
        self.cov_tau = cov_tau
        self.mean = mean
        self.dim = dim
        self.epsilon = epsilon
        self.lag = lag
        self.scaling = scaling

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        valid = ('kinetic_map', 'commute_map', None)
        if value not in valid:
            raise ValueError('Valid settings for scaling are one of {valid}, but was {invalid}'
                             .format(valid=valid, invalid=value))
        self._scaling = value

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, value):
        self._diagonalized = False
        self._cov = value

    @property
    def cov_tau(self):
        return self._cov_tau

    @cov_tau.setter
    def cov_tau(self, value):
        self._diagonalized = False
        self._cov_tau = value

    @property
    def eigenvectors(self):
        r""" Eigenvectors of the TICA problem, columnwise

        Returns
        -------
        eigenvectors: (N,M) ndarray
        """
        if not self._diagonalized:
            self._diagonalize()
        return self._eigenvectors

    @property
    def eigenvalues(self):
        r""" Eigenvalues of the TICA problem (usually denoted :math:`\lambda`)

        Returns
        -------
        eigenvalues: 1D np.array
        """
        if not self._diagonalized:
            self._diagonalize()
        return self._eigenvalues

    @staticmethod
    def _cumvar(eigenvalues):
        cumvar = np.cumsum(eigenvalues ** 2)
        cumvar /= cumvar[-1]
        return cumvar

    @property
    def cumvar(self):
        r""" Cumulative sum of the the TICA eigenvalues

        Returns
        -------
        cumvar: 1D np.array
        """
        return TICAModelBase._cumvar(self.eigenvalues)

    @staticmethod
    def _dimension(rank, dim, eigenvalues):
        """ output dimension """
        if dim is None or (isinstance(dim, float) and dim == 1.0):
            return rank
        if isinstance(dim, float):
            # subspace_variance, reduce the output dimension if needed
            return min(len(eigenvalues), np.searchsorted(TICAModelBase._cumvar(eigenvalues), dim) + 1)
        else:
            return np.min([rank, dim])

    def dimension(self):
        """ output dimension """
        if self.cov is None:  # no data yet
            if isinstance(self.dim, int):  # return user choice
                import warnings
                warnings.warn('Returning user-input for dimension, since this model has not yet been estimated.')
                return self.dim
            raise RuntimeError('Please call set_model_params prior using this method.')

        if not self._diagonalized:
            self._diagonalize()
        return self._dimension(self._rank, self.dim, self.eigenvalues)

    def _compute_diag(self):
        from sktime.numeric.eigen import ZeroRankError
        try:
            eigenvalues, eigenvectors, rank = eig_corr(self.cov, self.cov_tau, self.epsilon,
                                                       sign_maxelement=True, return_rank=True)
        except ZeroRankError:
            raise ZeroRankError('All input features are constant in all time steps. '
                                'No dimension would be left after dimension reduction.')
        return eigenvalues, eigenvectors, rank

    def _diagonalize(self):
        # diagonalize with low rank approximation
        eigenvalues, eigenvectors, self._rank = self._compute_diag()
        if self.scaling == 'kinetic_map':  # scale by eigenvalues
            eigenvectors *= eigenvalues[None, :]
        elif self.scaling == 'commute_map':  # scale by (regularized) timescales
            timescales = 1-self.lag / np.log(np.abs(eigenvalues))
            # dampen timescales smaller than the lag time, as in section 2.5 of ref. [5]
            regularized_timescales = 0.5 * timescales * np.maximum(np.tanh(np.pi * ((timescales - self.lag) / self.lag) + 1), 0)

            eigenvectors *= np.sqrt(regularized_timescales / 2)

        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors
        self._diagonalized = True

    @property
    def timescales(self):
        r"""Implied timescales of the TICA transformation

        For each :math:`i`-th eigenvalue, this returns

        .. math::

            t_i = -\frac{\tau}{\log(|\lambda_i|)}

        where :math:`\tau` is the :py:obj:`lag` of the TICA object and :math:`\lambda_i` is the `i`-th
        :py:obj:`eigenvalue <eigenvalues>` of the TICA object.

        Returns
        -------
        timescales: 1D np.array
            numpy array with the implied timescales. In principle, one should expect as many timescales as
            input coordinates were available. However, less eigenvalues will be returned if the TICA matrices
            were not full rank or :py:obj:`var_cutoff` was parsed
        """
        return -self.lag / np.log(np.abs(self.eigenvalues))

    @property
    def feature_TIC_correlation(self):
        r"""Instantaneous correlation matrix between mean-free input features and TICs

        Denoting the input features as :math:`X_i` and the TICs as :math:`\theta_j`, the instantaneous, linear correlation
        between them can be written as

        .. math::

            \mathbf{Corr}(X_i - \mu_i, \mathbf{\theta}_j) = \frac{1}{\sigma_{X_i - \mu_i}}\sum_l \sigma_{(X_i - \mu_i)(X_l - \mu_l} \mathbf{U}_{li}

        The matrix :math:`\mathbf{U}` is the matrix containing, as column vectors, the eigenvectors of the TICA
        generalized eigenvalue problem .

        Returns
        -------
        feature_TIC_correlation : ndarray(n,m)
            correlation matrix between input features and TICs. There is a row for each feature and a column
            for each TIC.
        """
        feature_sigma = np.sqrt(np.diag(self.cov))
        return np.dot(self.cov, self.eigenvectors[:, : self.dimension()]) / feature_sigma[:, np.newaxis]



class TICABase(Estimator):

    r""" Time-lagged independent component analysis (TICA) [1]_, [2]_, [3]_.

    Parameters
    ----------
    lag : int
        lag time
    dim : int or float, optional, default 0.95
        Number of dimensions (independent components) to project onto.

      * if dim is not set (None) all available ranks are kept:
          `n_components == min(n_samples, n_uncorrelated_features)`
      * if dim is an integer >= 1, this number specifies the number
        of dimensions to keep.
      * if dim is a float with ``0 < dim < 1``, select the number
        of dimensions such that the amount of kinetic variance
        that needs to be explained is greater than the percentage
        specified by dim.
    var_cutoff: None, deprecated
        use dim with a float in range (0, 1]
    kinetic_map : bool, optional, default True, deprecated
        use scaling='kinetic_map'
    commute_map : bool, optional, default False, deprecated
        use scaling='commute_map'
    epsilon : float
        eigenvalue norm cutoff. Eigenvalues of C0 with norms <= epsilon will be
        cut off. The remaining number of eigenvalues define the size of the output.
    stride: int, optional, default = 1
        Use only every stride-th time step. By default, every time step is used.
    skip : int, default=0
        skip the first initial n frames per trajectory.
    reversible: bool, default=True
        symmetrize correlation matrices C_0, C_{\tau}.
    scaling: str or None, default='kinetic_map'
        * None: unscaled.
        * 'kinetic_map': Eigenvectors will be scaled by eigenvalues. As a result, Euclidean
          distances in the transformed data approximate kinetic distances [4]_.
          This is a good choice when the data is further processed by clustering.
        * 'commute_map': Eigenvector_i will be scaled by sqrt(timescale_i / 2). As a result,
          Euclidean distances in the transformed data will approximate commute distances [5]_.

    Notes
    -----
    Given a sequence of multivariate data :math:`X_t`, computes the mean-free
    covariance and time-lagged covariance matrix:

    .. math::

        C_0 &=      (X_t - \mu)^T (X_t - \mu) \\
        C_{\tau} &= (X_t - \mu)^T (X_{t + \tau} - \mu)

    and solves the eigenvalue problem

    .. math:: C_{\tau} r_i = C_0 \lambda_i(tau) r_i,

    where :math:`r_i` are the independent components and :math:`\lambda_i(tau)` are
    their respective normalized time-autocorrelations. The eigenvalues are
    related to the relaxation timescale by

    .. math:: t_i(tau) = -\tau / \ln |\lambda_i|.

    When used as a dimension reduction method, the input data is projected
    onto the dominant independent components.

    References
    ----------
    .. [1] Perez-Hernandez G, F Paul, T Giorgino, G De Fabritiis and F Noe. 2013.
       Identification of slow molecular order parameters for Markov model construction
       J. Chem. Phys. 139, 015102. doi:10.1063/1.4811489
    .. [2] Schwantes C, V S Pande. 2013.
       Improvements in Markov State Model Construction Reveal Many Non-Native Interactions in the Folding of NTL9
       J. Chem. Theory. Comput. 9, 2000-2009. doi:10.1021/ct300878a
    .. [3] L. Molgedey and H. G. Schuster. 1994.
       Separation of a mixture of independent signals using time delayed correlations
       Phys. Rev. Lett. 72, 3634.
    .. [4] Noe, F. and Clementi, C. 2015. Kinetic distance and kinetic maps from molecular dynamics simulation.
        J. Chem. Theory. Comput. doi:10.1021/acs.jctc.5b00553
    .. [5] Noe, F., Banisch, R., Clementi, C. 2016. Commute maps: separating slowly-mixing molecular configurations
       for kinetic modeling. J. Chem. Theory. Comput. doi:10.1021/acs.jctc.6b00762

    """
    def __init__(self, epsilon=None, reversible=True, dim=0.95,
                 scaling='kinetic_map'):
        super(TICABase, self).__init__()
        self.epsilon = epsilon
        self.reversible = reversible
        self.dim = dim
        self.scaling = scaling

    def _create_model(self) -> TICAModelBase:
        return TICAModelBase()

    def transform(self, X):
        r"""Projects the data onto the dominant independent components.

        Parameters
        ----------
        X : ndarray(n, m)
            the input data

        Returns
        -------
        Y : ndarray(n,)
            the projected data
        """
        model = self.fetch_model()
        X_meanfree = X - model.mean
        Y = np.dot(X_meanfree, model.eigenvectors[:, 0:model.dimension()])
        return Y

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
        if self._covar is None:
            self._covar = OnlineCovariance(compute_c00=True, compute_c0t=True, compute_ctt=False, remove_data_mean=True,
                                           reversible=self.reversible, bessel=False, ncov=5)
        self._covar.partial_fit(X)
        return self


    def fit(self, X, **kw):
        covar = OnlineCovariance(compute_c00=True, compute_c0t=True, compute_ctt=False, remove_data_mean=True,
                                 reversible=self.reversible, bessel=False, ncov=5)
        covar.fit(X, **kw)

        self.model.update_model_params(mean=covar.mean,
                                       cov=covar.c00,
                                       cov_tau=covar.c0t)
        self.model._diagonalize()

        return self.model

    def fetch_model(self):
        if self._covar is None:
            raise RuntimeError('call fit or partial_fit prior fetching the model.')
        # obtain the running covariance model, set the TICA model and diagonolize it.
        covar_model = self._covar.fetch_model()
        self._model.set_covar_model(covar_model)
        # TODO: do we want the model always to be diagonalized?
        self._model._diagonalize()
        return self._model
