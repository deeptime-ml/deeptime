import numpy as np

from sktime.base import Model, Estimator, Transformer
from sktime.covariance.online_covariance import OnlineCovariance
from sktime.numeric.eigen import eig_corr

__author__ = 'marscher'


class TICAModel(Model, Transformer):

    def __init__(self, mean_0=None, cov_00=None, cov_0t=None, dim=None, epsilon=1e-6, scaling=None):
        self.cov_00 = cov_00
        self.cov_0t = cov_0t
        self.mean_0 = mean_0
        self.dim = dim
        self.epsilon = epsilon
        self.scaling = scaling
        self._rank = None

    def transform(self, data):
        data_meanfree = data - self.mean_0
        return np.dot(data_meanfree, self.eigenvectors[:, :self.output_dimension()])

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, value):
        if not (value is None or 0 < value <= 1 or (isinstance(value, int)) and value > 0):
            raise ValueError('dim has to be either None, a float 0 < dim <= 1, '
                             'or a positive integer, but was {}'.format(value))
        self._dim = value

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
    def cov_00(self):
        return self._cov

    @cov_00.setter
    def cov_00(self, value):
        self._diagonalized = False
        self._cov = value

    @property
    def cov_0t(self):
        return self._cov_tau

    @cov_0t.setter
    def cov_0t(self, value):
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
        return TICAModel._cumvar(self.eigenvalues)

    @property
    def rank(self):
        if not self._diagonalized:
            self._diagonalize()
        return self._rank

    def output_dimension(self):
        """ output dimension """
        if self.cov_00 is None:
            raise RuntimeError('Model has no covariance matrix to compute the output dimension on.')

        if self.dim is None or (isinstance(self.dim, float) and self.dim == 1.0):
            return self.rank
        if isinstance(self.dim, float):
            # subspace_variance, reduce the output dimension if needed
            return min(len(self.eigenvalues), np.searchsorted(TICAModel._cumvar(self.eigenvalues), self.dim) + 1)
        else:
            return np.min([self.rank, self.dim])

    def _diagonalize(self):
        from sktime.numeric.eigen import ZeroRankError

        # diagonalize with low rank approximation
        try:
            eigenvalues, eigenvectors, rank = eig_corr(self.cov_00, self.cov_0t, self.epsilon,
                                                       sign_maxelement=True, return_rank=True)
        except ZeroRankError:
            raise ZeroRankError('All input features are constant in all time steps. '
                                'No dimension would be left after dimension reduction.')
        if self.scaling == 'kinetic_map':  # scale by eigenvalues
            eigenvectors *= eigenvalues[None, :]
        elif self.scaling == 'commute_map':  # scale by (regularized) timescales
            timescales = 1 - self.lagtime / np.log(np.abs(eigenvalues))
            # dampen timescales smaller than the lag time, as in section 2.5 of ref. [5]
            regularized_timescales = 0.5 * timescales * np.maximum(
                np.tanh(np.pi * ((timescales - self.lagtime) / self.lagtime) + 1), 0)

            eigenvectors *= np.sqrt(regularized_timescales / 2)

        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors
        self._rank = rank
        self._diagonalized = True

    def timescales(self, lagtime):
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
        return - lagtime / np.log(np.abs(self.eigenvalues))

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
        feature_sigma = np.sqrt(np.diag(self.cov_00))
        return np.dot(self.cov_00, self.eigenvectors[:, : self.output_dimension()]) / feature_sigma[:, np.newaxis]


class TICA(Estimator, Transformer):

    r""" Time-lagged independent component analysis (TICA) [1]_, [2]_, [3]_.

    Parameters
    ----------
    lagtime : int
        the time of the lag
    dim : int or float, optional, default 0.95
        Number of dimensions (independent components) to project onto.

      * if dim is not set (None) all available ranks are kept:
          `n_components == min(n_samples, n_uncorrelated_features)`
      * if dim is an integer >= 1, this number specifies the number
        of dimensions to keep.
      * if dim is a float with ``0 < dim <= 1``, select the number
        of dimensions such that the amount of kinetic variance
        that needs to be explained is greater than the percentage
        specified by dim.
    epsilon : float
        eigenvalue norm cutoff. Eigenvalues of C0 with norms <= epsilon will be
        cut off. The remaining number of eigenvalues define the size of the output.
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
    def __init__(self, epsilon=1e-6, reversible=True, dim=0.95,
                 scaling='kinetic_map', ncov=5, reweighting_transformation=None):
        super(TICA, self).__init__()
        # tica parameters
        self._model.epsilon = epsilon
        self._model.dim = dim
        self._model.scaling = scaling

        # online cov parameters
        self.reversible = reversible
        self._covar = OnlineCovariance(compute_c00=True, compute_c0t=True, compute_ctt=False, remove_data_mean=True,
                                       reversible=self.reversible, bessel=False, ncov=ncov)
        self.reweighting_transformation = reweighting_transformation

    def _create_model(self) -> TICAModel:
        return TICAModel()

    def transform(self, data):
        r"""Projects the data onto the dominant independent components.

        Parameters
        ----------
        data : ndarray(n, m)
            the input data

        Returns
        -------
        Y : ndarray(n,)
            the projected data
        """
        return self.fetch_model().transform(data)

    def partial_fit(self, X, weights=None, column_selection=None):
        """ incrementally update the covariances and mean.

        Parameters
        ----------
        X: array, list of arrays
            input data.
        """
        # compute koopman weights for unlagged data.
        weights = self._get_weights(weights, X[0])
        self._covar.partial_fit(X, weights=weights, column_selection=column_selection)
        return self

    def fit(self, X, y=None, weights=None, column_selection=None):
        # compute koopman weights for unlagged data.
        weights = self._get_weights(weights, X[0])
        self._covar.fit(X, weights=weights, column_selection=column_selection)
        return self

    def _get_weights(self, weights, X):
        if self.reweighting_transformation is not None:
            if weights is not None:
                raise ValueError('Weights given but Koopman reweighting already in place. '
                                 'Either use reweighting_transformation=None, weights=w or '
                                 'reweighting_transformation=(A, b) '
                                 'and weights=None in fit() or partial_fit()')
            u, u_const = self.reweighting_transformation
            weights = X.dot(u) + u_const
        return weights

    def fetch_model(self) -> TICAModel:
        covar_model = self._covar.fetch_model()
        self._model.cov_00 = covar_model.cov_00
        self._model.cov_0t = covar_model.cov_0t
        self._model.mean_0 = covar_model.mean_0
        return self._model
