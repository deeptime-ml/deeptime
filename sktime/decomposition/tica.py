from numbers import Real, Integral
from typing import Optional

import numpy as np

from ..base import Model, Estimator, Transformer
from ..covariance.covariance import Covariance
from ..numeric.eigen import eig_corr
from ..util import cached_property

__author__ = 'marscher, clonker'


class TICAModel(Model, Transformer):
    r""" Class which holds the results from the :class:`TICA` estimator.

    Diagonalization to obtain rank, eigenvalues,
    and eigenvectors is performed lazily.

    See Also
    --------
    TICA : TICA estimator
    """
    def __init__(self, lagtime: int, mean_0: np.ndarray, cov_00: np.ndarray, cov_0t: np.ndarray, dim: Optional[Real],
                 epsilon=1e-6, scaling=None):
        r"""
        Initializes a new TICA model.

        Parameters
        ----------
        lagtime : int
            The lagtime at which the covariances were estimated.
        mean_0 : (n,) ndarray
            Instantaneous data frame mean.
        cov_00 : (n, n) ndarray
            Instantaneous covariance matrix.
        cov_0t : (n, n) ndarray
            Time-lagged covariance matrix.
        dim : int or float or None
            Dimension parameter which was used for estimation. For a detailed description, see :attr:`TICA.dim`. The
            resulting output dimension can be obtained by the property :attr:`output_dimension`.
        epsilon : float, optional, default=None
            Eigenvalue norm cutoff. Eigenvalues of C0 with norms <= epsilon will be cut off.
        scaling : str, optional, default=None
            Scaling parameter, see :attr:`TICA.scaling`.
        """
        super().__init__()
        self._lagtime = lagtime
        self._cov_00 = cov_00
        self._cov_0t = cov_0t
        self._mean_0 = mean_0
        self._dim = dim
        self._epsilon = epsilon
        self._scaling = scaling
        self._rank = None

    def transform(self, data, **kw):
        r""" Removes mean from data and projects it into the TICA basis.

        Parameters
        ----------
        data : ndarray
            Input data.
        **kw
            Ignored kwargs

        Returns
        -------
        projection : ndarray
            Projected data.
        """
        data_meanfree = data - self.mean_0
        return np.dot(data_meanfree, self.eigenvectors[:, :self.output_dimension])

    @property
    def epsilon(self):
        r""" For estimation used eigenvalue norm cutoff. """
        return self._epsilon

    @property
    def mean_0(self):
        r""" Instantaneous mean of data. """
        return self._mean_0

    @property
    def lagtime(self):
        r""" The lagtime that was used to estimate the model. """
        return self._lagtime

    @property
    def dim(self) -> Optional[Real]:
        r""" The dim parameter that was used to estimate the model, see :attr:`TICA.dim`. """
        return self._dim

    @property
    def scaling(self):
        r""" The scaling parameter that was used to estimate the model, see :attr:`TICA.scaling`. """
        return self._scaling

    @property
    def cov_00(self):
        r""" Instantaneous covariances. """
        return self._cov_00

    @property
    def cov_0t(self):
        r""" Time-shifted covariances. """
        return self._cov_0t

    @property
    def eigenvectors(self):
        r""" Eigenvectors of the TICA problem, column-wise.

        :type: (N,M) ndarray

        Examples
        --------
        >>> model = TICAModel(lagtime=1, mean_0=np.zeros((2,)), cov_00=np.eye(2), cov_0t=np.eye(2), dim=None)
        >>> eigvec_0 = model.eigenvectors[:, 0]  # note, that these are the right eigenvectors
        >>> eigvec_1 = model.eigenvectors[:, 1]  # so they are stored in a column-matrix
        """
        return self._rank_eigenvalues_eigenvectors[2]

    @property
    def eigenvalues(self):
        r""" Eigenvalues of the TICA problem (usually denoted :math:`\lambda`)

        :type: (N,) ndarray
        """
        return self._rank_eigenvalues_eigenvectors[1]

    @staticmethod
    def _cumvar(eigenvalues):
        r""" Compute cumulative variance.

        Parameters
        ----------
        eigenvalues : ndarray
            The eigenvalues

        Returns
        -------
        cumvar : ndarray
            The cumulative variance
        """
        cumvar = np.cumsum(eigenvalues ** 2)
        cumvar /= cumvar[-1]
        return cumvar

    @property
    def cumvar(self):
        r""" Cumulative sum of the the TICA eigenvalues

        Returns
        -------
        cumvar : 1D ndarray
            The cumulative sum.
        """
        return TICAModel._cumvar(self.eigenvalues)

    @property
    def rank(self) -> int:
        r""" The rank of :math:`\mathrm{cov}_{00}^{-0.5}`.

        :type: int
        """
        return self._rank_eigenvalues_eigenvectors[0]

    @property
    def output_dimension(self):
        r""" Effective output dimension, computed from :attr:`cov_00` and :attr:`dim` parameters.

        :type: int
        """
        """ output dimension """
        if self.cov_00 is None:
            raise RuntimeError('Model has no covariance matrix to compute the output dimension on.')

        if self.dim is None or (isinstance(self.dim, float) and self.dim == 1.0):
            return self.rank
        if isinstance(self.dim, float):
            # subspace_variance, reduce the output dimension if needed
            return min(len(self.eigenvalues), np.searchsorted(TICAModel._cumvar(self.eigenvalues), float(self.dim)) + 1)
        else:
            return np.min([self.rank, self.dim])

    @cached_property
    def _rank_eigenvalues_eigenvectors(self):
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
            timescales = 1. - self.lagtime / np.log(np.abs(eigenvalues))
            # dampen timescales smaller than the lag time, as in section 2.5 of ref. [5]
            regularized_timescales = 0.5 * timescales * np.maximum(
                np.tanh(np.pi * ((timescales - self.lagtime) / self.lagtime) + 1), 0)

            eigenvectors *= np.sqrt(regularized_timescales / 2)
        return rank, eigenvalues, eigenvectors

    def timescales(self, lagtime):
        r"""Implied timescales of the TICA transformation

        For each :math:`i`-th eigenvalue, this returns

        .. math::

            t_i = -\frac{\tau}{\log(|\lambda_i|)}

        where :math:`\tau` is the :attr:`lagtime` of the TICA object and :math:`\lambda_i` is the `i`-th
        :attr:`eigenvalue <eigenvalues>` of the TICA object.

        Returns
        -------
        timescales: 1D np.array
            numpy array with the implied timescales. In principle, one should expect as many timescales as
            input coordinates were available. However, less eigenvalues will be returned if the TICA matrices
            were not full rank or :attr:`dim` contained a floating point percentage, i.e., was interpreted as
            variance cutoff.
        """
        return - lagtime / np.log(np.abs(self.eigenvalues))

    @property
    def feature_tic_correlation(self):
        r"""Instantaneous correlation matrix between mean-free input features and TICs

        Denoting the input features as :math:`X_i` and the TICs as :math:`\theta_j`, the instantaneous, linear correlation
        between them can be written as

        .. math::

            \mathbf{Corr}(X_i - \mu_i, \mathbf{\theta}_j) = \frac{1}{\sigma_{X_i - \mu_i}}\sum_l \sigma_{(X_i - \mu_i)(X_l - \mu_l)} \mathbf{U}_{li}

        The matrix :math:`\mathbf{U}` is the matrix containing, as column vectors, the eigenvectors of the TICA
        generalized eigenvalue problem .

        Returns
        -------
        feature_TIC_correlation : ndarray(n,m)
            correlation matrix between input features and TICs. There is a row for each feature and a column
            for each TIC.
        """
        feature_sigma = np.sqrt(np.diag(self.cov_00))
        return np.dot(self.cov_00, self.eigenvectors[:, : self.output_dimension]) / feature_sigma[:, np.newaxis]


class TICA(Estimator, Transformer):
    r""" Time-lagged independent component analysis (TICA).

    TICA is a linear transformation method. In contrast to PCA, which finds
    coordinates of maximal variance, TICA finds coordinates of maximal
    autocorrelation at the given lag time. Therefore, TICA is useful in order
    to find the *slow* components in a dataset and thus an excellent choice to
    transform molecular dynamics data before clustering data for the
    construction of a Markov model. When the input data is the result of a
    Markov process (such as thermostatted molecular dynamics), TICA finds in
    fact an approximation to the eigenfunctions and eigenvalues of the
    underlying Markov operator [1]_.

    It estimates a TICA transformation from *data*. The resulting model can be used to obtain eigenvalues, eigenvectors,
    or project input data onto the slowest TICA components.

    Notes
    -----
    Given a sequence of multivariate data :math:`X_t`, it computes the
    mean-free covariance and time-lagged covariance matrix:

    .. math::

        C_0 &=      (X_t - \mu)^T \mathrm{diag}(w) (X_t - \mu) \\
        C_{\tau} &= (X_t - \mu)^T \mathrm{diag}(w) (X_t + \tau - \mu)

    where w is a vector of weights for each time step. By default, these weights
    are all equal to one, but different weights are possible, like the re-weighting
    to equilibrium described in [6]_. Subsequently, the eigenvalue problem

    .. math:: C_{\tau} r_i = C_0 \lambda_i r_i,

    is solved,where :math:`r_i` are the independent components and :math:`\lambda_i` are
    their respective normalized time-autocorrelations. The eigenvalues are
    related to the relaxation timescale by

    .. math::

        t_i = -\frac{\tau}{\ln |\lambda_i|}.

    When used as a dimension reduction method, the input data is projected
    onto the dominant independent components.

    TICA was originally introduced for signal processing in [2]_. It was
    introduced to molecular dynamics and as a method for the construction
    of Markov models in [1]_ and [3]_. It was shown in [1]_ that when applied
    to molecular dynamics data, TICA is an approximation to the eigenvalues
    and eigenvectors of the true underlying dynamics.

    Examples
    --------
    Invoke TICA transformation with a given lag time and output dimension:

    >>> import numpy as np
    >>> from sktime.decomposition import TICA
    >>> data = np.random.random((100,3))
    >>> # fixed output dimension
    >>> estimator = TICA(lagtime=2, dim=1).fit(data)
    >>> model_onedim = estimator.fetch_model()
    >>> projected_data = model_onedim.transform(data)
    >>> np.testing.assert_equal(projected_data.shape[1], 1)

    or invoke it with a percentage value of to-be captured kinetic variance (80% in the example)

    >>> estimator = TICA(lagtime=2, dim=0.8).fit(data)
    >>> model_var = estimator.fetch_model()
    >>> projected_data = model_var.transform(data)

    For a brief explaination why TICA outperforms PCA to extract a good reaction
    coordinate have a look `here
    <http://docs.markovmodel.org/lecture_tica.html#Example:-TICA-versus-PCA-in-a-stretched-double-well-potential>`_.

    See also
    --------
    :class:`TICAModel` : TICA estimation output model

    References
    ----------

    .. [1] Perez-Hernandez G, F Paul, T Giorgino, G De Fabritiis and F Noe. 2013.
       Identification of slow molecular order parameters for Markov model construction
       J. Chem. Phys. 139, 015102. doi:10.1063/1.4811489

    .. [2] L. Molgedey and H. G. Schuster. 1994.
       Separation of a mixture of independent signals using time delayed correlations
       Phys. Rev. Lett. 72, 3634.

    .. [3] Schwantes C, V S Pande. 2013.
       Improvements in Markov State Model Construction Reveal Many Non-Native Interactions in the Folding of NTL9
       J. Chem. Theory. Comput. 9, 2000-2009. doi:10.1021/ct300878a

    .. [4] Noe, F. and Clementi, C. 2015. Kinetic distance and kinetic maps from molecular dynamics simulation.
        J. Chem. Theory. Comput. doi:10.1021/acs.jctc.5b00553

    .. [5] Noe, F., Banisch, R., Clementi, C. 2016. Commute maps: separating slowly-mixing molecular configurations
       for kinetic modeling. J. Chem. Theory. Comput. doi:10.1021/acs.jctc.6b00762

    .. [6] Wu, H., Nueske, F., Paul, F., Klus, S., Koltai, P., and Noe, F. 2016. Bias reduced variational
        approximation of molecular kinetics from short off-equilibrium simulations. J. Chem. Phys. (submitted),
        https://arxiv.org/abs/1610.06773.

    .. [7] Chan, T. F., Golub G. H., LeVeque R. J. 1979. Updating formulae and pairwiese algorithms for
        computing sample variances. Technical Report STAN-CS-79-773, Department of Computer Science, Stanford University.
    """

    def __init__(self, lagtime: int, epsilon: float = 1e-6, reversible: bool = True, dim: Optional[Real] = 0.95,
                 scaling: Optional[str] = 'kinetic_map', ncov: int = 5):
        r"""Constructs a new TICA estimator.

        Parameters
        ----------
        lagtime : int, optional, default = 10
            the lag time, in multiples of the input time step
        epsilon : float, optional, default=1e-6
            Eigenvalue norm cutoff. Eigenvalues of C0 with norms <= epsilon will be
            cut off. The remaining number of eigenvalues define the size
            of the output.
        dim : None, int, or float, optional, default 0.95
            Number of dimensions (independent components) to project onto.

          * if dim is not set (None) all available ranks are kept:
              `n_components == min(n_samples, n_uncorrelated_features)`
          * if dim is an integer >= 1, this number specifies the number
            of dimensions to keep.
          * if dim is a float with ``0 < dim <= 1``, select the number
            of dimensions such that the amount of kinetic variance
            that needs to be explained is greater than the percentage
            specified by dim.
        scaling: str or None, default='kinetic_map'
            Can be set to :code:`None`, 'kinetic_map', or 'commute_map'. For more details see :attr:`scaling`.
        ncov : int, default=infinity
            Limit the memory usage of the algorithm from [7]_ to an amount that corresponds
            to ncov additional copies of each correlation matrix. Influences performance and numerical stability.
        """
        super(TICA, self).__init__()
        # tica parameters
        self.epsilon = epsilon
        self.dim = dim
        self.scaling = scaling

        # online cov parameters
        self.reversible = reversible
        self._covar = Covariance(lagtime=lagtime, compute_c00=True, compute_c0t=True, compute_ctt=False,
                                 remove_data_mean=True, reversible=self.reversible, bessels_correction=False,
                                 ncov=ncov)

    @property
    def epsilon(self) -> float:
        r""" Eigenvalue norm cutoff. """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        self._epsilon = value

    @property
    def scaling(self) -> Optional[str]:
        r""" Scaling parameter. Can take the following values:

        * None: unscaled.
        * 'kinetic_map': Eigenvectors will be scaled by eigenvalues. As a result, Euclidean
          distances in the transformed data approximate kinetic distances [4]_.
          This is a good choice when the data is further processed by clustering.
        * 'commute_map': Eigenvector i will be scaled by sqrt(timescale_i / 2). As a result,
          Euclidean distances in the transformed data will approximate commute distances [5]_.

        :getter: Yields the currently configured scaling.
        :setter: Sets a new scaling.
        :type: str or None
        """
        return self._scaling

    @scaling.setter
    def scaling(self, value: Optional[str]):
        valid_scalings = [None, 'kinetic_map', 'commute_map']
        if value not in valid_scalings:
            raise ValueError("Scaling parameter is allowed to be one of {}".format(valid_scalings))
        self._scaling = value

    @property
    def ncov(self):
        r""" Depth of the moments storage in [7]_. This parameter influences performance and numerical stability. """
        return self._ncov

    @ncov.setter
    def ncov(self, value):
        self._covar.ncov = value

    @property
    def dim(self) -> Optional[float]:
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
                raise ValueError("TICA: Invalid dimension parameter, if it is given in terms of the "
                                 "dimension (integer), must be positive.")
        elif isinstance(value, Real) and (value <= 0. or float(value) > 1.0):
            raise ValueError("TICA: Invalid dimension parameter, if it is given in terms of a floating point, "
                             "can only be in the interval (0, 1].")
        elif value is not None and not isinstance(value, (Integral, Real)):
            raise ValueError("Invalid type for dimension, got {}".format(value))
        self._dim = value

    def transform(self, data, **kw):
        r"""Projects the data onto the dominant independent components.

        Parameters
        ----------
        data : ndarray(n, m)
            the input data
        **kw
            Ignored kwargs

        Returns
        -------
        Y : ndarray(n,)
            the projected data
        """
        return self.fetch_model().transform(data)

    def partial_fit(self, data, weights=None, column_selection=None):
        """ incrementally update the covariances and mean.

        Parameters
        ----------
        data : array, list of arrays
            input data.
        weights : array or list of arrays, optional, default=None
            Optional reweighting factors.
        column_selection : ndarray, optional, default=None
            Columns of the trajectories to restrict estimation to. Must be given in terms of an index array.
        """
        self._covar.partial_fit(data, weights=weights, column_selection=column_selection)
        return self

    def fit(self, data, lagtime=None, weights=None, column_selection=None, **kw):
        r""" Fit a new :class:`TICAModel` based on provided data.

        Parameters
        ----------
        data : (T, n) ndarray
            timeseries data
        lagtime : int, optional, default=None
            Override for :attr:`lagtime`.
        weights : ndarray or object, optional, default=None
            * An object that allows to compute re-weighting factors to estimate equilibrium means and correlations from
              off-equilibrium data. The only requirement is that weights possesses a method weights(X), that accepts a
              trajectory X (np.ndarray(T, n)) and returns a vector of re-weighting factors (np.ndarray(T,)). See:

              * :class:`KoopmanEstimator <sktime.covariance.KoopmanEstimator>`

            * A list of ndarrays (ndim=1) specifies the weights for each frame of each trajectory.
        column_selection : (d, dtype=int) ndarray, optional, default=None
            Optional column selection within provided data.
        **kw
            Ignored keyword arguments for scikit-learn compatibility.

        Returns
        -------
        self : TICA
            Reference to self.
        """
        self._covar.fit(data, lagtime=lagtime, weights=weights, column_selection=column_selection)
        return self

    def fetch_model(self) -> TICAModel:
        r""" Yields the estimated model.

        Returns
        -------
        model : TICAModel
            The estimated model.
        """
        covar_model = self._covar.fetch_model()
        return TICAModel(lagtime=self.lagtime, mean_0=covar_model.mean_0, cov_00=covar_model.cov_00,
                         cov_0t=covar_model.cov_0t, dim=self.dim, epsilon=self.epsilon, scaling=self.scaling)

    @property
    def lagtime(self):
        r""" The lagtime at which covariances are estimated.

        :getter: Yields the currently configured lagtime.
        :setter: Sets a new lagtime, must be >= 0.
        :type: int
        """
        return self._covar.lagtime

    @lagtime.setter
    def lagtime(self, value):
        self._covar.lagtime = value
