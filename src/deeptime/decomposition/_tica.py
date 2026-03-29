from typing import Optional, Union, Callable

import numpy as np

from ._vamp import VAMP
from ..basis import Identity
from ..covariance import Covariance, CovarianceModel
from ..numeric import eig_corr

__author__ = 'marscher, clonker'


class TICA(VAMP):
    r""" Time-lagged independent component analysis (TICA).

    TICA is a linear transformation method. In contrast to PCA, which finds
    coordinates of maximal variance, TICA finds coordinates of maximal
    autocorrelation at the given lag time. Therefore, TICA is useful in order
    to find the *slow* components in a dataset and thus an excellent choice to
    transform molecular dynamics data before clustering data for the
    construction of a Markov model. When the input data is the result of a
    Markov process (such as thermostatted molecular dynamics), TICA finds in
    fact an approximation to the eigenfunctions and eigenvalues of the
    underlying Markov operator :footcite:`perez2013identification`.

    It estimates a TICA transformation from *data*. The resulting model can be used
    to obtain eigenvalues, eigenvectors,
    or project input data onto the slowest TICA components.

    Parameters
    ----------
    lagtime : int or None, optional, default=None
        The lagtime under which covariances are estimated. This is only relevant when estimating from data, in case
        covariances are provided this should either be None or exactly the value that was used to estimate
        said covariances.
    epsilon : float, optional, default=1e-6
        Eigenvalue norm cutoff. Eigenvalues of C0 with norms <= epsilon will be
        cut off. The remaining number of eigenvalues define the size
        of the output.
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
    scaling : str or None, default='kinetic_map'
        Can be set to :code:`None`, 'kinetic_map' (:footcite:`noe2015kinetic`),
        or 'commute_map' (:footcite:`noe2016commute`). For more details see :attr:`scaling`.
    observable_transform : callable, optional, default=Identity
        A feature transformation on the raw data which is used to estimate the model.

    Notes
    -----
    Given a sequence of multivariate data :math:`X_t`, it computes the
    mean-free covariance and time-lagged covariance matrix:

    .. math::

        \begin{aligned}
        C_0 &=      (X_t - \mu)^T \mathrm{diag}(w) (X_t - \mu) \\
        C_{\tau} &= (X_t - \mu)^T \mathrm{diag}(w) (X_{t + \tau} - \mu)
        \end{aligned}

    where :math:`w` is a vector of weights for each time step. By default, these weights
    are all equal to one, but different weights are possible, like the re-weighting
    to equilibrium described in :footcite:`wu2017variational`. Subsequently, the eigenvalue problem

    .. math:: C_{\tau} r_i = C_0 \lambda_i r_i,

    is solved,where :math:`r_i` are the independent components and :math:`\lambda_i` are
    their respective normalized time-autocorrelations. The eigenvalues are
    related to the relaxation timescale by

    .. math::

        t_i = -\frac{\tau}{\ln |\lambda_i|}.

    When used as a dimension reduction method, the input data is projected
    onto the dominant independent components.

    Under the assumption of reversible dynamics and the limit of good statistics, the time-lagged autocovariance
    :math:`C_\tau` is symmetric. Due to finite data, this symmetry is explicitly enforced in the estimator.

    TICA was originally introduced for signal processing in :footcite:`molgedey1994separation`. It was
    introduced to molecular dynamics and as a method for the construction
    of Markov models in :footcite:`perez2013identification` and :footcite:`schwantes2013improvements`. It was shown
    in :footcite:`perez2013identification` that when applied
    to molecular dynamics data, TICA is an approximation to the eigenvalues
    and eigenvectors of the true underlying dynamics.

    Examples
    --------
    Invoke TICA transformation with a given lag time and output dimension:

    >>> import numpy as np
    >>> from deeptime.decomposition import TICA
    >>> data = np.random.random((100,3))
    >>> # fixed output dimension
    >>> estimator = TICA(dim=1, lagtime=2).fit(data)
    >>> model_onedim = estimator.fetch_model()
    >>> projected_data = model_onedim.transform(data)
    >>> np.testing.assert_equal(projected_data.shape[1], 1)

    or invoke it with a percentage value of to-be captured kinetic variance (80% in the example)

    >>> estimator = TICA(var_cutoff=0.8, lagtime=2).fit(data)
    >>> model_var = estimator.fetch_model()
    >>> projected_data = model_var.transform(data)

    For a brief explaination why TICA outperforms PCA to extract a good reaction coordinate have a look `here
    <http://docs.markovmodel.org/lecture_tica.html#Example:-TICA-versus-PCA-in-a-stretched-double-well-potential>`_.

    See also
    --------
    CovarianceKoopmanModel : TICA estimation output model

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, lagtime: Optional[int] = None, epsilon: float = 1e-6, dim: Optional[int] = None,
                 var_cutoff: Optional[float] = None, scaling: Optional[str] = 'kinetic_map',
                 observable_transform: Callable[[np.ndarray], np.ndarray] = Identity()):
        super(TICA, self).__init__(lagtime=lagtime, dim=dim, var_cutoff=var_cutoff, scaling=scaling, epsilon=epsilon,
                                   observable_transform=observable_transform)

    @classmethod
    def covariance_estimator(cls, lagtime: int, ncov: Union[int] = float('inf')):
        return Covariance(lagtime=lagtime, compute_c00=True, compute_c0t=True, compute_ctt=False,
                          remove_data_mean=True, reversible=True, bessels_correction=False,
                          ncov=ncov)

    @staticmethod
    def _decomposition(covariances, epsilon, scaling, dim, var_cutoff) -> VAMP._DiagonalizationResults:
        from deeptime.numeric import ZeroRankError

        # diagonalize with low rank approximation
        try:
            eigenvalues, eigenvectors, rank = eig_corr(covariances.cov_00, covariances.cov_0t, epsilon,
                                                       canonical_signs=True, return_rank=True)
        except ZeroRankError:
            raise ZeroRankError('All input features are constant in all time steps. '
                                'No dimension would be left after dimension reduction.')
        if scaling in ('km', 'kinetic_map'):  # scale by eigenvalues
            eigenvectors *= eigenvalues[None, :]
        elif scaling == 'commute_map':  # scale by (regularized) timescales
            lagtime = covariances.lagtime
            timescales = 1. - lagtime / np.log(np.abs(eigenvalues))
            # dampen timescales smaller than the lag time, as in section 2.5 of ref. [5]
            regularized_timescales = 0.5 * timescales * np.maximum(
                np.tanh(np.pi * ((timescales - lagtime) / lagtime) + 1), 0)

            eigenvectors *= np.sqrt(regularized_timescales / 2)

        return VAMP._DiagonalizationResults(
            rank0=rank, rankt=rank, singular_values=eigenvalues,
            left_singular_vecs=eigenvectors, right_singular_vecs=eigenvectors
        )

    @property
    def epsilon(self) -> float:
        r""" Eigenvalue norm cutoff.

        :type: float
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        self._epsilon = value

    @property
    def scaling(self) -> Optional[str]:
        r""" Scaling parameter. Can take the following values:

        * None: unscaled.
        * 'kinetic_map': Eigenvectors will be scaled by eigenvalues. As a result, Euclidean
          distances in the transformed data approximate kinetic distances :footcite:`noe2015kinetic`.
          This is a good choice when the data is further processed by clustering.
        * 'commute_map': Eigenvector i will be scaled by sqrt(timescale_i / 2). As a result,
          Euclidean distances in the transformed data will approximate commute distances :footcite:`noe2016commute`.

        :getter: Yields the currently configured scaling.
        :setter: Sets a new scaling.
        :type: str or None
        """
        return self._scaling

    @scaling.setter
    def scaling(self, value: Optional[str]):
        valid_scalings = [None, 'kinetic_map', 'km', 'commute_map']
        if value not in valid_scalings:
            raise ValueError("Scaling parameter is allowed to be one of {}".format(valid_scalings))
        self._scaling = value

    def fit_from_covariances(self, covariances: Union[Covariance, CovarianceModel]):
        r"""Fits a model based on provided symmetrized covariances.

        Parameters
        ----------
        covariances : Covariance or CovarianceModel
            The covariances

        Returns
        -------
        self : TICA
            Reference to self.
        """
        covariances = self._to_covariance_model(covariances)
        if not covariances.symmetrized:
            raise ValueError("The covariance model must be estimated such that the "
                             "autocorrelations are symmetric!")
        if covariances.cov_0t is None:
            raise ValueError("The covariance model must contain C0t!")
        self._model = self._decompose(covariances)
        self._covariance_estimator = None
        return self
