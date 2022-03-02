from math import sqrt
from typing import Optional, Callable, Union, List

import numpy as np

from .._transition_counting import TransitionCountEstimator
from ...base import Estimator
from ...numeric import is_square_matrix
from .._base import _MSMBaseEstimator, BayesianMSMPosterior
from . import MarkovStateModel, MaximumLikelihoodMSM
from ...util.decorators import deprecated_method

__author__ = 'noe, marscher, clonker'


class BayesianMSM(_MSMBaseEstimator):
    r""" Bayesian estimator for MSMs given discrete trajectory statistics.

    Implementation following :footcite:`trendelkamp2015estimation`.

    Parameters
    ----------
    n_samples : int, optional, default=100
        Number of sampled transition matrices used in estimation of confidences.
    n_steps : int, optional, default=None
        Number of Gibbs sampling steps for each transition matrix. If None, nsteps will be determined
        automatically as the square root of the number of states in the full state space of the count matrix.
        This is a heuristic for the number of steps it takes to decorrelate between samples.
    reversible : bool, optional, default=True
        If true compute reversible MSM, else non-reversible MSM.
    stationary_distribution_constraint : ndarray, optional, default=None
        Stationary vector on the full set of states. Assign zero stationary probabilities to states for which the
        stationary vector is unknown. Estimation will be made such that the resulting ensemble of transition
        matrices is defined on the intersection of the states with positive stationary vector and the largest
        connected set (undirected in the default case).
    sparse : bool, optional, default=False
        If true compute count matrix, transition matrix and all derived quantities using sparse matrix algebra. In
        this case python sparse matrices will be returned by the corresponding functions instead of numpy arrays.
        This behavior is suggested for very large numbers of states (e.g. > 4000) because it is likely to be much
        more efficient.
    maxiter : int, optional, default=1000000
        Optional parameter with reversible = True, sets the maximum number of iterations before the transition
        matrix estimation method exits.
    maxerr : float, optional, default=1e-8
        Optional parameter with reversible = True. Convergence tolerance for transition matrix estimation. This
        specifies the maximum change of the Euclidean norm of relative stationary probabilities
        (:math:`x_i = \sum_k x_{ik}`). The relative stationary probability changes
        :math:`e_i = (x_i^{(1)} - x_i^{(2)})/(x_i^{(1)} + x_i^{(2)})` are used in order to track changes in small
        probabilities. The Euclidean norm of the change vector, :math:`|e_i|_2`, is compared to maxerr.
    lagtime : int, optional, default=None
        The lagtime that is used when fitting directly from discrete trajectories.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    Note that the following example is only qualitatively and not
    quantitatively reproducible because it involves random numbers.

    We build a Bayesian Markov model for the following two trajectories at lag
    time :math:`\tau=2`:

    >>> import numpy as np
    >>> import deeptime
    >>> dtrajs = [np.array([0,1,2,2,2,2,1,2,2,2,1,0,0,0,0,0,0,0]), np.array([0,0,0,0,1,1,2,2,2,2,2,2,2,1,0,0])]
    >>> counts = deeptime.markov.TransitionCountEstimator(lagtime=2, count_mode="effective").fit(dtrajs).fetch_model()
    >>> mm = deeptime.markov.msm.BayesianMSM().fit(counts).fetch_model()

    The resulting Model contains a prior :class:`MSM <deeptime.markov.msm.MarkovStateModel>` as well as a list of sample
    MSMs. Its transition matrix comes from a maximum likelihood estimation. We can access, e.g., the transition matrix
    as follows:

    >>> print(mm.prior.transition_matrix)  # doctest: +SKIP
    [[ 0.70000001  0.16463699  0.135363  ]
     [ 0.38169055  0.          0.61830945]
     [ 0.12023989  0.23690297  0.64285714]]

    Furthermore, the bayesian MSM posterior returned by :meth:`fetch_model` is able to
    compute the probability distribution and statistical models of all methods
    that are offered by the MSM object. This works as follows. The :meth:`BayesianMSMPosterior.gather_stats` method
    takes as argument the method you want to evaluate and then returns a statistics summary over requested quantity:

    >>> print(mm.gather_stats('transition_matrix').mean)  # doctest: +SKIP
    [[ 0.71108663  0.15947371  0.12943966]
     [ 0.41076105  0.          0.58923895]
     [ 0.13079372  0.23005443  0.63915185]]

    Likewise, the standard deviation by element:

    >>> print(mm.gather_stats('transition_matrix').std)  # doctest: +SKIP
    [[ 0.13707029  0.09479627  0.09200214]
     [ 0.15247454  0.          0.15247454]
     [ 0.07701315  0.09385258  0.1119089 ]]

    And this is the 95% (2 sigma) confidence interval. You can control the
    percentile using the conf argument in this function:

    >>> stats = mm.gather_stats('transition_matrix')
    >>> print(stats.L) # doctest: +SKIP
    >>> print(stats.R)  # doctest: +SKIP
    [[ 0.44083423  0.03926518  0.0242113 ]
     [ 0.14102544  0.          0.30729828]
     [ 0.02440188  0.07629456  0.43682481]]
    [[ 0.93571706  0.37522581  0.40180041]
     [ 0.69307665  0.          0.8649215 ]
     [ 0.31029752  0.44035732  0.85994006]]

    If you want to compute expectations of functions that require arguments,
    just pass these arguments as well:

    >>> print(mm.gather_stats('mfpt', A=0, B=2)) # doctest: +SKIP
    12.9049811296

    And if you want to histogram the distribution or compute more complex
    statistical moment such as the covariance between different quantities,
    just get the full sample of your quantity of interest and evaluate it
    at will:

    >>> samples = mm.gather_stats('mfpt', store_samples=True, A=0, B=2).samples
    >>> print(samples[:4]) # doctest: +SKIP
    [7.9763615793248155, 8.6540958274695701, 26.295326015231058, 17.909895469938899]

    Internally, the SampledMSM object has 100 transition matrices (the number
    can be controlled by nsamples), that were computed by the transition matrix
    sampling method. All of the above sample functions iterate over these 100
    transition matrices and evaluate the requested function with the given
    parameters on each of them.
    """

    def __init__(self, n_samples: int = 100, n_steps: int = None, reversible: bool = True,
                 stationary_distribution_constraint: Optional[np.ndarray] = None,
                 sparse: bool = False, maxiter: int = int(1e6), maxerr: float = 1e-8, lagtime: Optional[int] = None):
        super(BayesianMSM, self).__init__(reversible=reversible, sparse=sparse)
        self.stationary_distribution_constraint = stationary_distribution_constraint
        self.maxiter = maxiter
        self.maxerr = maxerr
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.lagtime = lagtime

    @property
    def stationary_distribution_constraint(self) -> Optional[np.ndarray]:
        r"""
        The stationary distribution constraint that can either be None (no constraint) or constrains the
        count and transition matrices to states with positive stationary vector entries.

        :getter: Retrieves the currently configured constraint, can be None.
        :setter: Sets a stationary distribution constraint by giving a stationary vector as value. The estimated count-
                 and transition-matrices are restricted to states that have positive entries. In case the vector is
                 not normalized, setting it here implicitly copies and normalizes it.
        :type: ndarray or None
        """
        return self._stationary_distribution_constraint

    @stationary_distribution_constraint.setter
    def stationary_distribution_constraint(self, value: Optional[np.ndarray]):
        if value is not None and (np.any(value < 0) or np.any(value > 1)):
            raise ValueError("not a distribution, contained negative entries and/or entries > 1.")
        if value is not None and np.sum(value) != 1.0:
            # re-normalize if not already normalized
            value = np.copy(value) / np.sum(value)
        self._stationary_distribution_constraint = value

    def fetch_model(self) -> Optional[BayesianMSMPosterior]:
        r"""
        Yields the model that was estimated the most recent.

        Returns
        -------
        model : BayesianMSMPosterior or None
            The estimated model or None if fit was not called.
        """
        return self._model

    def fit(self, data, callback: Callable = None, **kw):
        """
        Performs the estimation on either a count matrix or a previously estimated TransitionCountModel.

        Parameters
        ----------
        data : (N,N) count matrix or TransitionCountModel or MaximumLikelihoodMSM or MarkovStateModel
            a count matrix or a transition count model that was estimated from data

        callback: callable, optional, default=None
            Function to be called to indicate progress of sampling.

        Other Parameters
        ----------------
        ignore_counting_mode : bool, optional, default=False
            Method does not raise if counting mode isn't of the "effective" family. Use with caution.

        Returns
        -------
        self : BayesianMSM
            Reference to self.
        """
        if isinstance(data, Estimator):
            if data.has_model:
                data = data.fetch_model()
            else:
                raise ValueError("Can only use estimators as input if they have been fit previously.")

        from deeptime.markov import TransitionCountModel
        if isinstance(data, TransitionCountModel) or is_square_matrix(data):
            return self.fit_from_counts(data)
        elif isinstance(data, MarkovStateModel):
            return self.fit_from_msm(data, callback=callback, **kw)
        else:
            if not self.lagtime and 'lagtime' not in kw.keys():
                raise ValueError("To fit directly from a discrete timeseries, a lagtime must be provided!")
            return self.fit_from_discrete_timeseries(data, kw.pop('lagtime', self.lagtime), callback=callback, **kw)

    def sample(self, prior: MarkovStateModel, n_samples: int, n_steps: Optional[int] = None, callback=None):
        r""" Performs sampling based on a prior.

        Parameters
        ----------
        prior : MarkovStateModel
            The MSM that is used as initial sampling point.
        n_samples : int
            The number of samples to draw.
        n_steps : int, optional, default=None
            The number of sampling steps for each transition matrix. If None, determined
            by :math:`\sqrt{\mathrm{n\_states}}`.
        callback : callable, optional, default=None
            Callback function that indicates progress of sampling.

        Returns
        -------
        samples : list of :obj:`MarkovStateModel`
            The generated samples

        Examples
        --------
        This method can in particular be used to append samples to an already estimated posterior:

        >>> import numpy as np
        >>> import deeptime as dt
        >>> dtrajs = [np.array([0,1,2,2,2,2,1,2,2,2,1,0,0,0,0,0,0,0]),
        ...           np.array([0,0,0,0,1,1,2,2,2,2,2,2,2,1,0,0])]
        >>> prior = dt.markov.msm.MaximumLikelihoodMSM().fit(dtrajs, lagtime=1)
        >>> estimator = dt.markov.msm.BayesianMSM()
        >>> posterior = estimator.fit(prior).fetch_model()
        >>> n_samples = len(posterior.samples)
        >>> posterior.samples.extend(estimator.sample(posterior.prior, n_samples=23))
        >>> assert len(posterior.samples) == n_samples + 23
        """
        if n_steps is None:
            # heuristic for number of steps to decorrelate
            n_steps = int(sqrt(prior.count_model.n_states_full))
        # transition matrix sampler
        from deeptime.markov.tools.estimation import tmatrix_sampler
        if self.stationary_distribution_constraint is None:
            tsampler = tmatrix_sampler(prior.count_model.count_matrix, reversible=self.reversible,
                                       T0=prior.transition_matrix, nsteps=n_steps)
        else:
            # Use the stationary distribution on the active set of states
            statdist_active = prior.stationary_distribution
            # We can not use the MLE as T0. Use the initialization in the reversible pi sampler
            tsampler = tmatrix_sampler(prior.count_model.count_matrix, reversible=self.reversible,
                                       mu=statdist_active, nsteps=n_steps)
        sample_Ps, sample_mus = tsampler.sample(nsamples=n_samples, return_statdist=True, callback=callback)
        # construct sampled MSMs
        samples = [
            MarkovStateModel(P, stationary_distribution=pi, reversible=self.reversible,
                             count_model=prior.count_model,
                             transition_matrix_tolerance=prior.transition_matrix_tolerance)
            for P, pi in zip(sample_Ps, sample_mus)
        ]
        return samples

    def fit_from_msm(self, msm: MarkovStateModel, callback=None, **kw):
        r""" Fits a bayesian posterior from a given Markov state model. The MSM must contain a count model to be able
        to produce confidences. Note that the count model should be produced using effective counting, otherwise
        counts are correlated and computed confidences are wrong.

        Parameters
        ----------
        msm : MarkovStateModel
            The Markov state model to use as sampling start point.
        callback : callable, optional, default=None
            Function to be called to indicate progress of sampling.

        Other Parameters
        ----------------
        ignore_counting_mode : bool, optional, default=False
            Method does not raise if counting mode isn't of the "effective" family. Use with caution.

        Returns
        -------
        self : BayesianMSM
            Reference to self.
        """
        if not msm.has_count_model:
            raise ValueError("Can only sample confidences with a count model. The counting mode should be 'effective'"
                             " to avoid correlations between counts and therefore wrong confidences.")
        if not kw.get("ignore_counting_mode", False) \
                and msm.count_model.counting_mode is not None and "effective" not in msm.count_model.counting_mode:
            raise ValueError("The transition count model was not estimated using an effective counting method, "
                             "therefore counts are likely to be strongly correlated yielding wrong confidences. "
                             "To ignore this, set `ignore_counting_mode` to True in the call to `fit`.")
        # use the same count matrix as the MLE. This is why we have effective as a default
        samples = self.sample(msm, self.n_samples, self.n_steps, callback)
        self._model = BayesianMSMPosterior(prior=msm, samples=samples)
        return self

    def fit_from_discrete_timeseries(self, discrete_timeseries: Union[np.ndarray, List[np.ndarray]],
                                     lagtime: int = None, count_mode: str = 'effective', callback=None, **kw):
        r""" Fits a BayesianMSM directly on timeseries data.

        Parameters
        ----------
        discrete_timeseries : list of ndarray
            Discrete trajectories.
        lagtime : int, optional, default=None
            The lagtime that is used for estimation. If None, uses the instance's lagtime attribute.
        count_mode : str, default='effective'
            The counting mode. Should be of the `effective` kind, otherwise the results may be heavily biased.
        callback : callable, optional, default=None
            Function to be called to indicate progress of sampling.
        **kw
            Optional keyword parameters.

        Returns
        -------
        self : BayesianMSM
            Reference to self.
        """
        counts = TransitionCountEstimator(lagtime=lagtime, count_mode=count_mode,
                                          sparse=self.sparse).fit_fetch(discrete_timeseries).submodel_largest()
        return self.fit_from_counts(counts, callback=callback, **kw)

    def fit_from_counts(self, counts, callback=None, **kw):
        r"""Fits a bayesian MSM on a count model or a count matrix.

        Parameters
        ----------
        counts : TransitionCountModel or (n, n) ndarray
            The transition counts.
        callback : callable, optional, default=None
            Function that is called to indicate progress of sampling.
        **kw
            Optional keyword parameters.

        Returns
        -------
        self : BayesianMSM
            Reference to self.
        """
        msm = MaximumLikelihoodMSM(
            reversible=self.reversible, stationary_distribution_constraint=self.stationary_distribution_constraint,
            sparse=self.sparse, maxiter=self.maxiter, maxerr=self.maxerr
        ).fit(counts).fetch_model()
        return self.fit_from_msm(msm, callback=callback, **kw)

    @deprecated_method("Deprecated in v0.4.1 and will be removed soon, please use model.ck_test.")
    def chapman_kolmogorov_validator(self, n_metastable_sets: int, mlags, test_model=None):
        r""" Replaced by `deeptime.markov.msm.BayesianMSMPosterior.ck_test`. """
        test_model = self.fetch_model() if test_model is None else test_model
        assert test_model is not None, "We need a test model via argument or an estimator which was already" \
                                       "fit to data."
        prior = test_model.prior
        assert prior.has_count_model, "The test model needs to have a count model, i.e., be estimated from data."
        pcca = prior.pcca(n_metastable_sets)
        from deeptime.markov._observables import MembershipsObservable
        obs = MembershipsObservable(test_model, pcca)
        from deeptime.util.validation import DeprecatedCKValidator

        def fit_for_lag(data, lag):
            from deeptime.markov import TransitionCountEstimator
            counting_mode = test_model.prior.count_model.counting_mode
            counts = TransitionCountEstimator(lag, counting_mode).fit(data, n_jobs=1).fetch_model().submodel_largest()
            return self.fit(counts).fetch_model()

        return DeprecatedCKValidator(self, fit_for_lag, mlags, obs, test_model)

