from typing import Optional, Callable

import numpy as np

from deeptime.base import Estimator
from deeptime.markov._base import _MSMBaseEstimator, BayesianPosterior
from deeptime.markov.msm import MarkovStateModel, MaximumLikelihoodMSM
from deeptime.numeric import is_square_matrix

__author__ = 'noe, marscher, clonker'


class BayesianMSM(_MSMBaseEstimator):
    r""" Bayesian estimator for MSMs given discrete trajectory statistics.

    Implementation following :cite:`bmm-est-trendelkamp2015estimation`.

    References
    ----------
    .. bibliography:: /references.bib
        :style: unsrt
        :filter: docname in docnames
        :keyprefix: bmm-est-

    Example
    -------
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

    Furthermore the bayesian MSM posterior returned by :meth:`fetch_model` is able to
    compute the probability distribution and statistical models of all methods
    that are offered by the MSM object. This works as follows. The :meth:`BayesianPosterior.gather_stats` method
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
                 sparse: bool = False, confidence: float = 0.954, maxiter: int = int(1e6), maxerr: float = 1e-8):
        r"""
        Constructs a new Bayesian estimator for MSMs.

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
        confidence : float, optional, default=0.954
            Confidence interval. By default two sigma (95.4%) is used. Use 68.3% for one sigma, 99.7% for three sigma.
        maxiter : int, optional, default=1000000
            Optional parameter with reversible = True, sets the maximum number of iterations before the transition
            matrix estimation method exits.
        maxerr : float, optional, default = 1e-8
            Optional parameter with reversible = True. Convergence tolerance for transition matrix estimation. This
            specifies the maximum change of the Euclidean norm of relative stationary probabilities
            (:math:`x_i = \sum_k x_{ik}`). The relative stationary probability changes
            :math:`e_i = (x_i^{(1)} - x_i^{(2)})/(x_i^{(1)} + x_i^{(2)})` are used in order to track changes in small
            probabilities. The Euclidean norm of the change vector, :math:`|e_i|_2`, is compared to maxerr.
        """

        super(BayesianMSM, self).__init__(reversible=reversible, sparse=sparse)
        self.stationary_distribution_constraint = stationary_distribution_constraint
        self.maxiter = maxiter
        self.maxerr = maxerr
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.confidence = confidence

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

    def fetch_model(self) -> BayesianPosterior:
        r"""
        Yields the model that was estimated the most recent.

        Returns
        -------
        model : BayesianPosterior or None
            The estimated model or None if fit was not called.
        """
        return self._model

    def fit(self, data, callback: Callable = None):
        """
        Performs the estimation on either a count matrix or a previously estimated TransitionCountModel.

        Parameters
        ----------
        data : (N,N) count matrix or TransitionCountModel or MaximumLikelihoodMSM or MarkovStateModel
            a count matrix or a transition count model that was estimated from data

        callback: callable, optional, default=None
            Function to be called to indicate progress of sampling.

        Returns
        -------
        self : BayesianMSM
            Reference to self.
        """
        from deeptime.markov import TransitionCountModel
        if isinstance(data, TransitionCountModel) and data.counting_mode is not None \
                and "effective" not in data.counting_mode:
            raise ValueError("The transition count model was not estimated using an effective counting method, "
                             "therefore counts are likely to be strongly correlated yielding wrong confidences.")

        if isinstance(data, Estimator):
            if data.has_model:
                data = data.fetch_model()
            else:
                raise ValueError("Can only use estimators as input if they have been fit previously.")

        if isinstance(data, TransitionCountModel) or is_square_matrix(data):
            msm = MaximumLikelihoodMSM(
                reversible=self.reversible, stationary_distribution_constraint=self.stationary_distribution_constraint,
                sparse=self.sparse, maxiter=self.maxiter, maxerr=self.maxerr
            ).fit(data).fetch_model()
        elif isinstance(data, MarkovStateModel):
            msm = data
        else:
            raise ValueError("Unsupported input data, can only be count matrix (or TransitionCountModel, "
                             "TransitionCountEstimator) or a MarkovStateModel instance or an estimator producing " 
                             "Markov state models.")

        return self.fit_from_msm(msm, callback=callback)

    def fit_from_msm(self, msm: MarkovStateModel, callback=None):
        r""" Fits a bayesian posterior from a given Markov state model. The MSM must contain a count model to be able
        to produce confidences. Note that the count model should be produced using effective counting, otherwise
        counts are correlated and computed confidences are wrong.

        Parameters
        ----------
        msm : MarkovStateModel
            The Markov state model to use as sampling start point.
        callback : callable, optional, default=None
            Function to be called to indicate progress of sampling.

        Returns
        -------
        self : BayesianMSM
            Reference to self.
        """
        if not msm.has_count_model:
            raise ValueError("Can only sample confidences with a count model. The counting mode should be 'effective'"
                             " to avoid correlations between counts and therefore wrong confidences.")
        # transition matrix sampler
        from deeptime.markov.tools.estimation import tmatrix_sampler
        from math import sqrt
        if self.n_steps is None:
            # heuristic for number of steps to decorrelate
            self.n_steps = int(sqrt(msm.count_model.n_states_full))
        # use the same count matrix as the MLE. This is why we have effective as a default
        if self.stationary_distribution_constraint is None:
            tsampler = tmatrix_sampler(msm.count_model.count_matrix, reversible=self.reversible,
                                       T0=msm.transition_matrix, nsteps=self.n_steps)
        else:
            # Use the stationary distribution on the active set of states
            statdist_active = msm.stationary_distribution
            # We can not use the MLE as T0. Use the initialization in the reversible pi sampler
            tsampler = tmatrix_sampler(msm.count_model.count_matrix, reversible=self.reversible,
                                       mu=statdist_active, nsteps=self.n_steps)
        sample_Ps, sample_mus = tsampler.sample(nsamples=self.n_samples, return_statdist=True, call_back=callback)
        # construct sampled MSMs
        samples = [
            MarkovStateModel(P, stationary_distribution=pi, reversible=self.reversible,
                             count_model=msm.count_model,
                             transition_matrix_tolerance=msm.transition_matrix_tolerance)
            for P, pi in zip(sample_Ps, sample_mus)
        ]
        self._model = BayesianPosterior(prior=msm, samples=samples)
        return self
