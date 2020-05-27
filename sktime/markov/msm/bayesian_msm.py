from typing import Optional, Callable

import numpy as np

from sktime.markov._base import _MSMBaseEstimator, BayesianPosterior
from sktime.markov.msm import MarkovStateModel, MaximumLikelihoodMSM

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
        data : (N,N) count matrix or TransitionCountModel
            a count matrix or a transition count model that was estimated from data

        callback: callable, optional, default=None
            function to be called to indicate progress of sampling.

        Returns
        -------
        self : BayesianMSM
            Reference to self.
        """
        from sktime.markov import TransitionCountModel
        if isinstance(data, TransitionCountModel) and data.counting_mode is not None \
                and "effective" not in data.counting_mode:
            raise ValueError("The transition count model was not estimated using an effective counting method, "
                             "therefore counts are likely to be strongly correlated yielding wrong confidences.")
        mle = MaximumLikelihoodMSM(
            reversible=self.reversible, stationary_distribution_constraint=self.stationary_distribution_constraint,
            sparse=self.sparse, maxiter=self.maxiter, maxerr=self.maxerr
        ).fit(data).fetch_model()

        # transition matrix sampler
        from msmtools.estimation import tmatrix_sampler
        from math import sqrt
        if self.n_steps is None:
            # heuristic for number of steps to decorrelate
            self.n_steps = int(sqrt(mle.count_model.n_states_full))
        # use the same count matrix as the MLE. This is why we have effective as a default
        if self.stationary_distribution_constraint is None:
            tsampler = tmatrix_sampler(mle.count_model.count_matrix, reversible=self.reversible,
                                       T0=mle.transition_matrix, nsteps=self.n_steps)
        else:
            # Use the stationary distribution on the active set of states
            statdist_active = mle.stationary_distribution
            # We can not use the MLE as T0. Use the initialization in the reversible pi sampler
            tsampler = tmatrix_sampler(mle.count_model.count_matrix, reversible=self.reversible,
                                       mu=statdist_active, nsteps=self.n_steps)

        sample_Ps, sample_mus = tsampler.sample(nsamples=self.n_samples, return_statdist=True, call_back=callback)
        # construct sampled MSMs
        samples = [
            MarkovStateModel(P, stationary_distribution=pi, reversible=self.reversible, count_model=mle.count_model)
            for P, pi in zip(sample_Ps, sample_mus)
        ]

        self._model = BayesianPosterior(prior=mle, samples=samples)

        return self
