__author__ = 'Simon Olsson, clonker'

import logging
from typing import Optional

import numpy as np
from deeptime.markov.tools import estimation as msmest
from scipy.sparse import issparse

from deeptime.markov import TransitionCountModel
from deeptime.markov.msm import MarkovStateModel
from .._base import _MSMBaseEstimator
from ...util.stats import confidence_interval
from ...util.types import ensure_dtraj_list, ensure_traj_list


class AMMOptimizerState:
    r""" State of the optimization in AMMs. Is later attached to the MSM object for evaluation purposes. """

    def __init__(self, expectations_by_state, experimental_measurements, experimental_measurement_weights,
                 stationary_distribution, tensor_cache_size, symmetrized_count_matrix, count_matrix_row_sums):
        r""" Creates a new AMM optimizer state.

        Parameters
        ----------
        expectations_by_state : (N, K) ndarray
            See :meth:`AugmentedMSMEstimator.__init__`.
        experimental_measurements : (K,) ndarray
            See :meth:`AugmentedMSMEstimator.__init__`.
        experimental_measurement_weights : (K,) ndarray
            See :meth:`AugmentedMSMEstimator.__init__`.
        stationary_distribution : (N, N) ndarray
            Reversibly estimated stationary distribution from count matrix.
        tensor_cache_size : int
            Number of slices of the R tensor simultaneously stored. This value is computed from the cache size
            set in :meth:`AugmentedMSMEstimator.__init__`.
        symmetrized_count_matrix : (N, N) ndarray
            Symmetrized count matrix :math:`C_\mathrm{sym} = 0.5 (C + C^\top)` .
        count_matrix_row_sums : (N, ) ndarray
            Row sums of count matrix.
        """
        self.lagrange = np.zeros_like(experimental_measurements)
        self.pi = stationary_distribution
        self.pi_hat = np.copy(stationary_distribution)
        self.m_hat = None
        self.experimental_measurements = experimental_measurements
        self.experimental_measurement_weights = experimental_measurement_weights
        self.expectations_by_state = expectations_by_state
        self.tensor_cache_size = tensor_cache_size
        self.symmetrized_count_matrix = symmetrized_count_matrix
        self.count_matrix_row_sums = count_matrix_row_sums
        self.X = np.empty_like(symmetrized_count_matrix)

        self.m_hat = np.empty(())
        self._slope_obs = None
        self.update_m_hat()
        self.delta_m_hat = 1e-1 * np.ones_like(self.m_hat)
        self.R_slices = np.empty(())
        self.R_slices_i = 0
        self.update_R_slices(0)
        self.Q = np.zeros((self.n_states, self.n_states), dtype=expectations_by_state.dtype)
        self.G = np.empty(())
        self._log_likelihood_prev = None
        self.log_likelihoods = []

    @property
    def log_likelihood_prev(self):
        r""" Property storing the previous log likelihood. Used in optimization procedure.

        :getter: gets the previous log likelihood
        :setter: sets the previous log likelihood
        :type: float
        """
        return self._log_likelihood_prev

    @log_likelihood_prev.setter
    def log_likelihood_prev(self, value):
        self._log_likelihood_prev = value

    @property
    def n_states(self):
        r""" Number of discrete states (N) that fall into the considered state set. """
        return self.expectations_by_state.shape[0]

    @property
    def n_experimental_observables(self):
        r""" Number of experimental observables (K). """
        return self.expectations_by_state.shape[1]

    @property
    def slope_obs(self) -> np.ndarray:
        r""" Numerically computed slope for Newton algorithm. """
        return self._slope_obs

    def update_m_hat(self):
        r""" Updates m_hat (expectation of observable of the Augmented Markov model) """
        self.m_hat = self.pi_hat.dot(self.expectations_by_state)
        self._slope_obs = self.m_hat - self.experimental_measurements

    def update_R_slices(self, i):
        """ Computation of multiple slices of R tensor.

        When estimate(.) is called the R-tensor is split into segments whose maximum size is
        specified by max_cache argument (see constructor).
        R_slices_i specifies which of the segments are currently in cache. For equations check SI of [1].

        """
        pek = self.pi_hat[:, None] * self.expectations_by_state[:, i * self.tensor_cache_size:(i + 1) * self.tensor_cache_size]
        pp = self.pi_hat[:, None] + self.pi_hat[None, :]
        ppmhat = pp * self.m_hat[i * self.tensor_cache_size:(i + 1) * self.tensor_cache_size, None, None]
        self.R_slices = (pek[:, None, :] + pek[None, :, :]).T - ppmhat
        self.R_slices_i = i

    def update_pi_hat(self):
        r""" Update stationary distribution estimate of Augmented Markov model ( :math:`\hat pi` ) """
        expons = np.einsum('i,ji->j', self.lagrange, self.expectations_by_state)
        # expons = (self.lagrange[:, None]*self.E_active.T).sum(axis=0)
        expons = expons - np.max(expons)

        _ph_unnom = self.pi * np.exp(expons)
        self.pi_hat = (_ph_unnom / _ph_unnom.sum()).reshape(-1, )

    def log_likelihood_biased(self, count_matrix, transition_matrix):
        """ Evaluate AMM likelihood. """
        ll_unbiased = msmest.log_likelihood(count_matrix, transition_matrix)
        ll_bias = -np.sum(self.experimental_measurement_weights * (self.m_hat - self.experimental_measurements) ** 2.)
        return ll_unbiased + ll_bias

    def _get_Rk(self, k):
        """ Convenience function to get cached value of an Rk slice of the R tensor.
        If we are outside cache, update the cache and return appropriate slice.

        Parameters
        ----------
        k : int
            k-th slice.

        Returns
        -------
            The k-th slice of the R tensor.
        """
        if k > (self.R_slices_i + 1) * self.tensor_cache_size or k < self.R_slices_i * self.tensor_cache_size:
            self.update_R_slices(np.floor(k / self.tensor_cache_size).astype(int))
            return self.R_slices[k % self.tensor_cache_size]
        else:
            return self.R_slices[k % self.tensor_cache_size]

    def update_Q(self):
        """ Compute Q, a weighted sum of the R-tensor.

        See SI of [1].
        """
        self.Q.fill(0.)
        for k in range(self.n_experimental_observables):
            self.Q = self.Q + self.experimental_measurement_weights[k] * self.slope_obs[k] * self._get_Rk(k)
        self.Q *= -2.

    def update_X_and_pi(self):
        r""" Updates estimate X and stationary distribution. """
        # evaluate count-over-pi
        c_over_pi = self.count_matrix_row_sums / self.pi
        D = c_over_pi[:, None] + c_over_pi + self.Q
        # update estimate
        self.X[:] = self.symmetrized_count_matrix / D

        # renormalize
        self.X /= np.sum(self.X)
        self.pi = np.sum(self.X, axis=1)

    def update_G(self):
        """ Update G, the observable covariance.

        See SI of [1].
        """
        self.G = (np.dot(self.expectations_by_state.T, self.expectations_by_state * self.pi_hat[:, None]) -
                  self.m_hat[:, None] * self.m_hat[None, :])


class AugmentedMSM(MarkovStateModel):
    r""" An augmented Markov state model.

    Implementation following :footcite:`olsson2017combining`.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, transition_matrix, stationary_distribution=None, reversible=None, n_eigenvalues=None, ncv=None,
                 count_model=None, amm_optimizer_state=None):
        super().__init__(transition_matrix=transition_matrix, stationary_distribution=stationary_distribution,
                         reversible=reversible, n_eigenvalues=n_eigenvalues, ncv=ncv, count_model=count_model)
        self._amm_optimizer_state = amm_optimizer_state
        # scoring and hmm coarse-graining not supported for augmented MSMs
        # it is recommended to set unsupported methods to None rather than raising, cf
        # https://docs.python.org/3/library/exceptions.html
        self.score = None
        self.hmm = None

    @property
    def optimizer_state(self):
        r""" The optimizer state after optimization. Can be None if instantiated directly. """
        return self._amm_optimizer_state


class AugmentedMSMEstimator(_MSMBaseEstimator):
    r""" Estimator for augmented Markov state models. :footcite:`olsson2017combining`
    This estimator is based on expectation values from experiments.
    In case the experimental data is a time series matching a discrete time series, a convenience function
    :meth:`estimator_from_feature_trajectories` is offered.

    Parameters
    ----------
    expectations_by_state : (n, k) ndarray
        Expectations by state. n Markov states, k experimental observables; each index is
        average over members of the Markov state.
    experimental_measurements : (k,) ndarray
        The experimental measurements.
    experimental_measurement_weights : (k,) ndarray
        Experimental measurements weights.
    eps : float, optional, default=0.05
        Additional convergence criterion used when some experimental data
        are outside the support of the simulation. The value of the eps
        parameter is the threshold of the relative change in the predicted
        observables as a function of fixed-point iteration:

        .. math::
            \mathrm{eps} > \frac{\mid o_{\mathrm{pred}}^{(i+1)}-o_{\mathrm{pred}}^{(i)}\mid }{\sigma}.

    support_ci : float, optional, default=1.0
        Confidence interval for determination whether experimental data are inside or outside Markov model support.
    maxiter : int, optional, default=500
        Optional parameter with specifies the maximum number of updates for Lagrange multiplier estimation.
    max_cache : int, optional, default=3000
        Maximum size (in megabytes) of cache when computing R tensor.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, expectations_by_state, experimental_measurements, experimental_measurement_weights,
                 eps=0.05, support_ci=1.00, maxiter=500, max_cache=3000):
        super().__init__(sparse=False, reversible=True)
        self.expectations_by_state = expectations_by_state
        self.experimental_measurements = experimental_measurements
        self.experimental_measurement_weights = experimental_measurement_weights
        self.convergence_criterion_lagrange = eps
        self.support_confidence = support_ci
        self.max_cache = max_cache
        self.maxiter = maxiter
        self._log = logging.getLogger(__name__)

    @staticmethod
    def estimator_from_feature_trajectories(discrete_trajectories, feature_trajectories, n_states,
                                            experimental_measurements, sigmas, eps=0.05, support_ci=1.0,
                                            maxiter=500, max_cache=3000):
        r""" Creates an AMM estimator from discrete trajectories and corresponding experimental data.
        
        Parameters
        ----------
        discrete_trajectories : array_like or list of array_like
            Discrete trajectories, stored as integer ndarrays (arbitrary size) or a single ndarray for
            only one trajectory.
        feature_trajectories : array_like or list of array_like
            The same shape (number of trajectories and timesteps) as dtrajs. Each timestep in each trajectory should
            match the shape of the measurements and sigmas, K.
        n_states : int
            Number of markov states in full state space.
        experimental_measurements : (K,) ndarray
            Experimental averages.
        sigmas : (K,) ndarray
            Standard error for each experimental observable.
        eps : float, default = 0.05
            Convergence criterium, see :meth:`__init__`.
        support_ci : float, default=1.0
            Confidence interval, see :meth:`__init__`.
        maxiter : int, optional, default=500
            Maximum number of iterations.
        max_cache : int, optional, default=3000
            Parameter which specifies the maximum size of cache used
            when performing estimation of AMM, in megabytes.

        Returns
        -------
        estimator : AugmentedMSMEstimator
            An estimator parameterized expectations by state based on feature trajectories.
        """
        discrete_trajectories = ensure_dtraj_list(discrete_trajectories)
        feature_trajectories = ensure_traj_list(feature_trajectories)
        # check input
        if np.all(sigmas > 0):
            _w = 1. / (2 * sigmas ** 2.)
        else:
            raise ValueError('Zero or negative standard errors supplied. Please revise input')
        if feature_trajectories[0].ndim < 2:
            raise ValueError("Supplied feature trajectories have inappropriate dimensions (%d), "
                             "should be at least 2." % feature_trajectories[0].ndim)
        if len(discrete_trajectories) != len(feature_trajectories):
            raise ValueError("A different number of dtrajs and ftrajs were supplied as input. "
                             "They must have exactly a one-to-one correspondence.")
        if not np.all([len(dt) == len(ft) for dt, ft in zip(discrete_trajectories, feature_trajectories)]):
            raise ValueError("One or more supplied dtraj-ftraj pairs do not have the same length.")
        # MAKE E matrix
        fta = np.concatenate(feature_trajectories)
        dta = np.concatenate(discrete_trajectories)
        _E = np.zeros((n_states, fta.shape[1]))
        for i, s in enumerate(range(n_states)):
            indices = np.where(dta == s)
            if len(indices[0]) > 0:
                _E[i, :] = fta[indices].mean(axis=0)
        # transition matrix estimator
        return AugmentedMSMEstimator(expectations_by_state=_E, experimental_measurements=experimental_measurements,
                                     experimental_measurement_weights=_w, eps=eps,
                                     support_ci=support_ci, maxiter=maxiter, max_cache=max_cache)

    @property
    def expectations_by_state(self):
        r""" The expectations by state (N) for each observable (K).

        :type: (N, K) ndarray
        """
        return self._E

    @expectations_by_state.setter
    def expectations_by_state(self, value):
        self._E = value

    @property
    def experimental_measurements(self):
        r""" Experimental measurement averages (K).

        :type: (K,) ndarray
        """
        return self._m

    @experimental_measurements.setter
    def experimental_measurements(self, value):
        self._m = value

    @property
    def experimental_measurement_weights(self):
        r""" Weights for experimental measurement averages (K).

        :type: (K,) ndarray
        """
        return self._w

    @experimental_measurement_weights.setter
    def experimental_measurement_weights(self, value):
        if np.any(value < 1e-12):
            raise ValueError("Some weights are close to zero or negative, but only weights greater or equal 1e-12 can"
                             "be dealt with appropriately.")
        self._w = value

    @property
    def uncertainties(self):
        r""" Uncertainties based on measurement weights. """
        if self.experimental_measurement_weights is not None:
            return np.sqrt(1. / 2. / self.w)
        else:
            return None

    @property
    def convergence_criterion_lagrange(self):
        r""" Additional convergence criterion used when some experimental data
        are outside the support of the simulation. The value of the eps
        parameter is the threshold of the relative change in the predicted
        observables as a function of fixed-point iteration:

        $$ \mathrm{eps} > \frac{\mid o_{\mathrm{pred}}^{(i+1)}-o_{\mathrm{pred}}^{(i)}\mid }{\sigma}. $$
        """
        return self._eps

    @convergence_criterion_lagrange.setter
    def convergence_criterion_lagrange(self, value):
        self._eps = value

    @property
    def support_confidence(self):
        r""" Confidence interval size for markov states. """
        return self._support_ci

    @support_confidence.setter
    def support_confidence(self, value):
        self._support_ci = value

    @property
    def max_cache(self):
        r""" Cache size during computation. """
        return self._max_cache

    @max_cache.setter
    def max_cache(self, value):
        self._max_cache = value

    @property
    def maxiter(self):
        r""" Maximum number of Newton iterations. """
        return self._maxiter

    @maxiter.setter
    def maxiter(self, value):
        self._maxiter = value

    def _newton_lagrange(self, state: AMMOptimizerState, count_matrix):
        """ This function performs a Newton update of the Lagrange multipliers.
        The iteration is constrained by strictly improving the AMM likelihood, and yielding meaningful stationary
        properties.
        """
        # initialize a number of values
        l_old = state.lagrange.copy()
        _ll_new = -np.inf
        frac = 1.
        mhat_old = state.m_hat.copy()
        while state.log_likelihood_prev > _ll_new or np.any(state.pi_hat < 1e-12):
            state.update_pi_hat()
            state.update_G()
            # Lagrange slope calculation
            dl = 2. * (frac * state.G * state.experimental_measurement_weights[:, None]
                       * state.slope_obs[:, None]).sum(axis=0)
            # update Lagrange multipliers
            state.lagrange = l_old - frac * dl
            state.update_pi_hat()
            # a number of sanity checks
            while np.any(state.pi_hat < 1e-12) and frac > 0.05:
                frac *= 0.5
                state.lagrange = l_old - frac * dl
                state.update_pi_hat()

            state.lagrange = l_old - frac * dl
            state.update_pi_hat()
            state.update_m_hat()
            state.update_Q()
            state.update_X_and_pi()

            transition_matrix = state.X / state.pi[:, None]
            _ll_new = state.log_likelihood_biased(count_matrix, transition_matrix)
            # decrease slope in Lagrange space (only used if loop is repeated, e.g. if sanity checks fail)
            frac *= 0.1

            if frac < 1e-12:
                self._log.info("Small gradient fraction")
                break

            state.delta_m_hat = state.m_hat - mhat_old
            state.log_likelihood_prev = float(_ll_new)

        state.log_likelihoods.append(_ll_new)

    def fetch_model(self) -> Optional[AugmentedMSM]:
        r""" Yields the most recently estimated AMM or None if :meth:`fit` was not called yet.

        Returns
        -------
        amm : AugmentedMSM or None
            The AMM instance.
        """
        return self._model

    def fit(self, data, *args, **kw):
        r""" Fits an AMM.

        Parameters
        ----------
        data : TransitionCountModel or (N, N) ndarray
            Count matrix over data.
        *args
            scikit-learn compatibility argument
        **kw
            scikit-learn compatibility argument

        Returns
        -------
        self : AugmentedMSMEstimator
            Reference to self.
        """
        if not isinstance(data, (TransitionCountModel, np.ndarray)):
            raise ValueError("Can only fit on a TransitionCountModel or a count matrix directly.")

        if isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[0] != data.shape[1] or np.any(data < 0.):
                raise ValueError("If fitting a count matrix directly, only non-negative square matrices can be used.")
            count_model = TransitionCountModel(data)
        else:
            count_model = data

        if len(self.experimental_measurement_weights) != self.expectations_by_state.shape[1]:
            raise ValueError("Experimental weights must span full observable space.")
        if len(self.experimental_measurements) != self.expectations_by_state.shape[1]:
            raise ValueError("Experimental measurements must span full observable state space.")

        count_matrix = count_model.count_matrix
        if issparse(count_matrix):
            count_matrix = count_matrix.toarray()

        # slice out active states from E matrix
        expectations_selected = self.expectations_by_state[count_model.state_symbols]
        count_matrix_symmetric = 0.5 * (count_matrix + count_matrix.T)
        nonzero_counts = np.nonzero(count_matrix_symmetric)
        counts_row_sums = np.sum(count_matrix, axis=1)
        expectations_confidence_interval = confidence_interval(expectations_selected, conf=self.support_confidence)

        measurements = self.experimental_measurements
        measurement_weights = self.experimental_measurement_weights

        count_outside = []
        count_inside = []

        i = 0
        # Determine which experimental values are outside the support as defined by the Confidence interval
        for confidence_lower, confidence_upper, measurement, weight in zip(
                expectations_confidence_interval[0], expectations_confidence_interval[1],
                measurements, measurement_weights):
            if measurement < confidence_lower or confidence_upper < measurement:
                self._log.info(f"Experimental value {measurement} is outside the "
                               f"support ({confidence_lower, confidence_upper})")
                count_outside.append(i)
            else:
                count_inside.append(i)
            i = i + 1

        # A number of initializations
        transition_matrix, stationary_distribution = msmest.transition_matrix(count_matrix, reversible=True,
                                                                              return_statdist=True)
        if issparse(transition_matrix):
            transition_matrix = transition_matrix.toarray()
        # Determine number of slices of R-tensors computable at once with the given cache size
        slices_z = np.floor(self.max_cache / (transition_matrix.nbytes / 1.e6)).astype(int)
        # Optimizer state
        state = AMMOptimizerState(expectations_selected, measurements, measurement_weights,
                                  stationary_distribution, slices_z, count_matrix_symmetric, counts_row_sums)
        ll_old = state.log_likelihood_biased(count_matrix, transition_matrix)

        state.log_likelihoods.append(ll_old)
        # make sure everything is initialized
        state.update_pi_hat()
        state.update_m_hat()
        state.update_Q()
        state.update_X_and_pi()

        ll_old = state.log_likelihood_biased(count_matrix, transition_matrix)
        state.log_likelihood_prev = ll_old
        state.update_G()

        #
        # Main estimation algorithm
        # 2-step algorithm, lagrange multipliers and pihat have different convergence criteria
        # when the lagrange multipliers have converged, pihat is updated until the log-likelihood has converged
        # (changes are smaller than 1e-3).
        # These do not always converge together, but usually within a few steps of each other.
        # A better heuristic for the latter may be necessary. For realistic cases (the two ubiquitin examples in [1])
        # this yielded results very similar to those with more stringent convergence criteria
        # (changes smaller than 1e-9) with convergence times
        # which are seconds instead of tens of minutes.
        #

        converged = False  # Convergence flag for lagrange multipliers
        i = 0
        die = False
        while i <= self.maxiter:
            pi_hat_old = state.pi_hat.copy()
            state.update_pi_hat()
            if not np.all(state.pi_hat > 0):
                state.pi_hat = pi_hat_old.copy()
                die = True
                self._log.warning("pihat does not have a finite probability for all states, terminating")
            state.update_m_hat()
            state.update_Q()

            if i > 1:
                X_old = np.copy(state.X)
                state.update_X_and_pi()
                if np.any(state.X[nonzero_counts] < 0) and i > 0:
                    die = True
                    self._log.warning(
                        "Warning: new X is not proportional to C... reverting to previous step and terminating")
                    state.X = X_old

            if not converged:
                self._newton_lagrange(state, count_matrix)
            else:  # once Lagrange multipliers are converged compute likelihood here
                transition_matrix = state.X / state.pi[:, None]
                _ll_new = state.log_likelihood_biased(count_matrix, transition_matrix)
                state.log_likelihoods.append(_ll_new)

            # General case fixed-point iteration
            if len(count_outside) > 0:
                if i > 1 and np.all((np.abs(state.delta_m_hat) / self.uncertainties) < self.convergence_criterion_lagrange)\
                        and not converged:
                    self._log.info(f"Converged Lagrange multipliers after {i} steps...")
                    converged = True
            # Special case
            else:
                if np.abs(state.log_likelihoods[-2] - state.log_likelihoods[-1]) < 1e-8:
                    self._log.info(f"Converged Lagrange multipliers after {i} steps...")
                    converged = True
            # if Lagrange multipliers are converged, check whether log-likelihood has converged
            if converged and np.abs(state.log_likelihoods[-2] - state.log_likelihoods[-1]) < 1e-8:
                self._log.info(f"Converged pihat after {i} steps...")
                die = True
            if die:
                break
            if i == self.maxiter:
                ll_diff = np.abs(state.log_likelihoods[-2] - state.log_likelihoods[-1])
                self._log.info(f"Failed to converge within {i} iterations. Log-likelihoods lastly changed by {ll_diff}."
                               f" Consider increasing max_iter(now={self.max_iter})")
            i += 1

        transition_matrix = msmest.transition_matrix(count_matrix, reversible=True, mu=state.pi_hat)
        self._model = AugmentedMSM(transition_matrix=transition_matrix, stationary_distribution=state.pi_hat,
                                   reversible=True, count_model=count_model, amm_optimizer_state=state)
        return self
