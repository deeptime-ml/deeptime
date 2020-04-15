# This file is part of PyEMMA.
#
# Copyright (c) 2014-2019 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

__author__ = 'Simon Olsson'

from typing import Optional

import numpy as np
from msmtools import estimation as msmest

from sktime.markov import TransitionCountModel
from sktime.markov.msm import MarkovStateModel
from sktime.util import confidence_interval
from .._base import _MSMBaseEstimator


class AMMOptimizerState(object):
    def __init__(self, E, m, w, pi, slices_z, symmetrized_count_matrix, count_matrix_row_sums):
        self.lagrange = np.zeros_like(m)
        self.pi = pi
        self.pi_hat = np.copy(pi)
        self.m_hat = None
        self.m = m
        self.w = w
        self.E = E
        self.slices_z = slices_z
        self.symmetrized_count_matrix = symmetrized_count_matrix
        self.count_matrix_row_sums = count_matrix_row_sums
        self.X = np.empty_like(symmetrized_count_matrix)

        self.m_hat = np.ndarray()
        self._slope_obs = None
        self.update_m_hat()
        self.d_m_hat = 1e-1 * np.ones_like(self.m_hat)
        self.R_slices = np.empty(())
        self.R_slices_i = 0
        self.update_R_slices(0)
        self.Q = np.zeros((self.n_states, self.n_states), dtype=E.dtype)
        self.G = np.empty(())
        self._log_likelihood_old = None
        self.log_likelihoods = []

    @property
    def log_likelihood_old(self):
        return self._log_likelihood_old

    @log_likelihood_old.setter
    def log_likelihood_old(self, value):
        self._log_likelihood_old = value

    @property
    def n_states(self):
        return self.E.shape[0]

    @property
    def n_experimental_observables(self):
        return self.E.shape[1]

    @property
    def slope_obs(self) -> np.ndarray:
        return self._slope_obs

    def update_m_hat(self):
        """ Updates m_hat (expectation of observable of the Augmented Markov model) """
        self.m_hat = self.pi_hat.dot(self.E)
        self._slope_obs = self.m_hat - self.m

    def update_R_slices(self, i):
        """ Computation of multiple slices of R tensor.

            When _estimate(.) is called the R-tensor is split into segments whose maximum size is
            specified by max_cache argument (see constructor).
            _Rsi specifies which of the segments are currently in cache.
             For equations check SI of [1].

        """
        pek = self.pi_hat[:, None] * self.E[:, i * self.slices_z:(i + 1) * self.slices_z]
        pp = self.pi_hat[:, None] + self.pi_hat[None, :]
        ppmhat = pp * self.m_hat[i * self.slices_z:(i + 1) * self.slices_z, None, None]
        self.R_slices = (pek[:, None, :] + pek[None, :, :]).T - ppmhat
        self.R_slices_i = i

    def update_pi_hat(self):
        r""" Update stationary distribution estimate of Augmented Markov model (\hat pi) """
        expons = np.einsum('i,ji->j', self.lagrange, self.E)
        # expons = (self.lagrange[:, None]*self.E_active.T).sum(axis=0)
        expons = expons - np.max(expons)

        _ph_unnom = self.pi * np.exp(expons)
        self.pi_hat = (_ph_unnom / _ph_unnom.sum()).reshape(-1, )

    def log_likelihood_biased(self, count_matrix, transition_matrix):
        """ Evaluate AMM likelihood. """
        ll_unbiased = msmest.log_likelihood(count_matrix, transition_matrix)
        ll_bias = -np.sum(self.w * (self.m_hat - self.E) ** 2.)
        return ll_unbiased + ll_bias

    def _get_Rk(self, k):
        """
          Convienence function to get cached value of an Rk slice of the R tensor.
          If we are outside cache, update the cache and return appropriate slice.

        """
        if k > (self.R_slices_i + 1) * self.slices_z or k < self.R_slices_i * self.slices_z:
            self.update_R_slices(np.floor(k / self.slices_z).astype(int))
            return self.R_slices[k % self.slices_z]
        else:
            return self.R_slices[k % self.slices_z]

    def update_Q(self):
        """ Compute Q, a weighted sum of the R-tensor.

            See SI of [1].
        """
        self.Q.fill(0.)
        for k in range(self.n_experimental_observables):
            self.Q = self.Q + self.w[k] * self.slope_obs[k] * self._get_Rk(k)
        self.Q *= -2.

    def update_X_and_pi(self):
        # evaluate count-over-pi
        c_over_pi = self.count_matrix_row_sums / self.pi
        D = c_over_pi[:, None] + c_over_pi + self.Q
        # update estimate
        self.X[:] = self.symmetrized_count_matrix / D

        # renormalize
        self.X /= np.sum(self.X)
        self.pi = np.sum(self.X, axis=1)

    def update_G(self):
        """ Update G.
            Observable covariance.
            See SI of [1].
        """
        self.G = (np.dot(self.E.T, self.E * self.pi_hat[:, None]) -
                  self.m_hat[:, None] * self.m_hat[None, :])


class AugmentedMSM(MarkovStateModel):

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


class AugmentedMSMEstimator(_MSMBaseEstimator):

    def __init__(self, E=None, m=None, w=None, eps=0.05, support_ci=1.00, maxiter=500, max_cache=3000):
        super().__init__(sparse=False, reversible=True)
        self.expectations_by_state = E
        self.experimental_measurements = m
        self.experimental_measurement_weights = w
        self.convergence_criterion_lagrange = eps
        self.support_confidence = support_ci
        self.max_cache = max_cache
        self.maxiter = maxiter

    @property
    def expectations_by_state(self):
        return self._E

    @expectations_by_state.setter
    def expectations_by_state(self, value):
        self._E = value

    @property
    def experimental_measurements(self):
        return self._m

    @experimental_measurements.setter
    def experimental_measurements(self, value):
        self._m = value

    @property
    def experimental_measurement_weights(self):
        return self._w

    @experimental_measurement_weights.setter
    def experimental_measurement_weights(self, value):
        if np.any(value < 1e-12):
            raise ValueError("Some weights are close to zero or negative, but only weights greater or equal 1e-12 can"
                             "be dealt with appropriately.")
        self._w = value

    @property
    def uncertainties(self):
        if self.experimental_measurement_weights is not None:
            return np.sqrt(1. / 2. / self.w)
        else:
            return None

    @property
    def convergence_criterion_lagrange(self):
        return self._eps

    @convergence_criterion_lagrange.setter
    def convergence_criterion_lagrange(self, value):
        self._eps = value

    @property
    def support_confidence(self):
        return self._support_ci

    @support_confidence.setter
    def support_confidence(self, value):
        self._support_ci = value

    @property
    def max_cache(self):
        return self._max_cache

    @max_cache.setter
    def max_cache(self, value):
        self._max_cache = value

    @property
    def maxiter(self):
        return self._maxiter

    @maxiter.setter
    def maxiter(self, value):
        self._maxiter = value

    def _newton_lagrange(self, state: AMMOptimizerState, count_matrix):
        """
          This function performs a Newton update of the Lagrange multipliers.
          The iteration is constrained by strictly improving the AMM likelihood, and yielding meaningful stationary
          properties.

          TODO: clean up and optimize code.
        """
        # initialize a number of values
        l_old = state.lagrange.copy()
        _ll_new = -np.inf
        frac = 1.
        mhat_old = state.m_hat.copy()
        while state.log_likelihood_old > _ll_new or np.any(state.pi_hat < 1e-12):
            state.update_pi_hat()
            state.update_G()
            # Lagrange slope calculation
            dl = 2. * (frac * state.G * state.w[:, None] * state.slope_obs[:, None]).sum(axis=0)
            # update Lagrange multipliers
            state.lagrange = l_old - frac * dl
            state.update_pi_hat()
            # a number of sanity checks
            while np.any(state.pi_hat < 1e-12) and frac > 0.05:
                frac = frac * 0.5
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
                self.logger.info("Small gradient fraction")
                break

            state.d_m_hat = state.m_hat - mhat_old
            state.log_likelihood_old = float(_ll_new)

        state.log_likelihoods.append(_ll_new)

    def fetch_model(self) -> Optional[AugmentedMSM]:
        return self._model

    def fit(self, data, *args, **kw):
        if not isinstance(data, (TransitionCountModel, np.ndarray)):
            raise ValueError("Can only fit on a TransitionCountModel or a count matrix directly.")

        if isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[0] != data.shape[1] or np.any(data < 0.):
                raise ValueError("If fitting a count matrix directly, only non-negative square matrices can be used.")
            count_model = TransitionCountModel(data)
        else:
            count_model = data

        if len(self.experimental_measurement_weights) != count_model.n_states_full:
            raise ValueError("Experimental weights must span full state space. In case some are excluded from "
                             "estimation due to connectivity issues they can be set to an arbitrary positive value.")
        if len(self.experimental_measurements) != count_model.n_states_full:
            raise ValueError("Experimental measurements must span full state space. In case some are excluded from "
                             "estimation due to connectivity issues they can be set to an arbitrary value.")

        # slice out active states from E matrix
        expectations_selected = self.expectations_by_state[count_model.state_symbols]
        count_matrix_symmetric = 0.5 * (count_model.count_matrix + count_model.count_matrix.T)
        nonzero_counts = np.nonzero(count_matrix_symmetric)
        counts_row_sums = np.sum(count_model.count_matrix, axis=1)
        expectations_confidence_interval = confidence_interval(expectations_selected, conf=self.support_confidence)

        measurements = self.experimental_measurements[count_model.state_symbols]
        measurement_weights = self.experimental_measurement_weights[count_model.state_symbols]

        count_outside = []
        count_inside = []

        i = 0
        # Determine which experimental values are outside the support as defined by the Confidence interval
        for emi, ema, mm, mw in zip(expectations_confidence_interval[0], expectations_confidence_interval[1],
                                    measurements, measurement_weights):
            if mm < emi or ema < mm:
                self.logger.info("Experimental value %f is outside the support (%f,%f)" % (mm, emi, ema))
                count_outside.append(i)
            else:
                count_inside.append(i)
            i = i + 1

        # A number of initializations
        transition_matrix, stationary_distribution = msmest.tmatrix(count_model.count_matrix, reversible=True,
                                                                    return_statdist=True)
        # Determine number of slices of R-tensors computable at once with the given cache size
        slices_z = np.floor(self.max_cache / (transition_matrix.nbytes / 1.e6)).astype(int)
        # Optimizer state
        state = AMMOptimizerState(expectations_selected, measurements, measurement_weights,
                                  stationary_distribution, slices_z, count_matrix_symmetric, counts_row_sums)
        ll_old = state.log_likelihood_biased(count_model.count_matrix, transition_matrix)

        state.log_likelihoods.append(ll_old)
        # make sure everything is initialized
        state.update_pi_hat()
        state.update_m_hat()
        state.update_Q()
        state.update_X_and_pi()

        ll_old = state.log_likelihood_biased(count_model.count_matrix, transition_matrix)
        state.log_likelihood_old = ll_old
        state.update_G()

        #
        # Main estimation algorithm
        # 2-step algorithm, lagrange multipliers and pihat have different convergence criteria
        # when the lagrange multipliers have converged, pihat is updated until the log-likelihood has converged (changes are smaller than 1e-3).
        # These do not always converge together, but usually within a few steps of each other.
        # A better heuristic for the latter may be necessary. For realistic cases (the two ubiquitin examples in [1])
        # this yielded results very similar to those with more stringent convergence criteria (changes smaller than 1e-9) with convergence times
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
                self.logger.warning("pihat does not have a finite probability for all states, terminating")
            state.update_m_hat()
            state.update_Q()

            if i > 1:
                X_old = np.copy(state.X)
                state.update_X_and_pi()
                if np.any(state.X[nonzero_counts] < 0) and i > 0:
                    die = True
                    self.logger.warning(
                        "Warning: new X is not proportional to C... reverting to previous step and terminating")
                    state.X = X_old

            if not converged:
                self._newton_lagrange(state, count_model.count_matrix)
            else:  # once Lagrange multipliers are converged compute likelihood here
                transition_matrix = state.X / state.pi[:, None]
                _ll_new = state.log_likelihood_biased(count_model.count_matrix, transition_matrix)
                state.log_likelihoods.append(_ll_new)

            # General case fixed-point iteration
            if len(count_outside) > 0:
                if i > 1 and np.all((np.abs(state.d_m_hat) / self.uncertainties) < self.convergence_criterion_lagrange) \
                        and not converged:
                    self.logger.info("Converged Lagrange multipliers after %i steps..." % i)
                    converged = True
            # Special case
            else:
                if np.abs(state.log_likelihoods[-2] - state.log_likelihoods[-1]) < 1e-8:
                    self.logger.info("Converged Lagrange multipliers after %i steps..." % i)
                    converged = True
            # if Lagrange multipliers are converged, check whether log-likelihood has converged
            if converged and np.abs(state.log_likelihoods[-2] - state.log_likelihoods[-1]) < 1e-8:
                self.logger.info("Converged pihat after %i steps..." % i)
                die = True
            if die:
                break
            if i == self.maxiter:
                self.logger.info("Failed to converge within %i iterations. "
                                 "Consider increasing max_iter(now=%i)" % (i, self.max_iter))
            i += 1

        transition_matrix = msmest.tmatrix(count_model.count_matrix, reversible=True, mu=state.pi_hat)
        self._model = AugmentedMSM(transition_matrix=transition_matrix, stationary_distribution=state.pi_hat,
                                   reversible=True, count_model=count_model, amm_optimizer_state=state)
        return self
