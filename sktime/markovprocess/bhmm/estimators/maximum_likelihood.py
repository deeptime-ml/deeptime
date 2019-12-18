
# This file is part of BHMM (Bayesian Hidden Markov Models).
#
# Copyright (c) 2016 Frank Noe (Freie Universitaet Berlin)
# and John D. Chodera (Memorial Sloan-Kettering Cancer Center, New York)
#
# BHMM is free software: you can redistribute it and/or modify
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

import numpy as np

# BHMM imports
from sktime.base import Estimator
from sktime.markovprocess.bhmm.estimators._tmatrix_disconnected import estimate_P, stationary_distribution
from sktime.markovprocess.bhmm.hmm.generic_hmm import HMM
from .. import hidden, init_hmm


class MaximumLikelihoodHMM(Estimator):
    """
    Maximum likelihood Hidden Markov model (HMM).

    This class is used to fit a maximum-likelihood HMM to data.

    References
    ----------
    [1] L. E. Baum and J. A. Egon, "An inequality with applications to statistical
        estimation for probabilistic functions of a Markov process and to a model
        for ecology," Bull. Amer. Meteorol. Soc., vol. 73, pp. 360-363, 1967.

    """
    def __init__(self, nstates, initial_model=None, output='gaussian',
                 reversible=True, stationary=False, p=None, accuracy=1e-3, maxit=1000, maxit_P=100000):
        """Initialize a Bayesian hidden Markov model sampler.

        Parameters
        ----------
        nstates : int
            The number of states in the model.
        initial_model : HMM, optional, default=None
            If specified, the given initial model will be used to initialize the
            BHMM. Otherwise, a heuristic scheme is used to generate an initial guess.
        type : str, optional, default=None
            Output model type from [None, 'gaussian', 'discrete'].
        reversible : bool, optional, default=True
            If True, a prior that enforces reversible transition matrices (detailed
            balance) is used; otherwise, a standard  non-reversible prior is used.
        stationary : bool, optional, default=False
            If True, the initial distribution of hidden states is self-consistently
            computed as the stationary distribution of the transition matrix. If
            False, it will be estimated from the starting states.
        p : ndarray (nstates), optional, default=None
            Initial or fixed stationary distribution. If given and stationary=True,
            transition matrices will be estimated with the constraint that they
            have p as their stationary distribution. If given and stationary=False,
            p is the fixed initial distribution of hidden states.
        accuracy : float
            convergence threshold for EM iteration. When two the likelihood does
            not increase by more than accuracy, the iteration is stopped successfully.
        maxit : int
            stopping criterion for EM iteration. When so many iterations are
            performed without reaching the requested accuracy, the iteration is
            stopped without convergence (a warning is given)
        maxit_P : int
            maximum number of iterations for reversible transition matrix estimation.
            Only used with reversible=True.

        """
        # Use user-specified initial model, if provided.
        super(MaximumLikelihoodHMM, self).__init__()
        self.initial_model = initial_model
        if initial_model is None:  # remember choice of output model for later creation of the HMM model
            self._output = output

        # Set parameters
        self._nstates = nstates
        self._reversible = reversible
        self._stationary = stationary

        # stationary and initial distribution
        self._fixed_stationary_distribution = None
        self._fixed_initial_distribution = None
        if p is not None:
            p = np.array(p)
            if stationary:
                self._fixed_stationary_distribution = p
            else:
                self._fixed_initial_distribution = p

        # convergence options
        self._accuracy = accuracy
        self._maxit = maxit
        self._maxit_P = maxit_P

    @property
    def is_reversible(self):
        r""" Whether the transition matrix is estimated with detailed balance constraints """
        return self._reversible

    @property
    def nstates(self):
        r""" Number of hidden states """
        return self._nstates

    @property
    def accuracy(self):
        r""" Convergence threshold for EM iteration """
        return self._accuracy

    @property
    def maxit(self):
        r""" Maximum number of iterations """
        return self._maxit

    def _forward_backward(self, obs, alpha, beta, gamma, pobs, counts):
        """
        Estimation step: Runs the forward-back algorithm on trajectory obs

        Parameters
        ----------
        obs: np.ndarray
            single observation corresponding to index itraj

        Returns
        -------
        logprob : float
            The probability to observe the observation sequence given the HMM
            parameters
        """
        # get parameters
        A = self._model.transition_matrix
        pi = self._model.initial_distribution
        T = len(obs)
        # compute output probability matrix
        self._model.output_model.p_obs(obs, out=pobs)
        # forward variables
        logprob, _ = hidden.forward(A, pobs, pi, T=T, alpha=alpha)
        # backward variables
        hidden.backward(A, pobs, T=T, beta_out=beta)
        # gamma
        hidden.state_probabilities(alpha, beta, T=T, gamma_out=gamma)
        # count matrix
        hidden.transition_counts(alpha, beta, A, pobs, T=T, out=counts)
        return logprob

    def _init_counts(self, gammas):
        gamma0_sum = np.zeros(self._nstates)
        # update state counts
        for g in gammas:
            gamma0_sum += g[0]
        return gamma0_sum

    @staticmethod
    def _transition_counts(count_matrices):
        C = np.add.reduce(count_matrices)
        return C

    def _update_model(self, observations, gammas, count_matrices, maxiter=10000000):
        """
        Maximization step: Updates the HMM model given the hidden state assignment and count matrices

        Parameters
        ----------
        gammas : [ ndarray(T,N, dtype=float) ]
            list of state probabilities for each trajectory
        count_matrices : [ ndarray(N,N, dtype=float) ]
            list of the Baum-Welch transition count matrices for each hidden
            state trajectory
        maxiter : int
            maximum number of iterations of the transition matrix estimation if
            an iterative method is used.

        """
        gamma0_sum = self._init_counts(gammas)
        C = self._transition_counts(count_matrices)

        # compute new transition matrix
        T = estimate_P(C, reversible=self._model.is_reversible, fixed_statdist=self._fixed_stationary_distribution,
                       maxiter=maxiter, maxerr=1e-12, mincount_connectivity=1e-16)
        # estimate stationary or init distribution
        if self._stationary:
            if self._fixed_stationary_distribution is None:
                pi = stationary_distribution(T, C=C, mincount_connectivity=1e-16)
            else:
                pi = self._fixed_stationary_distribution
        else:
            if self._fixed_initial_distribution is None:
                pi = gamma0_sum / np.sum(gamma0_sum)
            else:
                pi = self._fixed_initial_distribution

        # update model
        self._model.update(pi, T)

        # update output model
        self._model.output_model.fit(observations, gammas)

    def fit(self, observations, **kw):
        """
        Maximum-likelihood estimation of the HMM using the Baum-Welch algorithm

        Parameters
        ----------
        observations : list of numpy arrays representing temporal data
            `observations[i]` is a 1d numpy array corresponding to the observed
            trajectory index `i`

        Returns
        -------
        model : HMM
            The maximum likelihood HMM model.

        """
        # TODO: type checking for observations
        _maxT = max(len(obs) for obs in observations)
        # pre-construct hidden variables
        N = self.nstates
        alpha = np.zeros((_maxT, N))
        beta = np.zeros((_maxT, N))
        pobs = np.zeros((_maxT, N))
        gammas = [np.zeros((len(obs), N)) for obs in observations]
        count_matrices = [np.zeros((N, N)) for _ in observations]

        if self.initial_model is None:
            # Generate our own initial model.
            self._model = init_hmm(observations, self.nstates, output=self._output)
        else:
            self._model = self.initial_model

        it = 0
        likelihoods = np.empty(self.maxit)
        loglik = 0.0
        # flag if connectivity has changed (e.g. state lost) - in that case the likelihood
        # is discontinuous and can't be used as a convergence criterion in that iteration.
        tmatrix_nonzeros = self._model.transition_matrix.nonzero()
        converged = False

        while not converged and it < self.maxit:
            loglik = 0.0
            for obs, gamma, counts in zip(observations, gammas, count_matrices):
                loglik += self._forward_backward(obs, alpha, beta, gamma, pobs, counts)
            assert np.isfinite(loglik), it

            # convergence check
            if it > 0:
                dL = loglik - likelihoods[it-1]
                if dL < self._accuracy:
                    converged = True

            # update model
            self._update_model(observations, gammas, count_matrices, maxiter=self._maxit_P)

            # connectivity change check
            tmatrix_nonzeros_new = self._model.transition_matrix.nonzero()
            if not np.array_equal(tmatrix_nonzeros, tmatrix_nonzeros_new):
                converged = False  # unset converged
                tmatrix_nonzeros = tmatrix_nonzeros_new

            # end of iteration
            likelihoods[it] = loglik
            it += 1

        # truncate likelihood history
        likelihoods = np.resize(likelihoods, it)
        m = self._model
        # set final likelihood
        m._likelihood = loglik
        m._likelihoods = likelihoods
        m._gammas = gammas
        # set final count matrix
        m.transition_counts = self._transition_counts(count_matrices)
        m.initial_count = self._init_counts(gammas)

        # Compute hidden state trajectories using the Viterbi algorithm.
        m.hidden_state_trajectories = self._model.compute_viterbi_paths(observations)

        return self
