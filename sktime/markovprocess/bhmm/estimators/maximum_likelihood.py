
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

import bhmm
import numpy as np

# BHMM imports
from sktime.base import Estimator
from sktime.markovprocess.bhmm.estimators._tmatrix_disconnected import estimate_P, stationary_distribution
from sktime.markovprocess.bhmm.hmm.generic_hmm import HMM
from .. import hidden, init_hmm


# TODO: reactivate multiprocessing, parallelize model fitting and forward-backward
# from multiprocessing import Queue, Process, cpu_count


class MaximumLikelihoodEstimator(Estimator):
    """
    Maximum likelihood Hidden Markov model (HMM).

    This class is used to fit a maximum-likelihood HMM to data.

    Examples
    --------

    >>> import bhmm
    >>> bhmm.config.verbose = False
    >>>
    >>> from bhmm import testsystems
    >>> [model, O, S] = testsystems.generate_synthetic_observations(ntrajectories=5, length=1000)
    >>> mlhmm = MaximumLikelihoodEstimator(O, model.nstates)
    >>> model = mlhmm.fit()

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
        super(MaximumLikelihoodEstimator, self).__init__(model=initial_model)
        if initial_model is None:
            self._output = output
            self._model = None

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

    def _forward_backward(self, obs, itraj):
        """
        Estimation step: Runs the forward-back algorithm on trajectory with index itraj

        Parameters
        ----------
        obs: np.ndarray
            single observation corresponding to index itraj
        itraj : int
            index of the observation trajectory to process

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
        self._model.output_model.p_obs(obs, out=self._pobs)
        # forward variables
        logprob, _ = hidden.forward(A, self._pobs, pi, T=T, alpha_out=self._alpha)
        # backward variables
        hidden.backward(A, self._pobs, T=T, beta_out=self._beta)
        # gamma
        hidden.state_probabilities(self._alpha, self._beta, T=T, gamma_out=self._gammas[itraj])
        # count matrix
        hidden.transition_counts(self._alpha, self._beta, A, self._pobs, T=T, out=self._Cs[itraj])
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
        gamma : [ ndarray(T,N, dtype=float) ]
            list of state probabilities for each trajectory
        count_matrix : [ ndarray(N,N, dtype=float) ]
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

    def _create_model(self) -> HMM:
        # If we already have a provided model (since the construction of the estimator,
        # we do not want to override it here).
        if self._model is None:
            pass
        else:
            return HMM()

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
        _Ts = [len(o) for o in observations]
        _maxT = np.max(_Ts)
        # pre-construct hidden variables
        self._alpha = np.zeros((_maxT, self._nstates))
        self._beta = np.zeros((_maxT, self._nstates))
        self._pobs = np.zeros((_maxT, self._nstates))
        self._gammas = [np.zeros((len(obs), self._nstates)) for obs in observations]
        self._Cs = [np.zeros((self._nstates, self._nstates)) for _ in range(len(observations))]

        if self._model is None:
            # Generate our own initial model.
            self._model = init_hmm(observations, self.nstates, output=self._output)

        it = 0
        likelihoods = np.zeros(self.maxit)
        loglik = 0.0
        # flag if connectivity has changed (e.g. state lost) - in that case the likelihood
        # is discontinuous and can't be used as a convergence criterion in that iteration.
        tmatrix_nonzeros = self._model.transition_matrix.nonzero()
        converged = False

        while not converged and it < self.maxit:
            loglik = 0.0
            for k, obs in enumerate(observations):
                loglik += self._forward_backward(obs, k)
                assert np.isfinite(loglik), it

            # convergence check
            if it > 0:
                dL = loglik - likelihoods[it-1]
                if dL < self._accuracy:
                    converged = True

            # update model
            self._update_model(observations, self._gammas, self._Cs, maxiter=self._maxit_P)

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
        # set final likelihood
        self._model._likelihood = loglik
        self._model._likelihoods = likelihoods
        self._model._gammas = self._gammas
        # set final count matrix
        self.count_matrix = self._transition_counts(self._Cs)
        self.initial_count = self._init_counts(self._gammas)
        del self._gammas

        # Compute hidden state trajectories using the Viterbi algorithm.
        self._model.hidden_state_trajectories = self._model.compute_viterbi_paths(observations)

        return self

    # TODO: reactive multiprocessing
    # ###################
    # # MULTIPROCESSING
    #
    # def _forward_backward_worker(self, work_queue, done_queue):
    #     try:
    #         for k in iter(work_queue.get, 'STOP'):
    #             (weight, gamma, count_matrix) = self._forward_backward(k)
    #             done_queue.put((k, weight, gamma, count_matrix))
    #     except Exception, e:
    #         done_queue.put(e.message)
    #     return True
    #
    #
    # def fit_parallel(self):
    #     """
    #     Maximum-likelihood estimation of the HMM using the Baum-Welch algorithm
    #
    #     Returns
    #     -------
    #     The hidden markov model
    #
    #     """
    #     K = len(self.observations)#, len(A), len(B[0])
    #     gammas = np.empty(K, dtype=object)
    #     count_matrices = np.empty(K, dtype=object)
    #
    #     it        = 0
    #     converged = False
    #
    #     num_threads = min(cpu_count(), K)
    #     work_queue = Queue()
    #     done_queue = Queue()
    #     processes = []
    #
    #     while (not converged):
    #         print "it", it
    #         loglik = 0.0
    #
    #         # fill work queue
    #         for k in range(K):
    #             work_queue.put(k)
    #
    #         # start processes
    #         for w in xrange(num_threads):
    #             p = Process(target=self._forward_backward_worker, args=(work_queue, done_queue))
    #             p.start()
    #             processes.append(p)
    #             work_queue.put('STOP')
    #
    #         # end processes
    #         for p in processes:
    #             p.join()
    #
    #         # done signal
    #         done_queue.put('STOP')
    #
    #         # get results
    #         for (k, ll, gamma, count_matrix) in iter(done_queue.get, 'STOP'):
    #             loglik += ll
    #             gammas[k] = gamma
    #             count_matrices[k] = count_matrix
    #
    #         # update T, pi
    #         self._update_model(gammas, count_matrices)
    #
    #         self.likelihoods[it] = loglik
    #
    #         if it > 0:
    #             if loglik - self.likelihoods[it-1] < self.accuracy:
    #                 converged = True
    #
    #         it += 1
    #
    #     return self.model
