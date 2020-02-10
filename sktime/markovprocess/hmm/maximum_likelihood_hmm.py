# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
import collections
import warnings
from typing import List, Union

import numpy as np
from scipy.sparse import issparse

from sktime.base import Estimator
from sktime.markovprocess import MarkovStateModel, TransitionCountModel
from sktime.markovprocess._transition_matrix import estimate_P, stationary_distribution
from sktime.markovprocess.bhmm import discrete_hmm, init_discrete_hmm
from sktime.markovprocess.bhmm.init.discrete import init_discrete_hmm_spectral
from sktime.markovprocess.hmm import HiddenMarkovStateModel
from sktime.markovprocess.util import compute_dtrajs_effective
from sktime.util import ensure_dtraj_list

_HMMModelStorage = collections.namedtuple('_HMMModelStorage', ['transition_matrix', 'output_model',
                                                               'initial_distribution'])

def _forward(A, pobs, pi, T=None, alpha=None):
    """Compute P( obs | A, B, pi ) and all forward coefficients.

    Parameters
    ----------
    A : ndarray((N,N), dtype = float)
        transition matrix of the hidden states
    pobs : ndarray((T,N), dtype = float)
        pobs[t,i] is the observation probability for observation at time t given hidden state i
    pi : ndarray((N), dtype = float)
        initial distribution of hidden states
    T : int, optional, default = None
        trajectory length. If not given, T = pobs.shape[0] will be used.
    alpha : ndarray((T,N), dtype = float), optional, default = None
        container for the alpha result variables. If None, a new container will be created.

    Returns
    -------
    logprob : float
        The probability to observe the sequence `ob` with the model given
        by `A`, `B` and `pi`.
    alpha : ndarray((T,N), dtype = float), optional, default = None
        alpha[t,i] is the ith forward coefficient of time t. These can be
        used in many different algorithms related to HMMs.

    """
    if T is None:
        T = len(pobs)  # if not set, use the length of pobs as trajectory length
    elif T > len(pobs):
        raise TypeError('T must be at most the length of pobs.')
    if alpha is None:
        alpha = np.zeros_like(pobs)
    elif T > len(alpha):
        raise TypeError('alpha must at least have length T in order to fit trajectory.')

    return _bindings.forward(A, pobs, pi, alpha, T)

class MaximumLikelihoodHMSM(Estimator):

    def __init__(self, initial_model: HiddenMarkovStateModel, stride: Union[int, str] = 1,
                 lagtime: int = 1, reversible: bool = False, accuracy=1e-3, maxit=1000, model=None):
        super().__init__(model=model)
        self.initial_transition_model = initial_model
        self.stride = stride
        self.lagtime = lagtime
        self.reversible = reversible
        self.accuracy = accuracy
        self.maxit = maxit

    @property
    def accuracy(self) -> float:
        return self._accuracy

    @accuracy.setter
    def accuracy(self, value: float):
        self._accuracy = float(value)

    @property
    def maxit(self) -> int:
        return self._maxit

    @maxit.setter
    def maxit(self, value: int):
        self._maxit = int(value)

    @property
    def reversible(self) -> bool:
        return self._reversible

    @reversible.setter
    def reversible(self, value: bool):
        self._reversible = bool(value)

    @property
    def stride(self) -> Union[int, str]:
        return self._stride

    @stride.setter
    def stride(self, value):
        if isinstance(value, str):
            if value != 'effective':
                raise ValueError("stride value can only be either integer or 'effective'.")
            else:
                self._stride = value
        else:
            self._stride = int(value)

    @property
    def lagtime(self) -> int:
        return self._lagtime

    @lagtime.setter
    def lagtime(self, value: int):
        value = int(value)
        if value <= 0:
            raise ValueError("Lagtime must be positive!")
        self._lagtime = value

    @property
    def n_hidden_states(self) -> int:
        return self._initial_model.n_hidden_states

    @property
    def initial_transition_model(self) -> HiddenMarkovStateModel:
        return self._initial_transition_model

    @initial_transition_model.setter
    def initial_transition_model(self, value: HiddenMarkovStateModel) -> None:
        self._initial_transition_model = value

    def fit(self, dtrajs, initial_model=None, **kwargs):
        if initial_model is None:
            initial_model = self.initial_transition_model
        if initial_model is None or not isinstance(initial_model, HiddenMarkovStateModel):
            raise ValueError("For estimation, an initial model of type "
                             "`sktime.markovprocess.hmm.HiddenMarkovStateModel` is required.")

        # copy initial model
        transition_matrix = initial_model.transition_model.transition_matrix
        if issparse(transition_matrix):
            # want dense matrix, toarray makes a copy
            transition_matrix = transition_matrix.toarray()
        else:
            # new instance
            transition_matrix = np.copy(transition_matrix)

        hmm_data = _HMMModelStorage(transition_matrix=transition_matrix, output_model=initial_model.output_model.copy(),
                                    initial_distribution=initial_model.initial_distribution.copy())

        dtrajs = ensure_dtraj_list(dtrajs)
        dtrajs = compute_dtrajs_effective(dtrajs, lagtime=self.lagtime, n_states=initial_model.n_hidden_states,
                                          stride=self.stride)

        max_n_frames = max(len(obs) for obs in dtrajs)
        # pre-construct hidden variables
        N = self.n_states
        alpha = np.zeros((max_n_frames, N))
        beta = np.zeros((max_n_frames, N))
        pobs = np.zeros((max_n_frames, N))
        gammas = [np.zeros((len(obs), N)) for obs in dtrajs]
        count_matrices = [np.zeros((N, N)) for _ in dtrajs]

        it = 0
        likelihoods = np.empty(self.maxit)
        # flag if connectivity has changed (e.g. state lost) - in that case the likelihood
        # is discontinuous and can't be used as a convergence criterion in that iteration.
        tmatrix_nonzeros = hmm_data.transition_matrix.nonzero()
        converged = False

        while not converged and it < self.maxit:
            loglik = 0.0
            for obs, gamma, counts in zip(dtrajs, gammas, count_matrices):
                loglik += self._forward_backward(hmm_data, obs, alpha, beta, gamma, pobs, counts)
            assert np.isfinite(loglik), it

            # convergence check
            if it > 0:
                dL = loglik - likelihoods[it - 1]
                if dL < self._accuracy:
                    converged = True

            # update model
            self._update_model(hmm_data, dtrajs, gammas, count_matrices, maxiter=self._maxit_P)

            # connectivity change check
            tmatrix_nonzeros_new = hmm_data.transition_matrix.nonzero()
            if not np.array_equal(tmatrix_nonzeros, tmatrix_nonzeros_new):
                converged = False  # unset converged
                tmatrix_nonzeros = tmatrix_nonzeros_new

            # end of iteration
            likelihoods[it] = loglik
            it += 1

        likelihoods = np.resize(likelihoods, it)

        transition_counts = self._reduce_transition_counts(count_matrices)

        count_model = TransitionCountModel(count_matrix=transition_counts, lagtime=self.lagtime,
                                           physical_time=self.physical_time)
        model = HiddenMarkovStateModel(
            transition_model=hmm_data.transition_matrix,
            output_model=hmm_data.output_model,
            initial_distribution=hmm_data.initial_distribution
        )
        # todo make ctor args
        model._count_model = count_model
        model._likelihoods = likelihoods
        model._gammas = gammas
        model._initial_count = self._init_counts(gammas)
        model._hidden_state_trajectories = model.compute_viterbi_paths(dtrajs)

        self._model = model
        return self

    @staticmethod
    def _forward_backward(model: _HMMModelStorage, obs, alpha, beta, gamma, counts):
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
        A = model.transition_matrix
        pi = model.initial_distribution
        T = len(obs)
        # compute output probability matrix
        pobs = model.output_model.to_state_probability_trajectory(obs)
        # forward variables
        logprob, _ = hidden.forward(A, pobs, pi, T=T, alpha=alpha)
        # backward variables
        hidden.backward(A, pobs, T=T, beta_out=beta)
        # gamma
        hidden.state_probabilities(alpha, beta, T=T, gamma_out=gamma)
        # count matrix
        hidden.transition_counts(alpha, beta, A, pobs, T=T, out=counts)
        return logprob, pobs

    def _init_counts(self, gammas):
        gamma0_sum = np.zeros(self.n_hidden_states)
        # update state counts
        for g in gammas:
            gamma0_sum += g[0]
        return gamma0_sum

    @staticmethod
    def _reduce_transition_counts(count_matrices):
        C = np.add.reduce(count_matrices)
        return C

    def _update_model(self, model: _HMMModelStorage, observations: List[np.ndarray], gammas: List[np.ndarray],
                      count_matrices: List[np.ndarray], maxiter: int = int(1e7)):
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
        C = self._reduce_transition_counts(count_matrices)

        # compute new transition matrix
        T = estimate_P(C, reversible=self.reversible, fixed_statdist=self._fixed_stationary_distribution,
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

        model.initial_distribution = pi
        model.transition_matrix = T
        model.output_model.fit(observations, gammas)


class MaximumLikelihoodHMSM2(Estimator):
    """
    Maximum likelihood hidden markov state model estimator.
    """

    def __init__(self, n_states=2, lagtime=1, stride=1, msm_init='largest-strong', reversible=True, stationary=False,
                 connectivity=None, observe_nonempty=True, separate=None,
                 physical_time='1 step', accuracy=1e-3, maxit=1000):
        r"""Maximum likelihood estimator for a Hidden MSM given a MSM

        Parameters
        ----------
        n_states : int, optional, default=2
            number of hidden states
        lag : int, optional, default=1
            lagtime to fit the HMSM at
        stride : str or int, default=1
            stride between two lagged trajectories extracted from the input
            trajectories. Given trajectory s[t], stride and lag will result
            in trajectories
                s[0], s[lag], s[2 lag], ...
                s[stride], s[stride + lag], s[stride + 2 lag], ...
            Setting stride = 1 will result in using all data (useful for maximum
            likelihood estimator), while a Bayesian estimator requires a longer
            stride in order to have statistically uncorrelated trajectories.
            Setting stride = 'effective' uses the largest neglected timescale as
            an fit for the correlation time and sets the stride accordingly
        msm_init : str or :class:`MSM <sktime.markovprocess.MarkovStateModel>`
            MSM object to initialize the estimation, or one of following keywords:

            * 'largest-strong' or None (default) : Estimate MSM on the largest
                strongly connected set and use spectral clustering to generate an
                initial HMM
            * 'all' : Estimate MSM(s) on the full state space to initialize the
                HMM. This fit may be weakly connected or disconnected.
        reversible : bool, optional, default = True
            If true compute reversible MSM, else non-reversible MSM
        stationary : bool, optional, default=False
            If True, the initial distribution of hidden states is self-consistently computed as the stationary
            distribution of the transition matrix. If False, it will be estimated from the starting states.
            Only set this to true if you're sure that the observation trajectories are initiated from a global
            equilibrium distribution.
        connectivity : str, optional, default = None
            Defines if the resulting HMM will be defined on all hidden states or on
            a connected subset. Connectivity is defined by counting only
            transitions with at least mincount_connectivity counts.
            If a subset of states is used, all estimated quantities (transition
            matrix, stationary distribution, etc) are only defined on this subset
            and are correspondingly smaller than n_states.
            Following modes are available:

            * None or 'all' : The active set is the full set of states.
              Estimation is done on all weakly connected subsets separately. The
              resulting transition matrix may be disconnected.
            * 'largest' : The active set is the largest reversibly connected set.
            * 'populous' : The active set is the reversibly connected set with most counts.
        separate : None or iterable of int
            Force the given set of observed states to stay in a separate hidden state.
            The remaining n_states-1 states will be assigned by a metastable decomposition.
        observe_nonempty : bool
            If True, will restricted the observed states to the states that have
            at least one observation in the lagged input trajectories.
            If an initial MSM is given, this option is ignored and the observed
            subset is always identical to the active set of that MSM.
        physical_time : str, optional, default='1 step'
            Description of the physical time corresponding to the trajectory time
            step.  May be used by analysis algorithms such as plotting tools to
            pretty-print the axes. By default '1 step', i.e. there is no physical
            time unit. Specify by a number, whitespace and unit. Permitted units
            are (* is an arbitrary string):

            |  'fs',  'femtosecond*'
            |  'ps',  'picosecond*'
            |  'ns',  'nanosecond*'
            |  'us',  'microsecond*'
            |  'ms',  'millisecond*'
            |  's',   'second*'

        accuracy : float, optional, default = 1e-3
            convergence threshold for EM iteration. When two the likelihood does
            not increase by more than accuracy, the iteration is stopped
            successfully.
        maxit : int, optional, default = 1000
            stopping criterion for EM iteration. When so many iterations are
            performed without reaching the requested accuracy, the iteration is
            stopped without convergence (a warning is given)

        """
        super(MaximumLikelihoodHMSM2, self).__init__()
        self.n_hidden_states = n_states
        self.lagtime = lagtime
        self.stride = stride
        self.msm_init = msm_init
        self.reversible = reversible
        self.stationary = stationary
        self.connectivity = connectivity
        self.separate = separate
        self.observe_nonempty = observe_nonempty
        self.physical_time = physical_time
        self.accuracy = accuracy
        self.maxit = maxit

    def fetch_model(self) -> HiddenMarkovStateModel:
        return self._model

    @staticmethod
    def initial_guess(dtrajs, lagtime, n_hidden_states, stride) -> HiddenMarkovStateModel:
        dtrajs = ensure_dtraj_list(dtrajs)
        dtrajs_lagged_strided = compute_dtrajs_effective(dtrajs, lagtime=lagtime,
                                                         n_states=n_hidden_states,
                                                         stride=stride)

    def fit(self, dtrajs, **kwargs):
        dtrajs = ensure_dtraj_list(dtrajs)
        # CHECK LAG
        trajlengths = [len(dtraj) for dtraj in dtrajs]
        if self.lagtime >= np.max(trajlengths):
            raise ValueError(f'Illegal lag time {self.lagtime}, needs to be smaller than longest input trajectory.')
        if self.lagtime > np.mean(trajlengths):
            warnings.warn(f'Lag time {self.lagtime} is on the order of mean trajectory length '
                          f'{np.mean(trajlengths)}. It is recommended to fit at least four lag times in each '
                          'trajectory. HMM might be inaccurate.')

        dtrajs_lagged_strided = compute_dtrajs_effective(dtrajs, lagtime=self.lagtime,
                                                         n_states=self.n_hidden_states,
                                                         stride=self.stride)

        # INIT HMM
        if isinstance(self.msm_init, str):
            args = dict(observations=dtrajs_lagged_strided, n_states=self.n_hidden_states, lag=1,
                        reversible=self.reversible, stationary=True, regularize=True,
                        separate=self.separate)
            if self.msm_init == 'largest-strong':
                args['method'] = 'lcs-spectral'
            elif self.msm_init == 'all':
                args['method'] = 'spectral'

            hmm_init = init_discrete_hmm(**args)
        elif isinstance(self.msm_init, MarkovStateModel):
            msm_count_model = self.msm_init.count_model
            # pcca = self.msm_init.pcca(n_metastable_sets=self.n_hidden_states)

            p0, P0, pobs0 = init_discrete_hmm_spectral(msm_count_model.count_matrix.toarray(),
                                                       self.n_hidden_states, reversible=self.reversible,
                                                       stationary=True, P=self.msm_init.transition_matrix,
                                                       separate=self.separate)
            hmm_init = discrete_hmm(p0, P0, pobs0)
        else:
            raise RuntimeError("msm init was neither a string (largest-strong or spectral) nor "
                               "a MarkovStateModel: {}".format(self.msm_init))

        # ---------------------------------------------------------------------------------------
        # Estimate discrete HMM
        # ---------------------------------------------------------------------------------------
        from sktime.markovprocess.bhmm.estimators.maximum_likelihood import MaximumLikelihoodHMM
        hmm_est = MaximumLikelihoodHMM(self.n_hidden_states, initial_model=hmm_init,
                                       output='discrete', reversible=self.reversible, stationary=self.stationary,
                                       accuracy=self.accuracy, maxit=self.maxit)
        hmm = hmm_est.fit(dtrajs_lagged_strided).fetch_model()
        # observation_state_symbols = np.unique(np.concatenate(dtrajs_lagged_strided))
        # update the count matrix from the counts obtained via the Viterbi paths.
        hmm_count_model = TransitionCountModel(count_matrix=hmm.transition_counts,
                                               lagtime=self.lagtime,
                                               physical_time=self.physical_time)
        # set model parameters
        self._model = HiddenMarkovStateModel(transition_matrix=hmm.transition_matrix,
                                             observation_probabilities=hmm.output_model.output_probabilities,
                                             stride=self.stride,
                                             stationary_distribution=hmm.stationary_distribution,
                                             initial_counts=hmm.initial_count,
                                             reversible=self.reversible,
                                             initial_distribution=hmm.initial_distribution, count_model=hmm_count_model,
                                             bhmm_model=hmm,
                                             observation_state_symbols=None)
        return self

    @property
    def msm_init(self):
        """ MSM initialization method, should be one of:
        * instance of :class:`MSM <sktime.markovprocess.MarkovStateModel>`

        or a string:

        * 'largest-strong' or None (default) : Estimate MSM on the largest
            strongly connected set and use spectral clustering to generate an
            initial HMM
        * 'all' : Estimate MSM(s) on the full state space to initialize the
            HMM. This fit maybe weakly connected or disconnected.
        """
        return self._msm_init

    @msm_init.setter
    def msm_init(self, value: [str, MarkovStateModel]):
        if isinstance(value, MarkovStateModel) and value.count_model is None:
            raise NotImplementedError('Requires markov state model instance that contains a count model '
                                      'with count matrix for estimation.')
        elif isinstance(value, str):
            supported = ('largest-strong', 'all')
            if value not in supported:
                raise NotImplementedError(f'unknown msm_init value, was "{value}",'
                                          f'but valid options are {supported}.')
        self._msm_init = value

    @property
    def connectivity(self):
        return self._connectivity

    @connectivity.setter
    def connectivity(self, value):
        allowed = (None, 'largest', 'populous')
        if value not in allowed:
            raise ValueError(f'Illegal value for connectivity: {value}. Allowed values are one of: {allowed}.')
        self._connectivity = value
