import collections
from typing import List, Union, Optional

import numpy as np
from deeptime.markov._base import MembershipsChapmanKolmogorovValidator
from scipy.sparse import issparse

from ...base import Estimator
from .._transition_matrix import estimate_P, stationary_distribution
from ._hidden_markov_model import HiddenMarkovModel, viterbi
from ..msm import MarkovStateModel
from .. import TransitionCountModel, compute_dtrajs_effective
from ._hmm_bindings import util as _util
from ...util.types import ensure_timeseries_data


class MaximumLikelihoodHMM(Estimator):
    """ Maximum likelihood Hidden Markov model (HMM) estimator.

    This class is used to fit a maximum-likelihood HMM to data. It uses an initial guess HMM with can be obtained with
    one of the provided heuristics and then uses the Baum-Welch algorithm :footcite:`baum1967inequality` to fit
    the initial guess to provided data.

    The type of output model (gaussian or discrete)
    and the number of hidden states are extracted from the initial model. In case no initial distribution was given,
    the initial model assumes a uniform initial distribution.

    Parameters
    ----------
    initial_model : HiddenMarkovModel
        This model will be used to initialize the hidden markov model estimation routine. Since it is prone to
        get stuck in local optima, several initializations should be tried and scored and/or one of the available
        initialization heuristics should be applied, if appropriate.
    stride : int or str, optional, default=1
        stride between two lagged trajectories extracted from the input trajectories. Given trajectory s[t], stride
        and lag will result in trajectories

            :code:`s[0], s[lag], s[2 lag], ...`

            :code:`s[stride], s[stride + lag], s[stride + 2 lag], ...`

        Setting stride = 1 will result in using all data (useful for maximum likelihood estimator), while a
        Bayesian estimator requires a longer stride in order to have statistically uncorrelated trajectories.
        Setting stride = 'effective' uses the largest neglected timescale as an fit for the correlation time and
        sets the stride accordingly.
    lagtime : int, optional, default=1
        Lag parameter used for fitting the HMM
    reversible : bool, optional, default=True
        If True, a prior that enforces reversible transition matrices (detailed balance) is used;
        otherwise, a standard  non-reversible prior is used.
    stationary : bool, optional, default=False
        If True, the initial distribution of hidden states is self-consistently computed as the stationary
        distribution of the transition matrix. If False, it will be estimated from the starting states.
        Only set this to true if you're sure that the observation trajectories are initiated from a global
        equilibrium distribution.
    p : (n,) ndarray, optional, default=None
        Initial or fixed stationary distribution. If given and stationary=True, transition matrices will be
        estimated with the constraint that they have the set parameter as their stationary distribution.
        If given and stationary=False, the parameter is the fixed initial distribution of hidden states.
    accuracy : float, optional, default=1e-3
        Convergence threshold for EM iteration. When two the likelihood does not increase by more than
        accuracy, the iteration is stopped successfully.
    maxit : int, optional, default=1000
        Stopping criterion for EM iteration. When this many iterations are performed without reaching the requested
        accuracy, the iteration is stopped without convergence and a warning is given.
    maxit_reversible : int, optional, default=1000000
        Maximum number of iterations for reversible transition matrix estimation. Only used with reversible=True.

    References
    ----------
    .. footbibliography::
    """

    _HMMModelStorage = collections.namedtuple('_HMMModelStorage', ['transition_matrix', 'output_model',
                                                                   'initial_distribution'])

    def __init__(self, initial_model: HiddenMarkovModel, stride: Union[int, str] = 1,
                 lagtime: int = 1, reversible: bool = True, stationary: bool = False,
                 p: Optional[np.ndarray] = None, accuracy: float = 1e-3,
                 maxit: int = 1000, maxit_reversible: int = 100000):
        super().__init__()
        self.initial_model = initial_model
        self.stride = stride
        self.lagtime = lagtime
        self.reversible = reversible
        self.stationary = stationary
        if stationary:
            self.fixed_stationary_distribution = p
            self.fixed_initial_distribution = None
        else:
            self.fixed_stationary_distribution = None
            self.fixed_initial_distribution = p
        self.accuracy = accuracy
        self.maxit = maxit
        self.maxit_reversible = maxit_reversible

    def fetch_model(self) -> HiddenMarkovModel:
        r""" Yields the current HiddenMarkovModel or None if :meth:`fit` was not called yet.

        Returns
        -------
        model : HiddenMarkovModel or None
            The model.
        """
        return self._model

    @property
    def maxit_reversible(self) -> int:
        r"""Maximum number of iterations for reversible transition matrix estimation. Only used with reversible=True."""
        return self._maxit_reversible

    @maxit_reversible.setter
    def maxit_reversible(self, value: int):
        self._maxit_reversible = int(value)

    @property
    def fixed_stationary_distribution(self) -> Optional[np.ndarray]:
        r"""Fix the stationary distribution to the provided value. Only used when :attr:`stationary` is True, otherwise
        refer to :attr:`fixed_initial_distribution`.
        """
        return self._fixed_stationary_distribution

    @fixed_stationary_distribution.setter
    def fixed_stationary_distribution(self, value: Optional[np.ndarray]):
        if value is not None and value.shape[0] != self.n_hidden_states:
            raise ValueError("Fixed stationary distribution must be as long as there are hidden states.")
        self._fixed_stationary_distribution = value

    @property
    def fixed_initial_distribution(self) -> Optional[np.ndarray]:
        r"""Fix the initial distribution to the provided value. Only used when :attr:`stationary` is False, otherwise
        refer to :attr:`fixed_stationary_distribution`.
        """
        return self._fixed_initial_distribution

    @fixed_initial_distribution.setter
    def fixed_initial_distribution(self, value: Optional[np.ndarray]):
        if value is not None and value.shape[0] != self.n_hidden_states:
            raise ValueError("Fixed initial distribution must be as long as there are hidden states.")
        self._fixed_initial_distribution = value

    @property
    def stationary(self) -> bool:
        r""" If True, the initial distribution of hidden states is self-consistently computed as the stationary
        distribution of the transition matrix. If False, it will be estimated from the starting states.
        Only set this to true if you're sure that the observation trajectories are initiated from a global
        equilibrium distribution.
        """
        return self._stationary

    @stationary.setter
    def stationary(self, value: bool):
        self._stationary = bool(value)

    @property
    def accuracy(self) -> float:
        r""" Convergence threshold for EM iteration. """
        return self._accuracy

    @accuracy.setter
    def accuracy(self, value: float):
        self._accuracy = float(value)

    @property
    def maxit(self) -> int:
        r""" Stopping criterion for EM iteration. """
        return self._maxit

    @maxit.setter
    def maxit(self, value: int):
        self._maxit = int(value)

    @property
    def reversible(self) -> bool:
        r""" Whether the hidden transition model should be estimated so that it is reversible. """
        return self._reversible

    @reversible.setter
    def reversible(self, value: bool):
        self._reversible = bool(value)

    @property
    def stride(self) -> Union[int, str]:
        r""" Stride to be applied to the input data. Must be compatible with how the initial model was estimated. """
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
        r""" The lag time at which transitions are counted. """
        return self._lagtime

    @lagtime.setter
    def lagtime(self, value: int):
        value = int(value)
        if value <= 0:
            raise ValueError("Lagtime must be positive!")
        self._lagtime = value

    @property
    def n_hidden_states(self) -> int:
        r""" The number of hidden states, coincides with the number of hidden states in the initial model."""
        return self.initial_model.n_hidden_states

    @property
    def initial_model(self) -> HiddenMarkovModel:
        r""" The initial transition model. """
        return self._initial_transition_model

    @initial_model.setter
    def initial_model(self, value: HiddenMarkovModel) -> None:
        self._initial_transition_model = value

    def fit(self, dtrajs, initial_model=None, **kwargs):
        r""" Fits a new :class:`HMM <HiddenMarkovModel>` to data.

        Parameters
        ----------
        dtrajs : array_like or list of array_like
            Timeseries data.
        initial_model : HiddenMarkovModel, optional, default=None
            Override for :attr:`initial_model`.
        **kwargs
            Ignored kwargs for scikit-learn compatibility.

        Returns
        -------
        self : MaximumLikelihoodHMM
            Reference to self.
        """
        if initial_model is None:
            initial_model = self.initial_model
        if initial_model is None or not isinstance(initial_model, HiddenMarkovModel):
            raise ValueError("For estimation, an initial model of type "
                             "`deeptime.markov.hmm.HiddenMarkovModel` is required.")

        # copy initial model
        transition_matrix = initial_model.transition_model.transition_matrix
        if issparse(transition_matrix):
            # want dense matrix, toarray makes a copy
            transition_matrix = transition_matrix.toarray()
        else:
            # new instance
            transition_matrix = np.copy(transition_matrix)

        hmm_data = MaximumLikelihoodHMM._HMMModelStorage(transition_matrix=transition_matrix,
                                                         output_model=initial_model.output_model.copy(),
                                                         initial_distribution=initial_model.initial_distribution.copy())

        dtrajs = ensure_timeseries_data(dtrajs)
        dtrajs = compute_dtrajs_effective(dtrajs, lagtime=self.lagtime, n_states=initial_model.n_hidden_states,
                                          stride=self.stride)

        max_n_frames = max(len(obs) for obs in dtrajs)
        # pre-construct hidden variables
        N = initial_model.n_hidden_states
        alpha = np.zeros((max_n_frames, N))
        beta = np.zeros((max_n_frames, N))
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
                loglik_update, _ = self._forward_backward(hmm_data, obs, alpha, beta, gamma, counts)
                loglik += loglik_update
            assert np.isfinite(loglik), it

            # convergence check
            if it > 0:
                dL = loglik - likelihoods[it - 1]
                if dL < self.accuracy:
                    converged = True

            # update model
            self._update_model(hmm_data, dtrajs, gammas, count_matrices, maxiter=self.maxit_reversible)

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

        count_model = TransitionCountModel(count_matrix=transition_counts, lagtime=self.lagtime)
        transition_model = MarkovStateModel(hmm_data.transition_matrix, reversible=self.reversible,
                                            count_model=count_model)
        hidden_state_trajs = [
            viterbi(hmm_data.transition_matrix, hmm_data.output_model.to_state_probability_trajectory(obs),
                    hmm_data.initial_distribution) for obs in dtrajs
        ]
        model = HiddenMarkovModel(
            transition_model=transition_model,
            output_model=hmm_data.output_model,
            initial_distribution=hmm_data.initial_distribution,
            likelihoods=likelihoods,
            state_probabilities=gammas,
            initial_count=self._init_counts(gammas),
            hidden_state_trajectories=hidden_state_trajs,
            stride=self.stride
        )
        self._model = model
        return self

    @staticmethod
    def _forward_backward(model: _HMMModelStorage, obs, alpha, beta, gamma, counts):
        """ Estimation step: Runs the forward-back algorithm on trajectory obs

        Parameters
        ----------
        model: _HMMModelStorage
            named tuple with transition matrix, initial distribution, output model
        obs: np.ndarray
            single observation corresponding to index itraj
        alpha: ndarray
            forward coefficients
        beta: ndarray
            backward coefficients
        gamma: ndarray
            gammas
        counts: ndarray
            count matrix

        Returns
        -------
        logprob : float
            The probability to observe the observation sequence given the HMM
            parameters
        pobs : ndarray
            state probability trajectory obtained from obs
        """
        # get parameters
        A = model.transition_matrix
        pi = model.initial_distribution
        T = len(obs)
        # compute output probability matrix
        pobs = model.output_model.to_state_probability_trajectory(obs)
        pobs = pobs.astype(A.dtype)
        # run forward - backward pass
        logprob = _util.forward_backward(A, pobs, pi, alpha, beta, gamma, counts, T)
        return logprob, pobs

    def _init_counts(self, gammas):
        gamma0_sum = np.zeros(self.n_hidden_states)
        # update state counts
        for g in gammas:
            gamma0_sum += g[0]
        return gamma0_sum

    @staticmethod
    def _reduce_transition_counts(count_matrices):
        return np.add.reduce(count_matrices)

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
        C = self._reduce_transition_counts(count_matrices)

        # compute new transition matrix
        T = estimate_P(C, reversible=self.reversible, fixed_statdist=self.fixed_stationary_distribution,
                       maxiter=maxiter, maxerr=1e-12, mincount_connectivity=1e-16)
        # estimate stationary or init distribution
        if self.stationary:
            if self.fixed_stationary_distribution is None:
                pi = stationary_distribution(T, C=C, mincount_connectivity=1e-16)
            else:
                pi = self.fixed_stationary_distribution
        else:
            if self.fixed_initial_distribution is None:
                gamma0_sum = self._init_counts(gammas)
                pi = gamma0_sum / np.sum(gamma0_sum)
            else:
                pi = self.fixed_initial_distribution

        model.initial_distribution[:] = pi
        model.transition_matrix[:] = T
        model.output_model.fit(observations, gammas)

    def chapman_kolmogorov_validator(self, mlags, test_model: HiddenMarkovModel = None):
        r""" Creates a validator instance which can be used to perform a Chapman-Kolmogorov test.

        Parameters
        ----------
        mlags : int or int-array
            Multiples of lag times for testing the Model, e.g. range(10).
            A single int will trigger a range, i.e. mlags=10 maps to
            mlags=range(1, 10). The setting None will choose mlags automatically
            according to the longest available trajectory.
        test_model : HiddenMarkovModel, optional, default=None
            The model that is tested. If not provided, uses this estimator's encapsulated model.

        Returns
        -------
        validator : MembershipsChapmanKolmogorovValidator
            The validator.

        Raises
        ------
        AssertionError
            If test_model is None and this estimator has not been :meth:`fit` on data yet or the output model
            was not a discrete output model.
        """
        test_model = self.fetch_model() if test_model is None else test_model
        assert test_model is not None, "We need a test model via argument or an estimator which was already" \
                                       "fit to data."
        from . import DiscreteOutputModel
        assert isinstance(test_model.output_model, DiscreteOutputModel), \
            "Can only perform CKTest for discrete output models"
        lagtime = test_model.lagtime
        return MLHMMChapmanKolmogorovValidator(test_model, self, np.eye(test_model.n_hidden_states), lagtime, mlags)


def _ck_estimate_model_for_lag(estimator: MaximumLikelihoodHMM, model: HiddenMarkovModel, data, lagtime):
    from .init.discrete import metastable_from_data
    initial_model = metastable_from_data(data, n_hidden_states=model.n_hidden_states, lagtime=lagtime,
                                         stride=estimator.stride, reversible=estimator.reversible,
                                         stationary=estimator.stationary)
    estimator = MaximumLikelihoodHMM(initial_model, lagtime=lagtime, reversible=estimator.reversible,
                                     stationary=estimator.stationary, accuracy=estimator.accuracy,
                                     maxit=estimator.maxit, maxit_reversible=estimator.maxit_reversible)
    hmm = estimator.fit(data).fetch_model()
    return hmm.submodel_largest(dtrajs=data)


class MLHMMChapmanKolmogorovValidator(MembershipsChapmanKolmogorovValidator):

    def fit(self, data, n_jobs=1, progress=None, estimate_model_for_lag=None, **kw):
        if n_jobs != 1:
            import warnings
            warnings.warn("Ignoring n_jobs for hmm cktest")
        return super().fit(data, n_jobs=1, estimate_model_for_lag=_ck_estimate_model_for_lag, progress=progress, **kw)
