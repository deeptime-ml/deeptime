import collections
from copy import deepcopy
from typing import Optional, Union, List

import numpy as np

from deeptime.markov.tools.analysis import is_connected
from deeptime.markov.tools.estimation import sample_tmatrix, transition_matrix

from deeptime.base import Estimator
from deeptime.markov._base import BayesianPosterior, MembershipsChapmanKolmogorovValidator
from deeptime.markov._transition_matrix import stationary_distribution
from deeptime.markov.hmm import HiddenMarkovModel
from ._output_model import DiscreteOutputModel
from ._util import observations_in_state, sample_hidden_state_trajectory
from deeptime.markov.msm import MarkovStateModel
from deeptime.markov import TransitionCountModel, compute_dtrajs_effective, number_of_states
from deeptime.markov.hmm._hmm_bindings import util as _bd_util
from deeptime.util.types import ensure_dtraj_list

__author__ = 'noe, clonker'

__all__ = [
    'BayesianHMMPosterior',
    'BayesianHMM',
]

from ...util.platform import handle_progress_bar


class BayesianHMMPosterior(BayesianPosterior):
    r""" Bayesian Hidden Markov model with samples of posterior and prior.

    Parameters
    ----------
    prior : HiddenMarkovModel, optional, default=None
        The prior.
    samples : list of HiddenMarkovModel, optional, default=()
        Sampled models.

    See Also
    --------
    BayesianHMM
    """

    def __init__(self,
                 prior: Optional[HiddenMarkovModel] = None,
                 samples: Optional[List[HiddenMarkovModel]] = ()):
        super(BayesianHMMPosterior, self).__init__(prior=prior, samples=samples)

    def submodel_largest(self, directed=True, connectivity_threshold='1/n', observe_nonempty=True, dtrajs=None):
        r""" Creates a submodel from the largest connected set.

        Parameters
        ----------
        directed : bool, optional, default=True
            Whether the connectivity graph on the count matrix is interpreted as directed.
        connectivity_threshold : float or '1/n', optional, default='1/n'.
            Connectivity threshold. counts that are below the specified value are disregarded when finding connected
            sets. In case of '1/n', the threshold gets resolved to :math:`1 / n\_states\_full`.
        observe_nonempty : bool, optional, default=True
            Whether to restrict to observable states which are observed in provided dtrajs. If True, dtrajs must not
            be None.
        dtrajs : array_like or list of array_like, optional, default=None
            Time series on which is evaluated whether observable states in the model were actually observed.

        Returns
        -------
        submodel : BayesianHMMPosterior
            The submodel.
        """
        dtrajs = ensure_dtraj_list(dtrajs)
        states = self.prior.states_largest(directed=directed, connectivity_threshold=connectivity_threshold)
        obs = self.prior.nonempty_obs(dtrajs) if observe_nonempty else None
        return self.submodel(states=states, obs=obs)

    def submodel_populous(self, directed=True, connectivity_threshold='1/n', observe_nonempty=True, dtrajs=None):
        r""" Creates a submodel from the most populated connected set.

        Parameters
        ----------
        directed : bool, optional, default=True
            Whether the connectivity graph on the count matrix is interpreted as directed.
        connectivity_threshold : float or '1/n', optional, default='1/n'.
            Connectivity threshold. counts that are below the specified value are disregarded when finding connected
            sets. In case of '1/n', the threshold gets resolved to :math:`1 / n\_states\_full`.
        observe_nonempty : bool, optional, default=True
            Whether to restrict to observable states which are observed in provided dtrajs. If True, dtrajs must not
            be None.
        dtrajs : array_like or list of array_like, optional, default=None
            Time series on which is evaluated whether observable states in the model were actually observed and
            which states were the most populated.

        Returns
        -------
        submodel : BayesianHMMPosterior
            The submodel.
        """
        dtrajs = ensure_dtraj_list(dtrajs)
        states = self.prior.states_populous(strong=directed, connectivity_threshold=connectivity_threshold)
        obs = self.prior.nonempty_obs(dtrajs) if observe_nonempty else None
        return self.submodel(states=states, obs=obs)

    @property
    def hidden_state_trajectories_samples(self):
        r""" Hidden state trajectories of sampled HMMs. Available if the estimator was configured to save them,
        see :attr:`BayesianHMM.store_hidden`.
        """
        return [s.hidden_state_trajectories for s in self.samples]

    @property
    def samples(self) -> Optional[List[HiddenMarkovModel]]:
        r""" The sampled models. """
        return self._samples

    @samples.setter
    def samples(self, value: Optional[List[HiddenMarkovModel]]):
        self._samples = value

    @property
    def prior(self) -> HiddenMarkovModel:
        r""" The prior model. """
        return self._prior

    @prior.setter
    def prior(self, value: HiddenMarkovModel):
        self._prior = value

    def submodel(self, states=None, obs=None):
        r""" Creates a submodel from this model restricted to a selection of observable and hidden states.

        Parameters
        ----------
        states : ndarray or None, optional, default=None
            The hidden states to restrict to. If None there is no restriction.
        obs : ndarray or None, optional, default=None
            The observable states to restrict to. If None there is no restriction. Only makes sense with
            :class:`DiscreteOutputModel`.

        Returns
        -------
        submodel : BayesianHMMPosterior
            The submodel.
        """
        # restrict prior
        sub_model = self.prior.submodel(states=states, obs=obs)
        # restrict reduce samples
        subsamples = [sample.submodel(states=states, obs=obs)
                      for sample in self]
        return BayesianHMMPosterior(sub_model, subsamples)


class BayesianHMM(Estimator):
    r""" Estimator for a Bayesian Hidden Markov state model.

    The theory and estimation procedure are described
    in :footcite:`noe2013projected`, :footcite:`chodera2011bayesian`.

    Parameters
    ----------
    initial_hmm : :class:`HMM <deeptime.markov.hmm.HiddenMarkovModel>`
       Single-point estimate of HMM object around which errors will be evaluated.
       There is a static method available that can be used to generate a default prior, see :meth:`default`.
    n_samples : int, optional, default=100
       Number of sampled models.
    stride : str or int, default='effective'
       stride between two lagged trajectories extracted from the input
       trajectories. Given trajectory s[t], stride and lag will result
       in trajectories

           :code:`s[0], s[tau], s[2 tau], ...`

           :code:`s[stride], s[stride + tau], s[stride + 2 tau], ...`

       Setting stride = 1 will result in using all data (useful for
       maximum likelihood estimator), while a Bayesian estimator requires
       a longer stride in order to have statistically uncorrelated
       trajectories. Setting stride = 'effective' uses the largest
       neglected timescale as an estimate for the correlation time and
       sets the stride accordingly.
    initial_distribution_prior : None, str, float or ndarray(n)
       Prior for the initial distribution of the HMM. Will only be active
       if stationary=False (stationary=True means that p0 is identical to
       the stationary distribution of the transition matrix).
       Currently implements different versions of the Dirichlet prior that
       is conjugate to the Dirichlet distribution of p0. p0 is sampled from:

       .. math::
           p0 \sim \prod_i (p0)_i^{a_i + n_i - 1}

       where :math:`n_i` are the number of times a hidden trajectory was in
       state :math:`i` at time step 0 and :math:`a_i` is the prior count.
       Following options are available:

       * 'mixed' (default),  :math:`a_i = p_{0,init}`, where :math:`p_{0,init}`
         is the initial distribution of initial_model.
       *  ndarray(n) or float,
          the given array will be used as A.
       *  'uniform',  :math:`a_i = 1`
       *   None,  :math:`a_i = 0`. This option ensures coincidence between
           sample mean an MLE. Will sooner or later lead to sampling problems,
           because as soon as zero trajectories are drawn from a given state,
           the sampler cannot recover and that state will never serve as a starting
           state subsequently. Only recommended in the large data regime and
           when the probability to sample zero trajectories from any state
           is negligible.
    transition_matrix_prior : str or ndarray(n, n)
       Prior for the HMM transition matrix.
       Currently implements Dirichlet priors if reversible=False and reversible
       transition matrix priors as described in :footcite:`trendelkamp2015estimation` if reversible=True. For the
       nonreversible case the posterior of transition matrix :math:`P` is:

       .. math::
           P \sim \prod_{i,j} p_{ij}^{b_{ij} + c_{ij} - 1}

       where :math:`c_{ij}` are the number of transitions found for hidden
       trajectories and :math:`b_{ij}` are prior counts.

       * 'mixed' (default),  :math:`b_{ij} = p_{ij,\mathrm{init}}`, where :math:`p_{ij,\mathrm{init}}`
         is the transition matrix of initial_model. That means one prior
         count will be used per row.
       * ndarray(n, n) or broadcastable,
         the given array will be used as B.
       * 'uniform',  :math:`b_{ij} = 1`
       * None,  :math:`b_ij = 0`. This option ensures coincidence between
         sample mean an MLE. Will sooner or later lead to sampling problems,
         because as soon as a transition :math:`ij` will not occur in a
         sample, the sampler cannot recover and that transition will never
         be sampled again. This option is not recommended unless you have
         a small HMM and a lot of data.
    store_hidden : bool, optional, default=False
       Store hidden trajectories in sampled HMMs, see
       :attr:`BayesianHMMPosterior.hidden_state_trajectories_samples`.
    reversible : bool, optional, default=True
       If True, a prior that enforces reversible transition matrices (detailed balance) is used;
       otherwise, a standard  non-reversible prior is used.
    stationary : bool, optional, default=False
       If True, the stationary distribution of the transition matrix will be used as initial distribution.
       Only use True if you are confident that the observation trajectories are started from a global
       equilibrium. If False, the initial distribution will be estimated as usual from the first step
       of the hidden trajectories.

    References
    ----------
    .. footbibliography::
    """

    _SampleStorage = collections.namedtuple(
        '_SampleStorage', ['transition_matrix', 'output_model', 'stationary_distribution', 'initial_distribution',
                           'counts', 'hidden_trajs']
    )

    def __init__(self, initial_hmm: HiddenMarkovModel,
                 n_samples: int = 100,
                 n_transition_matrix_sampling_steps: int = 1000,
                 stride: Union[str, int] = 'effective',
                 initial_distribution_prior: Optional[Union[str, float, np.ndarray]] = 'mixed',
                 transition_matrix_prior: Optional[Union[str, np.ndarray]] = 'mixed',
                 store_hidden: bool = False,
                 reversible: bool = True,
                 stationary: bool = False):
        super().__init__()
        self.initial_hmm = initial_hmm
        self.n_samples = n_samples
        self.n_transition_matrix_sampling_steps = n_transition_matrix_sampling_steps
        self.stride = stride
        self.initial_distribution_prior = initial_distribution_prior
        self.transition_matrix_prior = transition_matrix_prior
        self.store_hidden = store_hidden
        self.reversible = reversible
        self.stationary = stationary

    @staticmethod
    def default(dtrajs, n_hidden_states: int, lagtime: int, n_samples: int = 100,
                stride: Union[str, int] = 'effective',
                initial_distribution_prior: Optional[Union[str, float, np.ndarray]] = 'mixed',
                transition_matrix_prior: Optional[Union[str, np.ndarray]] = 'mixed',
                separate: Optional[Union[int, List[int]]] = None,
                store_hidden: bool = False,
                reversible: bool = True,
                stationary: bool = False,
                prior_submodel: bool = True):
        """ Computes a default prior for a BHMM and uses that for error estimation.
        For a more detailed description of the arguments please
        refer to :class:`HMM <deeptime.markov.hmm.HiddenMarkovModel>` or
        :meth:`__init__`.

        Returns
        -------
        estimator : BayesianHMM
            Estimator that is initialized with a default prior model.
        """
        from deeptime.markov.hmm import init, MaximumLikelihoodHMM
        dtrajs = ensure_dtraj_list(dtrajs)
        init_hmm = init.discrete.metastable_from_data(dtrajs, n_hidden_states=n_hidden_states, lagtime=lagtime,
                                                      stride=stride, reversible=reversible, stationary=stationary,
                                                      separate_symbols=separate)
        hmm = MaximumLikelihoodHMM(init_hmm, stride=stride, lagtime=lagtime, reversible=reversible,
                                   stationary=stationary, accuracy=1e-2).fit(dtrajs).fetch_model()
        if prior_submodel:
            hmm = hmm.submodel_largest(connectivity_threshold=0, observe_nonempty=False, dtrajs=dtrajs)
        estimator = BayesianHMM(hmm, n_samples=n_samples, stride=stride,
                                initial_distribution_prior=initial_distribution_prior,
                                transition_matrix_prior=transition_matrix_prior, store_hidden=store_hidden,
                                reversible=reversible, stationary=stationary)
        return estimator

    @property
    def stationary(self):
        r""" If True, the stationary distribution of the transition matrix will be used as initial distribution.
        Only use True if you are confident that the observation trajectories are started from a global
        equilibrium. If False, the initial distribution will be estimated as usual from the first step
        of the hidden trajectories.
        """
        return self._stationary

    @stationary.setter
    def stationary(self, value):
        self._stationary = value

    @property
    def reversible(self):
        r""" If True, a prior that enforces reversible transition matrices (detailed balance) is used;
        otherwise, a standard  non-reversible prior is used.
        """
        return self._reversible

    @reversible.setter
    def reversible(self, value):
        self._reversible = value

    @property
    def store_hidden(self):
        r"""  Store hidden trajectories in sampled HMMs, see
        :attr:`BayesianHMMPosterior.hidden_state_trajectories_samples`.
        """
        return self._store_hidden

    @store_hidden.setter
    def store_hidden(self, value):
        self._store_hidden = value

    @property
    def transition_matrix_prior(self):
        r""" Prior for the transition matrix. For a more detailed description refer to :meth:`__init__`. """
        return self._transition_matrix_prior

    @transition_matrix_prior.setter
    def transition_matrix_prior(self, value):
        self._transition_matrix_prior = value

    @property
    def initial_distribution_prior(self) -> Optional[Union[str, float, np.ndarray]]:
        r""" Prior for the initial distribution. For a more detailed description refer to :meth:`__init__`. """
        return self._initial_distribution_prior

    @initial_distribution_prior.setter
    def initial_distribution_prior(self, value: Optional[Union[str, float, np.ndarray]]):
        self._initial_distribution_prior = value

    @property
    def initial_hmm(self) -> HiddenMarkovModel:
        r""" The prior HMM. An estimator with a default prior HMM can be generated using the static :meth:`default`
        method.
        """
        return self._initial_hmm

    @initial_hmm.setter
    def initial_hmm(self, value: HiddenMarkovModel):
        if not isinstance(value, HiddenMarkovModel):
            raise ValueError(f"Initial hmm must be of type HiddenMarkovModel, but was {value.__class__.__name__}.")
        self._initial_hmm = value

    @property
    def n_samples(self) -> int:
        r""" Number of sampled models. """
        return self._n_samples

    @n_samples.setter
    def n_samples(self, value: int):
        self._n_samples = value

    def fetch_model(self) -> BayesianHMMPosterior:
        r""" Yields the current model or None if :meth:`fit` was not yet called.

        Returns
        -------
        posterior : BayesianHMMPosterior
            The model.
        """
        return self._model

    @property
    def _initial_distribution_prior_np(self) -> np.ndarray:
        r"""
        Internal method that evaluates the prior to its ndarray realization.
        """
        if self.initial_distribution_prior is None or self.initial_distribution_prior == 'sparse':
            prior = np.zeros(self.initial_hmm.n_hidden_states)
        elif isinstance(self.initial_distribution_prior, np.ndarray):
            if self.initial_distribution_prior.ndim == 1 \
                    and len(self.initial_distribution_prior) == self.initial_hmm.n_hidden_states:
                prior = np.asarray(self.initial_distribution_prior)
            else:
                raise ValueError(f"If the initial distribution prior is given as a np array, it must be 1-dimensional "
                                 f"and be as long as there are hidden states. "
                                 f"(ndim={self.initial_distribution_prior.ndim}, "
                                 f"len={len(self.initial_distribution_prior)})")
        elif self.initial_distribution_prior == 'mixed':
            prior = self.initial_hmm.initial_distribution
        elif self.initial_distribution_prior == 'uniform':
            prior = np.ones(self.initial_hmm.n_hidden_states)
        else:
            raise ValueError(f'Initial distribution prior mode undefined: {self.initial_distribution_prior}')
        return prior

    @property
    def _transition_matrix_prior_np(self) -> np.ndarray:
        r"""
        Internal method that evaluates the prior to its ndarray realization.
        """
        n_states = self.initial_hmm.n_hidden_states
        if self.transition_matrix_prior is None or self.transition_matrix_prior == 'sparse':
            prior = np.zeros((n_states, n_states))
        elif isinstance(self.transition_matrix_prior, np.ndarray):
            if np.array_equal(self.transition_matrix_prior.shape, (n_states, n_states)):
                prior = np.asarray(self.transition_matrix_prior)
            else:
                raise ValueError(f"If the initial distribution prior is given as a np array, it must be 2-dimensional "
                                 f"and a (n_hidden_states x n_hidden_states) matrix. "
                                 f"(ndim={self.transition_matrix_prior.ndim}, "
                                 f"shape={self.transition_matrix_prior.shape})")
        elif self.transition_matrix_prior == 'mixed':
            prior = np.copy(self.initial_hmm.transition_model.transition_matrix)
        elif self.transition_matrix_prior == 'uniform':
            prior = np.ones((n_states, n_states))
        else:
            raise ValueError(f'Initial distribution prior mode undefined: {self.transition_matrix_prior}')
        return prior

    def _update(self, model: _SampleStorage, observations, temp_alpha,
                transition_matrix_prior, initial_distribution_prior):
        """Update the current model using one round of Gibbs sampling."""
        self._update_hidden_state_trajectories(model, observations, temp_alpha)
        self._update_emission_probabilities(model, observations)
        self._update_transition_matrix(model, transition_matrix_prior, initial_distribution_prior,
                                       reversible=self.reversible, stationary=self.stationary,
                                       n_sampling_steps=self.n_transition_matrix_sampling_steps)

    @staticmethod
    def _update_hidden_state_trajectories(model: _SampleStorage, observations, temp_alpha):
        """Sample a new set of state trajectories from the conditional distribution P(S | T, E, O)"""
        model.hidden_trajs.clear()
        for obs in observations:
            s_t = sample_hidden_state_trajectory(model.transition_matrix, model.output_model,
                                                 model.initial_distribution, obs, temp_alpha)
            model.hidden_trajs.append(s_t)

    @staticmethod
    def _update_emission_probabilities(model: _SampleStorage, observations):
        """Sample a new set of emission probabilites from the conditional distribution P(E | S, O) """
        observations_by_state = [observations_in_state(model.hidden_trajs, observations, state)
                                 for state in range(model.transition_matrix.shape[0])]
        model.output_model.sample(observations_by_state)

    @staticmethod
    def _update_transition_matrix(model: _SampleStorage, transition_matrix_prior,
                                  initial_distribution_prior, reversible: bool = True, stationary: bool = False,
                                  n_sampling_steps: int = 1000):
        """ Updates the hidden-state transition matrix and the initial distribution """
        C = _bd_util.count_matrix(model.hidden_trajs, 1, model.transition_matrix.shape[0])
        model.counts[...] = C
        C = C + transition_matrix_prior

        # check if we work with these options
        if reversible and not is_connected(C, directed=True):
            raise NotImplementedError('Encountered disconnected count matrix with sampling option reversible:\n '
                                      f'{C}\nUse prior to ensure connectivity or use reversible=False.')
        # ensure consistent sparsity pattern (P0 might have additional zeros because of underflows)
        # TODO: these steps work around a bug in msmtools. Should be fixed there
        P0 = transition_matrix(C, reversible=reversible, maxiter=10000, warn_not_converged=False)
        zeros = np.where(P0 + P0.T == 0)
        C[zeros] = 0
        # run sampler
        Tij = sample_tmatrix(C, nsample=1, nsteps=n_sampling_steps, reversible=reversible)

        # INITIAL DISTRIBUTION
        if stationary:  # p0 is consistent with P
            p0 = stationary_distribution(Tij, C=C)
        else:
            n0 = BayesianHMM._count_init(model.hidden_trajs, model.transition_matrix.shape[0])
            first_timestep_counts_with_prior = n0 + initial_distribution_prior
            positive = first_timestep_counts_with_prior > 0
            p0 = np.zeros(n0.shape, dtype=model.transition_matrix.dtype)
            p0[positive] = np.random.dirichlet(first_timestep_counts_with_prior[positive])  # sample p0 from posterior

        # update HMM with new sample
        model.transition_matrix[...] = Tij
        model.stationary_distribution[...] = p0

    @staticmethod
    def _count_init(hidden_state_trajectories, n_hidden_states):
        """Compute the counts at the first time step

        Returns
        -------
        n : ndarray(nstates)
            n[i] is the number of trajectories starting in state i

        """
        n = [traj[0] for traj in hidden_state_trajectories]
        return np.bincount(n, minlength=n_hidden_states)

    def fit(self, data, n_burn_in: int = 0, n_thin: int = 1, progress=None, **kwargs):
        r""" Sample from the posterior.

        Parameters
        ----------
        data : array_like or list of array_like
            Input time series data.
        n_burn_in : int, optional, default=0
            The number of samples to discard to burn-in, following which :attr:`n_samples` samples will be generated.
        n_thin : int, optional, default=1
            The number of Gibbs sampling updates used to generate each returned sample.
        progress : iterable, optional, default=None
            Optional progressbar. Tested for tqdm.
        **kwargs
            Ignored kwargs for scikit-learn compatibility.

        Returns
        -------
        self : BayesianHMM
            Reference to self.
        """
        progress = handle_progress_bar(progress)
        dtrajs = ensure_dtraj_list(data)

        # fetch priors
        tmat = self.initial_hmm.transition_model.transition_matrix
        transition_matrix_prior = self._transition_matrix_prior_np

        initial_distribution_prior = self._initial_distribution_prior_np

        model = BayesianHMMPosterior()
        # update HMM Model
        model.prior = self.initial_hmm.copy()

        prior = model.prior

        # check if we are strongly connected in the reversible case (plus prior)
        if self.reversible and not is_connected(tmat + transition_matrix_prior, directed=True):
            raise NotImplementedError('Trying to sample disconnected HMM with option reversible:\n '
                                      f'{tmat}\n Use prior to connect, select connected subset, '
                                      f'or use reversible=False.')

        # EVALUATE STRIDE
        dtrajs_lagged_strided = compute_dtrajs_effective(
            dtrajs, lagtime=prior.lagtime, n_states=prior.n_hidden_states, stride=self.stride
        )
        # if stride is different to init_hmm, check if microstates in lagged-strided trajs are compatible
        if self.stride != self.initial_hmm.stride:
            symbols = np.unique(np.concatenate(dtrajs_lagged_strided))
            if not len(np.intersect1d(self.initial_hmm.observation_symbols, symbols)) == len(symbols):
                raise ValueError('Choice of stride has excluded a different set of microstates than in '
                                 'init_hmm. Set of observed microstates in time-lagged strided trajectories '
                                 'must match to the one used for init_hmm estimation.')

        # here we blow up the output matrix (if needed) to the FULL state space because we want to use dtrajs in the
        # Bayesian HMM sampler. This is just an initialization.
        n_states_full = number_of_states(dtrajs_lagged_strided)

        if prior.n_observation_states < n_states_full:
            eps = 0.01 / n_states_full  # default output probability, in order to avoid zero columns
            # full state space output matrix. make sure there are no zero columns
            full_obs_probabilities = eps * np.ones((prior.n_hidden_states, n_states_full), dtype=np.float64)
            # fill active states
            full_obs_probabilities[:, prior.observation_symbols] = np.maximum(eps, prior.output_probabilities)
            # renormalize B to make it row-stochastic
            full_obs_probabilities /= full_obs_probabilities.sum(axis=1)[:, None]
        else:
            full_obs_probabilities = prior.output_probabilities

        maxT = max(len(o) for o in dtrajs_lagged_strided)

        # pre-construct hidden variables
        temp_alpha = np.zeros((maxT, prior.n_hidden_states))

        has_all_obs_symbols = model.prior.n_observation_states == len(model.prior.observation_symbols_full)

        try:
            # sample model is basically copy of prior
            sample_model = BayesianHMM._SampleStorage(
                transition_matrix=prior.transition_model.transition_matrix.copy(),
                output_model=DiscreteOutputModel(full_obs_probabilities.copy()),
                initial_distribution=prior.initial_distribution.copy(),
                stationary_distribution=prior.transition_model.stationary_distribution.copy(),
                counts=prior.count_model.count_matrix.copy(),
                hidden_trajs=[]
            )

            # Run burn-in.
            for _ in range(n_burn_in):
                self._update(sample_model, dtrajs_lagged_strided, temp_alpha, transition_matrix_prior,
                             initial_distribution_prior)

            # Collect data.
            models = []
            for _ in progress(range(self.n_samples), desc="Drawing samples", leave=False):
                # Run a number of Gibbs sampling updates to generate each sample.
                for _ in range(n_thin):
                    self._update(sample_model, dtrajs_lagged_strided, temp_alpha, transition_matrix_prior,
                                 initial_distribution_prior)
                    sample_model.output_model.normalize()
                self._append_sample(models, prior, sample_model)

            if not has_all_obs_symbols:
                models = [m.submodel(states=None, obs=model.prior.observation_symbols) for m in models]

            model.samples = models
        finally:
            del temp_alpha

        # set new model
        self._model = model

        return self

    def _append_sample(self, models, prior, sample_model):
        # Save a copy of the current model.
        model_copy = deepcopy(sample_model)
        # the Viterbi path is discarded, but is needed to get a new transition matrix for each model.
        if not self.store_hidden:
            model_copy.hidden_trajs.clear()
        # potentially restrict sampled models to observed space
        # since model_copy is defined on full space, observation_symbols are also observation states
        count_model = TransitionCountModel(model_copy.counts, lagtime=prior.lagtime)
        models.append(HiddenMarkovModel(
            transition_model=MarkovStateModel(model_copy.transition_matrix,
                                              stationary_distribution=model_copy.stationary_distribution,
                                              reversible=self.reversible, count_model=count_model),
            output_model=model_copy.output_model, initial_distribution=model_copy.initial_distribution,
            hidden_state_trajectories=model_copy.hidden_trajs))

    def chapman_kolmogorov_validator(self, mlags, test_model: BayesianHMMPosterior = None):
        r"""Returns a Chapman-Kolmogorov validator based on this estimator and a test model.

        Parameters
        ----------
        mlags : int or int-array
            Multiple of lagtimes of the test_model to test against.
        test_model : BayesianHMMPosterior, optional, default=None
            The model that is tested. If not provided, uses this estimator's encapsulated model.

        Returns
        -------
        validator : markov.MembershipsChapmanKolmogorovValidator
            The validator.
        """
        test_model = self.fetch_model() if test_model is None else test_model
        assert test_model is not None, "We need a test model via argument or an estimator which was already" \
                                       "fit to data."

        from . import DiscreteOutputModel
        assert isinstance(test_model.prior.output_model, DiscreteOutputModel), \
            "Can only perform CKTest for discrete output models"
        memberships = np.eye(test_model.prior.n_hidden_states)
        lagtime = test_model.prior.lagtime
        return BayesianHMMChapmanKolmogorovValidator(test_model, self, memberships, lagtime, mlags)


def _ck_estimate_model_for_lag(estimator: BayesianHMM, model, data, lagtime):
    estimator = BayesianHMM.default(dtrajs=data, n_hidden_states=estimator.initial_hmm.n_hidden_states,
                                    lagtime=lagtime, n_samples=estimator.n_samples, stride=estimator.stride,
                                    initial_distribution_prior=estimator.initial_distribution_prior,
                                    transition_matrix_prior=estimator.transition_matrix_prior,
                                    reversible=estimator.reversible, stationary=estimator.stationary)
    return estimator.fit(data).fetch_model()


class BayesianHMMChapmanKolmogorovValidator(MembershipsChapmanKolmogorovValidator):

    def fit(self, data, n_jobs=1, progress=None, **kw):
        if n_jobs != 1:
            import warnings
            warnings.warn("ignoring n_jobs for HMM CKtest")

        return super().fit(data, n_jobs=1, estimate_model_for_lag=_ck_estimate_model_for_lag, progress=progress, **kw)
