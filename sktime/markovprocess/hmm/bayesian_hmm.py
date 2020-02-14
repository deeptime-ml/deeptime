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
from typing import Optional, Union, List

import numpy as np
from msmtools.analysis import is_connected
from msmtools.dtraj import number_of_states

from sktime.base import Estimator
from sktime.markovprocess._base import BayesianPosterior
from sktime.markovprocess.bhmm import discrete_hmm, bayesian_hmm
from sktime.markovprocess.hmm import HiddenMarkovStateModel, MaximumLikelihoodHMSM
from sktime.markovprocess.util import compute_dtrajs_effective
from sktime.util import ensure_dtraj_list

__author__ = 'noe'

__all__ = [
    'BayesianHMMPosterior',
    'BayesianHMSM',
]

class BayesianHMMPosterior(BayesianPosterior):
    r""" Bayesian Hidden Markov model with samples of posterior and prior. """

    def __init__(self,
                 prior: Optional[HiddenMarkovStateModel] = None,
                 samples: Optional[List[HiddenMarkovStateModel]] = (),
                 hidden_state_trajs: Optional[List[np.ndarray]] = ()):
        super(BayesianHMMPosterior, self).__init__(prior=prior, samples=samples)
        self.hidden_state_trajectories_samples = hidden_state_trajs

    def submodel_largest(self, strong=True, connectivity_threshold='1/n', observe_nonempty=True, dtrajs=None):
        dtrajs = ensure_dtraj_list(dtrajs)
        states = self.prior.states_largest(directed=strong, connectivity_threshold=connectivity_threshold)
        obs = self.prior.nonempty_obs(dtrajs) if observe_nonempty else None
        return self.submodel(states=states, obs=obs)

    def submodel_populous(self, strong=True, connectivity_threshold='1/n', observe_nonempty=True, dtrajs=None):
        dtrajs = ensure_dtraj_list(dtrajs)
        states = self.prior.states_populous(strong=strong, connectivity_threshold=connectivity_threshold)
        obs = self.prior.nonempty_obs(dtrajs) if observe_nonempty else None
        return self.submodel(states=states, obs=obs)

    def submodel(self, states=None, obs=None):
        # restrict prior
        sub_model = self.prior.submodel(states=states, obs=obs)
        # restrict reduce samples
        subsamples = [sample.submodel(states=states, obs=obs)
                      for sample in self]
        return BayesianHMMPosterior(sub_model, subsamples, self.hidden_state_trajectories_samples)


class BayesianHMSM(Estimator):
    r""" Estimator for a Bayesian Hidden Markov state model """

    def __init__(self, init_hmsm: HiddenMarkovStateModel,
                 n_states: int = 2,
                 lagtime: int = 1, n_samples: int = 100,
                 stride: Union[str, int] = 'effective',
                 p0_prior: Optional[Union[str, float, np.ndarray]] = 'mixed',
                 transition_matrix_prior: Union[str, np.ndarray] = 'mixed',
                 store_hidden: bool = False,
                 reversible: bool = True,
                 stationary: bool = False):
        r"""
        Estimator for a Bayesian Hidden Markov state model.

        Parameters
        ----------
        init_hmsm : :class:`HMSM <sktime.markovprocess.hidden_markov_model.HMSM>`
            Single-point estimate of HMSM object around which errors will be evaluated.
            There is a static method available that can be used to generate a default prior.
        n_states : int, optional, default=2
            number of hidden states
        lagtime : int, optional, default=1
            lagtime to estimate the HMSM at
        n_samples : int, optional, default=100
            Number of Gibbs sampling steps
        stride : str or int, default='effective'
            stride between two lagged trajectories extracted from the input
            trajectories. Given trajectory s[t], stride and lag will result
            in trajectories
                s[0], s[tau], s[2 tau], ...
                s[stride], s[stride + tau], s[stride + 2 tau], ...
            Setting stride = 1 will result in using all data (useful for
            maximum likelihood estimator), while a Bayesian estimator requires
            a longer stride in order to have statistically uncorrelated
            trajectories. Setting stride = 'effective' uses the largest
            neglected timescale as an estimate for the correlation time and
            sets the stride accordingly.
        p0_prior : None, str, float or ndarray(n)
            Prior for the initial distribution of the HMM. Will only be active
            if stationary=False (stationary=True means that p0 is identical to
            the stationary distribution of the transition matrix).
            Currently implements different versions of the Dirichlet prior that
            is conjugate to the Dirichlet distribution of p0. p0 is sampled from:

            .. math:
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
            transition matrix priors as described in [3]_ if reversible=True. For the
            nonreversible case the posterior of transition matrix :math:`P` is:

            .. math:
                P \sim \prod_{i,j} p_{ij}^{b_{ij} + c_{ij} - 1}
            where :math:`c_{ij}` are the number of transitions found for hidden
            trajectories and :math:`b_{ij}` are prior counts.

            * 'mixed' (default),  :math:`b_{ij} = p_{ij,init}`, where :math:`p_{ij,init}`
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
            store hidden trajectories in sampled HMMs
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
        .. [1] F. Noe, H. Wu, J.-H. Prinz and N. Plattner: Projected and hidden
            Markov models for calculating kinetics and metastable states of complex
            molecules. J. Chem. Phys. 139, 184114 (2013)
        .. [2] J. D. Chodera Et Al: Bayesian hidden Markov model analysis of
            single-molecule force spectroscopy: Characterizing kinetics under
            measurement uncertainty. arXiv:1108.1430 (2011)
        .. [3] Trendelkamp-Schroer, B., H. Wu, F. Paul and F. Noe:
            Estimation and uncertainty of reversible Markov models.
            J. Chem. Phys. 143, 174101 (2015).
        """

        super(BayesianHMSM, self).__init__()
        self.n_states = n_states
        self.lagtime = lagtime
        self.stride = stride

        self.p0_prior = p0_prior
        self.transition_matrix_prior = transition_matrix_prior
        self.init_hmsm = init_hmsm
        self.store_hidden = store_hidden
        self.reversible = reversible
        self.stationary = stationary
        self.n_samples = n_samples

    def fetch_model(self) -> BayesianHMMPosterior:
        return self._model

    @property
    def init_hmsm(self):
        return self._init_hmsm

    @init_hmsm.setter
    def init_hmsm(self, value: Optional[HiddenMarkovStateModel]):
        if value is not None and not issubclass(value.__class__, HiddenMarkovStateModel):
            raise ValueError('hmsm must be of type HMSM')
        self._init_hmsm = value

    @staticmethod
    def default_prior_estimator(n_states: int, lagtime: int, stride: Union[str, int] = 'effective',
                                reversible: bool = True, stationary: bool = False,
                                separate: Optional[Union[int, List[int]]] = None, dt_traj: str = '1 step'):
        accuracy = 1e-2  # sufficient accuracy for an initial guess
        prior_estimator = MaximumLikelihoodHMSM(
            n_states=n_states, lagtime=lagtime, stride=stride,
            reversible=reversible, stationary=stationary, physical_time=dt_traj,
            separate=separate, connectivity=None,
            accuracy=accuracy, observe_nonempty=False
        )
        return prior_estimator

    @staticmethod
    def default(dtrajs, n_states: int, lagtime: int, n_samples: int = 100,
                stride: Union[str, int] = 'effective',
                p0_prior: Optional[Union[str, float, np.ndarray]] = 'mixed',
                transition_matrix_prior: Union[str, np.ndarray] = 'mixed',
                separate: Optional[Union[int, List[int]]] = None,
                store_hidden: bool = False,
                reversible: bool = True,
                stationary: bool = False,
                dt_traj: str = '1 step'):
        """
        Computes a default prior for a BHMSM and uses that for error estimation.
        For a more detailed description of the arguments please
        refer to :class:`HMSM <sktime.markovprocess.hidden_markov_model.HMSM>` or
        :class:`BayesianHMSM <sktime.markovprocess.bayesian_hmsm.BayesianHMSM>`.
        """
        dtrajs = ensure_dtraj_list(dtrajs)
        prior_est = BayesianHMSM.default_prior_estimator(n_states=n_states, lagtime=lagtime, stride=stride,
                                                         reversible=reversible, stationary=stationary,
                                                         separate=separate, dt_traj=dt_traj)
        prior = prior_est.fit(dtrajs).fetch_model().submodel_largest(connectivity_threshold='1/n', dtrajs=dtrajs)

        estimator = BayesianHMSM(init_hmsm=prior, n_states=n_states, lagtime=lagtime, n_samples=n_samples,
                                 stride=stride, p0_prior=p0_prior, transition_matrix_prior=transition_matrix_prior,
                                 store_hidden=store_hidden, reversible=reversible,
                                 stationary=stationary)
        return estimator

    def fit(self, dtrajs, callback=None):
        dtrajs = ensure_dtraj_list(dtrajs)

        model = BayesianHMMPosterior()

        # check if n_states and lag are compatible
        if self.lagtime != self.init_hmsm.lagtime:
            raise ValueError('BayesianHMSM cannot be initialized with init_hmsm with incompatible lagtime.')
        if self.n_states != self.init_hmsm.n_states:
            raise ValueError('BayesianHMSM cannot be initialized with init_hmsm with incompatible n_states.')

        # EVALUATE STRIDE
        init_stride = self.init_hmsm.stride
        if self.stride == 'effective':
            from sktime.markovprocess.util import compute_effective_stride
            self.stride = compute_effective_stride(dtrajs, self.lagtime, self.n_states)

        # if stride is different to init_hmsm, check if microstates in lagged-strided trajs are compatible
        dtrajs_lagged_strided = compute_dtrajs_effective(
            dtrajs, lagtime=self.lagtime, n_states=self.n_states, stride=self.stride
        )
        if self.stride != init_stride:
            symbols = np.unique(np.concatenate(dtrajs_lagged_strided))
            if not np.all(self.init_hmsm.observation_state_symbols == symbols):
                raise ValueError('Choice of stride has excluded a different set of microstates than in '
                                 'init_hmsm. Set of observed microstates in time-lagged strided trajectories '
                                 'must match to the one used for init_hmsm estimation.')

        # as mentioned in the docstring, take init_hmsm observed set observation probabilities
        self.observe_nonempty = False

        # update HMM Model
        model.prior = self.init_hmsm.copy()

        prior = model.prior
        prior_count_model = prior.count_model
        # check if we have a valid initial model
        if self.reversible and not is_connected(prior_count_model.count_matrix):
            raise NotImplementedError(f'Encountered disconnected count matrix:\n{self.count_matrix} '
                                      f'with reversible Bayesian HMM sampler using lag={self.lag}'
                                      f' and stride={self.stride}. Consider using shorter lag, '
                                      'or shorter stride (to use more of the data), '
                                      'or using a lower value for mincount_connectivity.')

        # here we blow up the output matrix (if needed) to the FULL state space because we want to use dtrajs in the
        # Bayesian HMM sampler. This is just an initialization.
        n_states_full = number_of_states(dtrajs)

        if prior.n_observation_states < n_states_full:
            eps = 0.01 / n_states_full  # default output probability, in order to avoid zero columns
            # full state space output matrix. make sure there are no zero columns
            B_init = eps * np.ones((self.n_states, n_states_full), dtype=np.float64)
            # fill active states
            B_init[:, prior.observation_state_symbols] = np.maximum(eps, prior.observation_probabilities)
            # renormalize B to make it row-stochastic
            B_init /= B_init.sum(axis=1)[:, None]
        else:
            B_init = prior.observation_probabilities

        # HMM sampler
        if self.init_hmsm is not None:
            hmm_mle = self.init_hmsm.bhmm_model
        else:
            hmm_mle = discrete_hmm(prior.initial_distribution, prior.transition_matrix, B_init)

        sampled_hmm = bayesian_hmm(dtrajs_lagged_strided, hmm_mle, nsample=self.n_samples,
                                   reversible=self.reversible, stationary=self.stationary,
                                   p0_prior=self.p0_prior, transition_matrix_prior=self.transition_matrix_prior,
                                   store_hidden=self.store_hidden, callback=callback).fetch_model()

        # repackage samples as HMSM objects and re-normalize after restricting to observable set
        samples = []
        for sample in sampled_hmm:  # restrict to observable set if necessary
            P = sample.transition_matrix
            pi = sample.stationary_distribution
            pobs = sample.output_model.output_probabilities
            init_dist = sample.initial_distribution

            Bobs = pobs[:, prior.observation_state_symbols]
            pobs = Bobs / Bobs.sum(axis=1)[:, None]  # renormalize
            samples.append(HiddenMarkovStateModel(P, pobs, stationary_distribution=pi,
                                                  count_model=prior_count_model, initial_counts=sample.initial_count,
                                                  reversible=self.reversible, initial_distribution=init_dist))

        # store results
        if self.store_hidden:
            model.hidden_state_trajectories_samples = [s.hidden_state_trajectories for s in sampled_hmm]
        model.samples = samples

        # set new model
        self._model = model

        return self
