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

import typing

import numpy as np
from msmtools.analysis import is_connected
from msmtools.dtraj import number_of_states

from sktime.markovprocess.bhmm import discrete_hmm, bayesian_hmm
from sktime.markovprocess.hidden_markov_model import HMSM, HMMTransitionCountModel
from sktime.markovprocess.maximum_likelihood_hmsm import MaximumLikelihoodHMSM
from sktime.util import ensure_dtraj_list
from ._base import BayesianPosterior
from .maximum_likelihood_hmsm import MaximumLikelihoodHMSM as _MaximumLikelihoodHMSM

__author__ = 'noe'

__all__ = [
    'BayesianHMMPosterior',
    'BayesianHMSM',
]


class BayesianHMMPosterior(BayesianPosterior):
    r""" Bayesian Hidden Markov model with samples of posterior and prior. """

    def __init__(self,
                 prior: typing.Optional[HMSM] = None,
                 samples: typing.Optional[typing.List[HMSM]] = (),
                 hidden_state_trajs: typing.Optional[typing.List[np.ndarray]] = ()):
        super(BayesianHMMPosterior, self).__init__(prior=prior, samples=samples)
        self.hidden_state_trajectories_samples = hidden_state_trajs

    def submodel_largest(self, strong=True, mincount_connectivity='1/n', observe_nonempty=True, dtrajs=None):
        # todo (or solve differently?)
        pass

    def submodel(self, states=None, obs=None, mincount_connectivity='1/n'):
        # restrict prior
        sub_model = self.prior.submodel(states=states, obs=obs,
                                        mincount_connectivity=mincount_connectivity)
        # restrict reduce samples
        count_model = sub_model.count_model
        subsamples = [sample.submodel(states=count_model.active_set, obs=count_model.observable_set)
                      for sample in self]

        # TODO: how to handle hiddenstate traj samples?!
        return BayesianHMMPosterior(sub_model, subsamples)


class BayesianHMSM(_MaximumLikelihoodHMSM):
    r"""Estimator for a Bayesian Hidden Markov state model

    Parameters
    ----------
    nstates : int, optional, default=2
        number of hidden states
    lagtime : int, optional, default=1
        lagtime to estimate the HMSM at
    stride : str or int, default=1
        stride between two lagged trajectories extracted from the input
        trajectories. Given trajectory s[t], stride and lag will result
        in trajectories
            s[0], s[tau], s[2 tau], ...
            s[stride], s[stride + tau], s[stride + 2 tau], ...
        Setting stride = 1 will result in using all data (useful for
        maximum likelihood estimator), while a Bayesian estimator requires
        a longer stride in order to have statistically uncorrelated
        trajectories. Setting stride = None 'effective' uses the largest
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
    init_hmsm : :class:`HMSM <pyemma.msm.models.HMSM>`, default=None
        Single-point estimate of HMSM object around which errors will be evaluated.
        If None is give an initial estimate will be automatically generated using the
        given parameters.
    store_hidden : bool, optional, default=False
        store hidden trajectories in sampled HMMs

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

    def __init__(self, nstates=2, lagtime=1, stride='effective',
                 p0_prior='mixed', transition_matrix_prior='mixed',
                 nsamples=100, init_hmsm=None, reversible=True, stationary=False,
                 connectivity='largest', mincount_connectivity='1/n', separate=None, observe_nonempty=True,
                 dt_traj='1 step', conf=0.95, store_hidden=False):

        super(BayesianHMSM, self).__init__(nstates=nstates, lagtime=lagtime, stride=stride,
                                           reversible=reversible, stationary=stationary,
                                           connectivity=connectivity, mincount_connectivity=mincount_connectivity,
                                           observe_nonempty=observe_nonempty, separate=separate,
                                           dt_traj=dt_traj)
        self.p0_prior = p0_prior
        self.transition_matrix_prior = transition_matrix_prior
        self.nsamples = nsamples
        self.init_hmsm = init_hmsm
        self.conf = conf
        self.store_hidden = store_hidden

    def fetch_model(self) -> BayesianHMMPosterior:
        return self._model

    @property
    def init_hmsm(self):
        return self._init_hmsm

    @init_hmsm.setter
    def init_hmsm(self, value: typing.Optional[HMSM]):
        if value is not None and not issubclass(value.__class__, HMSM):
            raise ValueError('hmsm must be of type HMSM')
        self._init_hmsm = value

    def fit(self, dtrajs, call_back=None):
        self._model = BayesianHMMPosterior()
        dtrajs = ensure_dtraj_list(dtrajs)

        if self.init_hmsm is None:  # estimate using a maximum-likelihood hmm
            # TODO: we cannot call super().fit(), since it would overwrite the internal model again
            # TODO: it seems artificial, that bayesian_hmsm is a subclass of mlhmsm!
            self._model.prior = MaximumLikelihoodHMSM(nstates=self.nstates, lagtime=self.lagtime, stride=self.stride,
                                                      reversible=self.reversible, stationary=self.stationary, dt_traj=self.dt_traj,
                                                      separate=self.separate,
                                                      # these params from self.ctor will be overridden.
                                                      connectivity=None,
                                                      mincount_connectivity=0,
                                                      accuracy=1e-2,  # this is sufficient for an initial guess
                                                      observe_nonempty=False,
                                                      ).fit(dtrajs).fetch_model()
            dtrajs_lagged_strided = HMMTransitionCountModel.compute_dtrajs_effective(dtrajs, lagtime=self.lagtime,
                                                                                     nstates=self.nstates_full,
                                                                                     stride=self.stride)
        else:  # if given another initialization, must copy its attributes
            # check if nstates and lag are compatible
            if self.lagtime != self.init_hmsm.lagtime:
                raise ValueError('BayesianHMSM cannot be initialized with init_hmsm with incompatible lagtime.')
            if self.nstates != self.init_hmsm.nstates:
                raise ValueError('BayesianHMSM cannot be initialized with init_hmsm with incompatible nstates.')

            # EVALUATE STRIDE
            init_stride = self.init_hmsm.count_model.stride
            if self.stride == 'effective':
                from sktime.markovprocess.util import compute_effective_stride
                self.stride = compute_effective_stride(dtrajs, self.lagtime, self.nstates)

            # if stride is different to init_hmsm, check if microstates in lagged-strided trajs are compatible
            dtrajs_lagged_strided = HMMTransitionCountModel.compute_dtrajs_effective(
                dtrajs, lagtime=self.lagtime, nstates=self.nstates, stride=self.stride)
            if self.stride != init_stride:
                symbols = np.unique(np.concatenate(dtrajs_lagged_strided))
                if self.init_hmsm.count_model.symbols != symbols:
                    raise ValueError('Choice of stride has excluded a different set of microstates than in '
                                     'init_hmsm. Set of observed microstates in time-lagged strided trajectories '
                                     'must match to the one used for init_hmsm estimation.')

            # as mentioned in the docstring, take init_hmsm observed set observation probabilities
            self.observe_nonempty = False

            # update HMM Model
            self._model.prior = self.init_hmsm.copy()

        prior = self._model.prior
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
        nstates_full = number_of_states(dtrajs)

        if prior_count_model.nstates_obs < nstates_full:
            eps = 0.01 / nstates_full  # default output probability, in order to avoid zero columns
            # full state space output matrix. make sure there are no zero columns
            B_init = eps * np.ones((self.nstates, nstates_full), dtype=np.float64)
            # fill active states
            B_init[:, prior_count_model.observable_set] = np.maximum(eps, prior.observation_probabilities)
            # renormalize B to make it row-stochastic
            B_init /= B_init.sum(axis=1)[:, None]
        else:
            B_init = prior.observation_probabilities

        # HMM sampler
        if self.init_hmsm is not None:
            hmm_mle = self.init_hmsm.hmm
        else:
            hmm_mle = discrete_hmm(prior.initial_distribution, prior.transition_matrix, B_init)

        sampled_hmm = bayesian_hmm(dtrajs_lagged_strided, hmm_mle, nsample=self.nsamples,
                                   reversible=self.reversible, stationary=self.stationary,
                                   p0_prior=self.p0_prior, transition_matrix_prior=self.transition_matrix_prior,
                                   store_hidden=self.store_hidden, call_back=call_back).fetch_model()

        # Samples
        sample_inp = [(m.transition_matrix, m.stationary_distribution, m.output_model.output_probabilities,
                       m.initial_distribution)
                      for m in sampled_hmm]

        samples = []
        for P, pi, pobs, init_dist in sample_inp:  # restrict to observable set if necessary
            Bobs = pobs[:, prior_count_model.observable_set]
            pobs = Bobs / Bobs.sum(axis=1)[:, None]  # renormalize
            samples.append(HMSM(P, pobs, pi=pi, dt_model=self.dt_traj,
                                count_model=prior_count_model,
                                reversible=self.reversible, initial_distribution=init_dist))

        # store results
        if self.store_hidden:
            self._model.hidden_state_trajectories_samples = [s.hidden_state_trajectories for s in sampled_hmm]
        self._model.samples = samples

        return self
