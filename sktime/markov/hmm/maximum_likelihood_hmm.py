# This file is part of sktime.
#
# Copyright (c) 2020, 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# sktime is free software: you can redistribute it and/or modify
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
from typing import List, Union, Optional

import numpy as np
from scipy.sparse import issparse

from sktime.base import Estimator
from sktime.markov import Q_
from sktime.markov.transition_counting import TransitionCountEstimator, TransitionCountModel
from sktime.markov._transition_matrix import estimate_P, stationary_distribution, enforce_reversible_on_closed
from sktime.markov.hmm import HiddenMarkovStateModel
from sktime.markov.hmm.hmm import viterbi
from sktime.markov.hmm.output_model import GaussianOutputModel
from sktime.markov.msm import MarkovStateModel, MaximumLikelihoodMSM
from sktime.markov.pcca import PCCAModel
from sktime.markov.util import compute_dtrajs_effective
from sktime.util import ensure_dtraj_list
from sktime.markov.hmm._hmm_bindings import util as _util


def _regularize_hidden(p0, transition_matrix, reversible=True, stationary=False, C=None, eps=None):
    """ Regularizes the hidden initial distribution and transition matrix.

    Makes sure that the hidden initial distribution and transition matrix have
    nonzero probabilities by setting them to eps and then renormalizing.
    Avoids zeros that would cause estimation algorithms to crash or get stuck
    in suboptimal states.

    Parameters
    ----------
    p0 : ndarray(n)
        Initial hidden distribution of the HMM
    transition_matrix : ndarray(n, n)
        Hidden transition matrix
    reversible : bool
        HMM is reversible. Will make sure it is still reversible after modification.
    stationary : bool
        p0 is the stationary distribution of P. In this case, will not regularize
        p0 separately. If stationary=False, the regularization will be applied to p0.
    C : ndarray(n, n)
        Hidden count matrix. Only needed for stationary=True and P disconnected.
    eps : float or None
        minimum value of the resulting transition matrix. Default: evaluates
        to 0.01 / n. The coarse-graining equation can lead to negative elements
        and thus epsilon should be set to at least 0. Positive settings of epsilon
        are similar to a prior and enforce minimum positive values for all
        transition probabilities.

    Return
    ------
    p0 : ndarray(n)
        regularized initial distribution
    P : ndarray(n, n)
        regularized transition matrix

    """
    # input
    n = transition_matrix.shape[0]
    if eps is None:  # default output probability, in order to avoid zero columns
        eps = 0.01 / n

    # REGULARIZE P
    transition_matrix = np.maximum(transition_matrix, eps)
    # and renormalize
    transition_matrix /= transition_matrix.sum(axis=1)[:, None]
    # ensure reversibility
    if reversible:
        transition_matrix = enforce_reversible_on_closed(transition_matrix)

    # REGULARIZE p0
    if stationary:
        stationary_distribution(transition_matrix, C=C)
    else:
        p0 = np.maximum(p0, eps)
        p0 /= p0.sum()

    return p0, transition_matrix


def _regularize_pobs(output_probabilities, nonempty=None, separate=None, eps=None):
    """ Regularizes the output probabilities.

    Makes sure that the output probability distributions has
    nonzero probabilities by setting them to eps and then renormalizing.
    Avoids zeros that would cause estimation algorithms to crash or get stuck
    in suboptimal states.

    Parameters
    ----------
    output_probabilities : ndarray(n, m)
        HMM output probabilities
    nonempty : None or iterable of int
        Nonempty set. Only regularize on this subset.
    separate : None or iterable of int
        Force the given set of observed states to stay in a separate hidden state.
        The remaining n_states-1 states will be assigned by a metastable decomposition.

    Returns
    -------
    B : ndarray(n, m)
        Regularized output probabilities

    """
    # input
    output_probabilities = output_probabilities.copy()  # modify copy
    n, m = output_probabilities.shape  # number of hidden / observable states
    if eps is None:  # default output probability, in order to avoid zero columns
        eps = 0.01 / m
    # observable sets
    if nonempty is None:
        nonempty = np.arange(m)

    if separate is None:
        output_probabilities[:, nonempty] = np.maximum(output_probabilities[:, nonempty], eps)
    else:
        nonempty_nonseparate = np.array(list(set(nonempty) - set(separate)), dtype=int)
        nonempty_separate = np.array(list(set(nonempty).intersection(set(separate))), dtype=int)
        output_probabilities[:n - 1, nonempty_nonseparate] = np.maximum(
            output_probabilities[:n - 1, nonempty_nonseparate], eps)
        output_probabilities[n - 1, nonempty_separate] = np.maximum(output_probabilities[n - 1, nonempty_separate], eps)

    # renormalize and return copy
    output_probabilities /= output_probabilities.sum(axis=1)[:, None]
    return output_probabilities


def _coarse_grain_transition_matrix(P, M):
    """ Coarse grain transition matrix P using memberships M

    Computes

    .. math:
        Pc = (M' M)^-1 M' P M

    Parameters
    ----------
    P : ndarray(n, n)
        microstate transition matrix
    M : ndarray(n, m)
        membership matrix. Membership to macrostate m for each microstate.

    Returns
    -------
    Pc : ndarray(m, m)
        coarse-grained transition matrix.

    """
    # coarse-grain matrix: Pc = (M' M)^-1 M' P M
    W = np.linalg.inv(np.dot(M.T, M))
    A = np.dot(np.dot(M.T, P), M)
    P_coarse = np.dot(W, A)

    # this coarse-graining can lead to negative elements. Setting them to zero here.
    P_coarse = np.maximum(P_coarse, 0)
    # and renormalize
    P_coarse /= P_coarse.sum(axis=1)[:, None]

    return P_coarse


def initial_guess_discrete_from_msm(msm: MarkovStateModel, n_hidden_states: int,
                                    reversible: bool = True, stationary: bool = False,
                                    separate_symbols=None, regularize: bool = True) -> HiddenMarkovStateModel:
    r""" Makes an initial guess for an :class:`HMM <sktime.markov.hmm.HiddenMarkovStateModel>` with
    discrete output model from an already existing MSM over observable states. The procedure is described in [1]_ and
    uses PCCA+ [2]_ for coarse-graining the transition matrix and obtaining membership assignments.

    Parameters
    ----------
    msm : MarkovStateModel
        The markov state model over observable state space.
    n_hidden_states : int
        The desired number of hidden states.
    reversible : bool, optional, default=True
        Whether the HMM transition matrix is estimated so that it is reversibe.
    stationary : bool, optional, default=False
        If True, the initial distribution of hidden states is self-consistently computed as the stationary
        distribution of the transition matrix. If False, it will be estimated from the starting states.
        Only set this to true if you're sure that the observation trajectories are initiated from a global
        equilibrium distribution.
    separate_symbols : array_like, optional, default=None
        Force the given set of observed states to stay in a separate hidden state.
        The remaining nstates-1 states will be assigned by a metastable decomposition.
    regularize : bool, optional, default=True
        If set to True, makes sure that the hidden initial distribution and transition matrix have nonzero probabilities
        by setting them to eps and then renormalizing. Avoids zeros that would cause estimation algorithms to crash or
        get stuck in suboptimal states.

    Returns
    -------
    hmm_init : HiddenMarkovStateModel
        An initial guess for the HMM

    See Also
    --------
    DiscreteOutputModel : The type of output model this heuristic uses.
    initial_guess_discrete_from_data : Initial guess from data if no MSM is available yet
    initial_guess_gaussian_from_data : Initial guess with :class:`Gaussian output model <sktime.markov.hmm.GaussianOutputModel>`.

    References
    ----------
    .. [1] F. Noe, H. Wu, J.-H. Prinz and N. Plattner: Projected and hidden Markov models for calculating kinetics and
       metastable states of complex molecules. J. Chem. Phys. 139, 184114 (2013)
    .. [2] S. Roeblitz and M. Weber, Fuzzy spectral clustering by PCCA+:
       application to Markov state models and data classification.
       Adv Data Anal Classif 7, 147-179 (2013).
    """
    count_matrix = msm.count_model.count_matrix
    nonseparate_symbols = np.arange(msm.count_model.n_states_full)
    nonseparate_states = msm.count_model.symbols_to_states(nonseparate_symbols)
    nonseparate_msm = msm
    if separate_symbols is not None:
        separate_symbols = np.asanyarray(separate_symbols)
        if np.max(separate_symbols) >= msm.count_model.n_states_full:
            raise ValueError(f'Separate set has indices that do not exist in '
                             f'full state space: {np.max(separate_symbols)}')
        nonseparate_symbols = np.setdiff1d(nonseparate_symbols, separate_symbols)
        nonseparate_states = msm.count_model.symbols_to_states(nonseparate_symbols)
        nonseparate_count_model = msm.count_model.submodel(nonseparate_states)
        # make reversible
        nonseparate_count_matrix = nonseparate_count_model.count_matrix
        if issparse(nonseparate_count_matrix):
            nonseparate_count_matrix = nonseparate_count_matrix.toarray()
        P_nonseparate = estimate_P(nonseparate_count_matrix, reversible=True)
        pi = stationary_distribution(P_nonseparate, C=nonseparate_count_matrix)
        nonseparate_msm = MarkovStateModel(P_nonseparate, stationary_distribution=pi)
    if issparse(count_matrix):
        count_matrix = count_matrix.toarray()

    # if #metastable sets == #states, we can stop here
    n_meta = n_hidden_states if separate_symbols is None else n_hidden_states - 1
    if n_meta == nonseparate_msm.n_states:
        pcca = PCCAModel(nonseparate_msm.transition_matrix, nonseparate_msm.stationary_distribution, np.eye(n_meta),
                         np.eye(n_meta))
    else:
        pcca = nonseparate_msm.pcca(n_meta)
    if separate_symbols is not None:
        separate_states = msm.count_model.symbols_to_states(separate_symbols)
        memberships = np.zeros((msm.n_states, n_hidden_states))
        memberships[nonseparate_states, :n_hidden_states - 1] = pcca.memberships
        memberships[separate_states, -1] = 1
    else:
        memberships = pcca.memberships

    hidden_transition_matrix = _coarse_grain_transition_matrix(msm.transition_matrix, memberships)
    if reversible:
        hidden_transition_matrix = enforce_reversible_on_closed(hidden_transition_matrix)

    hidden_counts = memberships.T.dot(count_matrix).dot(memberships)
    hidden_pi = stationary_distribution(hidden_transition_matrix, C=hidden_counts)

    output_probabilities = np.zeros((n_hidden_states, msm.count_model.n_states_full))
    # we might have lost a few symbols, reduce nonsep symbols to the ones actually represented
    nonseparate_symbols = msm.count_model.state_symbols[nonseparate_states]
    if separate_symbols is not None:
        separate_symbols = msm.count_model.state_symbols[separate_states]
        output_probabilities[:n_hidden_states - 1, nonseparate_symbols] = pcca.metastable_distributions
        output_probabilities[-1, separate_symbols] = msm.stationary_distribution[separate_states]
    else:
        output_probabilities[:, nonseparate_symbols] = pcca.metastable_distributions

    # regularize
    eps_a = 0.01 / n_hidden_states if regularize else 0.
    hidden_pi, hidden_transition_matrix = _regularize_hidden(hidden_pi, hidden_transition_matrix, reversible=reversible,
                                                             stationary=stationary, C=hidden_counts, eps=eps_a)
    eps_b = 0.01 / msm.n_states if regularize else 0.
    output_probabilities = _regularize_pobs(output_probabilities, nonempty=None, separate=separate_symbols, eps=eps_b)
    return HiddenMarkovStateModel(transition_model=hidden_transition_matrix, output_model=output_probabilities,
                                  initial_distribution=hidden_pi)


def initial_guess_discrete_from_data(dtrajs, n_hidden_states, lagtime, stride=1, mode='largest-regularized',
                                     reversible: bool = True, stationary: bool = False,
                                     separate_symbols=None, states: Optional[np.ndarray] = None,
                                     regularize: bool = True, connectivity_threshold: Union[str, float] = 0.):
    r"""Estimates an initial guess :class:`HMM <sktime.markov.hmm.HiddenMarkovStateModel>` from given
    discrete trajectories.

    Following the procedure described in [1]_: First a :class:`MSM <sktime.markov.msm.MarkovStateModel>` is
    estimated, which is then subsequently coarse-grained with PCCA+ [2]_. After estimation of the MSM, this method
    class :meth:`initial_guess_discrete_from_msm`.

    Parameters
    ----------
    dtrajs : array_like or list of array_like
        A discrete trajectory or a list of discrete trajectories.
    n_hidden_states : int
        Number of hidden states.
    lagtime : int
        The lagtime at which transitions are counted.
    stride : int or str, optional, default=1
        stride between two lagged trajectories extracted from the input trajectories. Given trajectory :code:`s[t]`,
        stride and lag will result in trajectories

            :code:`s[0], s[lag], s[2 lag], ...`

            :code:`s[stride], s[stride + lag], s[stride + 2 lag], ...`

        Setting stride = 1 will result in using all data (useful for maximum likelihood estimator), while a Bayesian
        estimator requires a longer stride in order to have statistically uncorrelated trajectories. Setting
        :code:`stride='effective'` uses the largest neglected timescale as an estimate for the correlation time
        and sets the stride accordingly.
    mode : str, optional, default='largest-regularized'
        The mode at which the markov state model is estimated. Since the process is assumed to be reversible and
        finite statistics might lead to unconnected regions in state space, a subselection can automatically be made
        and the count matrix can be regularized. The following options are available:

        * 'all': all available states are taken into account
        * 'largest': the largest connected state set is selected, see
          :meth:`TransitionCountModel.submodel_largest <sktime.markov.TransitionCountModel.submodel_largest>`.
        * populus: the connected set with the largest population in the data, see
          :meth:`TransitionCountModel.submodel_largest <sktime.markov.TransitionCountModel.submodel_largest>`.

        For regularization, each of the options can be suffixed by a '-regularized', e.g., 'largest-regularized'.
        This means that the count matrix has no zero entries and everything is reversibly connected. In particular,
        a prior of the form

        .. math:: b_{ij}=\left \{ \begin{array}{rl}
                     \alpha & \text{, if }c_{ij}+c_{ji}>0, \\
                     0      & \text{, otherwise,}
                     \end{array} \right .

        with :math:`\alpha=10^{-3}` is added and all non-reversibly connected components are artifically connected
        by adding backward paths.
    reversible : bool, optional, default=True
        Whether the HMM transition matrix is estimated so that it is reversibe.
    stationary : bool, optional, default=False
        If True, the initial distribution of hidden states is self-consistently computed as the stationary
        distribution of the transition matrix. If False, it will be estimated from the starting states.
        Only set this to true if you're sure that the observation trajectories are initiated from a global
        equilibrium distribution.
    separate_symbols : array_like, optional, default=None
        Force the given set of observed states to stay in a separate hidden state.
        The remaining nstates-1 states will be assigned by a metastable decomposition.
    states : (dtype=int) ndarray, optional, default=None
        Artifically restrict count model to selection of states, even before regularization.
    regularize : bool, optional, default=True
        If set to True, makes sure that the hidden initial distribution and transition matrix have nonzero probabilities
        by setting them to eps and then renormalizing. Avoids zeros that would cause estimation algorithms to crash or
        get stuck in suboptimal states.
    connectivity_threshold : float or '1/n', optional, default=0.
        Connectivity threshold. counts that are below the specified value are disregarded when finding connected
        sets. In case of '1/n', the threshold gets resolved to :math:`1 / \mathrm{n\_states\_full}`.

    Returns
    -------
    hmm_init : HiddenMarkovStateModel
        An initial guess for the HMM

    See Also
    --------
    DiscreteOutputModel : The type of output model this heuristic uses.
    initial_guess_discrete_from_msm : Initial guess from an already existing :class:`MSM <sktime.markov.msm.MarkovStateModel>`.
    initial_guess_gaussian_from_data : Initial guess with :class:`Gaussian output model <sktime.markov.hmm.GaussianOutputModel>`.

    References
    ----------
    .. [1] F. Noe, H. Wu, J.-H. Prinz and N. Plattner: Projected and hidden Markov models for calculating kinetics and
       metastable states of complex molecules. J. Chem. Phys. 139, 184114 (2013)
    .. [2] S. Roeblitz and M. Weber, Fuzzy spectral clustering by PCCA+:
       application to Markov state models and data classification.
       Adv Data Anal Classif 7, 147-179 (2013).
    """
    if mode not in initial_guess_discrete_from_data.VALID_MODES \
            + [m + "-regularized" for m in initial_guess_discrete_from_data.VALID_MODES]:
        raise ValueError("mode can only be one of [{}]".format(", ".join(initial_guess_discrete_from_data.VALID_MODES)))

    dtrajs = ensure_dtraj_list(dtrajs)
    dtrajs = compute_dtrajs_effective(dtrajs, lagtime=lagtime, n_states=n_hidden_states, stride=stride)
    counts = TransitionCountEstimator(1, 'sliding', sparse=False).fit(dtrajs).fetch_model()
    if states is not None:
        counts = counts.submodel(states)
    if '-regularized' in mode:
        import msmtools.estimation as memest
        counts.count_matrix[...] += memest.prior_neighbor(counts.count_matrix, 0.001)
        nonempty = np.where(counts.count_matrix.sum(axis=0) + counts.count_matrix.sum(axis=1) > 0)[0]
        counts.count_matrix[nonempty, nonempty] = np.maximum(counts.count_matrix[nonempty, nonempty], 0.001)
    if 'all' in mode:
        pass  # no-op
    if 'largest' in mode:
        counts = counts.submodel_largest(directed=True, connectivity_threshold=connectivity_threshold,
                                         sort_by_population=False)
    if 'populous' in mode:
        counts = counts.submodel_largest(directed=True, connectivity_threshold=connectivity_threshold,
                                         sort_by_population=True)
    msm = MaximumLikelihoodMSM(reversible=True, allow_disconnected=True, maxiter=10000).fit(counts).fetch_model()
    return initial_guess_discrete_from_msm(msm, n_hidden_states, reversible, stationary, separate_symbols, regularize)


initial_guess_discrete_from_data.VALID_MODES = ['all', 'largest', 'populous']


def initial_guess_gaussian_from_data(dtrajs, n_hidden_states, reversible):
    r""" Makes an initial guess :class:`HMM <HiddenMarkovStateModel>` with Gaussian output model.

    To this end, a Gaussian mixture model is estimated using `scikit-learn <https://scikit-learn.org/>`_.

    Parameters
    ----------
    dtrajs : array_like or list of array_like
        Trajectories which are used for making the initial guess.
    n_hidden_states : int
        Number of hidden states.
    reversible : bool
        Whether the hidden transition matrix is estimated so that it is reversible.

    Returns
    -------
    hmm_init : HiddenMarkovStateModel
        An initial guess for the HMM

    See Also
    --------
    GaussianOutputModel : The type of output model this heuristic uses.
    initial_guess_discrete_from_data : Initial guess with :class:`Discrete output model <sktime.markov.hmm.DiscreteOutputModel>`.
    initial_guess_discrete_from_msm : Initial guess from an already
                                      existing :class:`MSM <sktime.markov.msm.MarkovStateModel>`
                                      with discrete output model.
    """
    # todo docs
    from sklearn.mixture import GaussianMixture
    # todo we dont actually want to depend on sklearn
    dtrajs = ensure_dtraj_list(dtrajs)
    collected_observations = np.concatenate(dtrajs)
    gmm = GaussianMixture(n_components=n_hidden_states)
    gmm.fit(collected_observations[:, None])
    output_model = GaussianOutputModel(n_hidden_states, means=gmm.means_[:, 0], sigmas=np.sqrt(gmm.covariances_[:, 0]))

    # Compute fractional state memberships.
    Nij = np.zeros((n_hidden_states, n_hidden_states))
    for o_t in dtrajs:
        # length of trajectory
        T = o_t.shape[0]
        # output probability
        pobs = output_model.to_state_probability_trajectory(o_t)
        # normalize
        pobs /= pobs.sum(axis=1)[:, None]
        # Accumulate fractional transition counts from this trajectory.
        for t in range(T - 1):
            Nij += np.outer(pobs[t, :], pobs[t + 1, :])

    # Compute transition matrix maximum likelihood estimate.
    import msmtools.estimation as msmest
    import msmtools.analysis as msmana
    Tij = msmest.transition_matrix(Nij, reversible=reversible)
    pi = msmana.stationary_distribution(Tij)
    return HiddenMarkovStateModel(transition_model=Tij, output_model=output_model, initial_distribution=pi)


class MaximumLikelihoodHMSM(Estimator):
    """ Maximum likelihood Hidden Markov model (HMM) estimator.

    This class is used to fit a maximum-likelihood HMM to data. It uses an initial guess HMM with can be obtained with
    one of the provided heuristics and then uses the Baum-Welch algorithm [1]_ to fit the inital guess to provided
    data.

    References
    ----------
    .. [1] L. E. Baum and J. A. Egon, "An inequality with applications to statistical
           estimation for probabilistic functions of a Markov process and to a model
           for ecology," Bull. Amer. Meteorol. Soc., vol. 73, pp. 360-363, 1967.
    """

    _HMMModelStorage = collections.namedtuple('_HMMModelStorage', ['transition_matrix', 'output_model',
                                                                   'initial_distribution'])

    def __init__(self, initial_model: HiddenMarkovStateModel, stride: Union[int, str] = 1,
                 lagtime: int = 1, reversible: bool = True, stationary: bool = False,
                 p: Optional[np.ndarray] = None, physical_time: str = '1 step', accuracy: float = 1e-3,
                 maxit: int = 1000, maxit_reversible: int = 100000):
        r"""
        Initialize a maximum likelihood hidden Markov model estimator. The type of output model (gaussian or discrete)
        and the number of hidden states are extracted from the initial model. In case no initial distribution was given,
        the initial model assumes a uniform initial distribution.

        Parameters
        ----------
        initial_model : HiddenMarkovStateModel
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
        accuracy : float, optional, default=1e-3
            Convergence threshold for EM iteration. When two the likelihood does not increase by more than
            accuracy, the iteration is stopped successfully.
        maxit : int, optional, default=1000
            Stopping criterion for EM iteration. When this many iterations are performed without reaching the requested
            accuracy, the iteration is stopped without convergence and a warning is given.
        maxit_reversible : int, optional, default=1000000
            Maximum number of iterations for reversible transition matrix estimation. Only used with reversible=True.
        """
        super().__init__()
        self.initial_transition_model = initial_model
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
        self.physical_time = physical_time

    def fetch_model(self) -> HiddenMarkovStateModel:
        r""" Yields the current HiddenMarkovStateModel or None if :meth:`fit` was not called yet.

        Returns
        -------
        model : HiddenMarkovStateModel or None
            The model.
        """
        return self._model

    @property
    def physical_time(self) -> Q_:
        r""" A description of the physical time.

        :getter: Yields a description of physical time in terms of a pint Quantity object.
        :setter: Sets a description of the physical time for input trajectories. Specify by a number, whitespace,
                 and unit. Permitted units are 'fs', 'ps', 'ns', 'us', 'ms', 's', and 'step'.
        :type: pint.Quantity or str
        """
        return self._physical_time

    @physical_time.setter
    def physical_time(self, value: str):
        self._physical_time = Q_(value)

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
        return self.initial_transition_model.n_hidden_states

    @property
    def initial_transition_model(self) -> HiddenMarkovStateModel:
        r""" The initial transition model. """
        return self._initial_transition_model

    @initial_transition_model.setter
    def initial_transition_model(self, value: HiddenMarkovStateModel) -> None:
        self._initial_transition_model = value

    def fit(self, dtrajs, initial_model=None, **kwargs):
        r""" Fits a new :class:`HMM <HiddenMarkovStateModel>` to data.

        Parameters
        ----------
        dtrajs : array_like or list of array_like
            Timeseries data.
        initial_model : HiddenMarkovStateModel, optional, default=None
            Override for :attr:`initial_transition_model`.
        **kwargs
            Ignored kwargs for scikit-learn compatibility.

        Returns
        -------
        self : MaximumLikelihoodHMSM
            Reference to self.
        """
        if initial_model is None:
            initial_model = self.initial_transition_model
        if initial_model is None or not isinstance(initial_model, HiddenMarkovStateModel):
            raise ValueError("For estimation, an initial model of type "
                             "`sktime.markov.hmm.HiddenMarkovStateModel` is required.")

        # copy initial model
        transition_matrix = initial_model.transition_model.transition_matrix
        if issparse(transition_matrix):
            # want dense matrix, toarray makes a copy
            transition_matrix = transition_matrix.toarray()
        else:
            # new instance
            transition_matrix = np.copy(transition_matrix)

        hmm_data = MaximumLikelihoodHMSM._HMMModelStorage(transition_matrix=transition_matrix,
                                                          output_model=initial_model.output_model.copy(),
                                                          initial_distribution=initial_model.initial_distribution.copy())

        dtrajs = ensure_dtraj_list(dtrajs)
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
                if dL < self._accuracy:
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

        count_model = TransitionCountModel(count_matrix=transition_counts, lagtime=self.lagtime,
                                           physical_time=self.physical_time)
        transition_model = MarkovStateModel(hmm_data.transition_matrix, reversible=self.reversible,
                                            count_model=count_model)
        hidden_state_trajs = [
            viterbi(hmm_data.transition_matrix, hmm_data.output_model.to_state_probability_trajectory(obs),
                    hmm_data.initial_distribution) for obs in dtrajs
        ]
        model = HiddenMarkovStateModel(
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
        # forward variables
        logprob = _util.forward(A, pobs, pi, alpha_out=alpha, T=T)
        # backward variables
        _util.backward(A, pobs, beta_out=beta, T=T)
        # gamma
        _util.state_probabilities(alpha, beta, gamma_out=gamma, T=T)
        # count matrix
        _util.transition_counts(alpha, beta, A, pobs, counts_out=counts, T=T)
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
