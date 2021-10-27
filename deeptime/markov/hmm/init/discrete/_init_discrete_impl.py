from typing import Optional, Union

import numpy as np
from scipy.sparse import issparse

from deeptime.util.types import ensure_dtraj_list


def _regularize_hidden(initial_distribution, transition_matrix, reversible=True, stationary=False, count_matrix=None,
                       eps=None):
    """ Regularizes the hidden initial distribution and transition matrix.

    Makes sure that the hidden initial distribution and transition matrix have
    nonzero probabilities by setting them to eps and then renormalizing.
    Avoids zeros that would cause estimation algorithms to crash or get stuck
    in suboptimal states.

    Parameters
    ----------
    initial_distribution : ndarray(n)
        Initial hidden distribution of the HMM
    transition_matrix : ndarray(n, n)
        Hidden transition matrix
    reversible : bool
        HMM is reversible. Will make sure it is still reversible after modification.
    stationary : bool
        p0 is the stationary distribution of P. In this case, will not regularize
        p0 separately. If stationary=False, the regularization will be applied to p0.
    count_matrix : ndarray(n, n)
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
        from deeptime.markov._transition_matrix import enforce_reversible_on_closed
        transition_matrix = enforce_reversible_on_closed(transition_matrix)

    # REGULARIZE p0
    if stationary:
        from deeptime.markov._transition_matrix import stationary_distribution
        stationary_distribution(transition_matrix, C=count_matrix)
    else:
        initial_distribution = np.maximum(initial_distribution, eps)
        initial_distribution /= initial_distribution.sum()

    return initial_distribution, transition_matrix


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
    output_probabilities : ndarray(n, m)
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

    .. math::
        P_c = (M' M)^-1 M' P M

    Parameters
    ----------
    P : ndarray(n, n)
        microstate transition matrix
    M : ndarray(n, m)
        membership matrix. Membership to macrostate m for each microstate.

    Returns
    -------
    P_coarse : ndarray(m, m)
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


def metastable_from_msm(msm, n_hidden_states: int,
                        reversible: bool = True, stationary: bool = False,
                        separate_symbols=None, regularize: bool = True):
    r""" Makes an initial guess for an :class:`HMM <deeptime.markov.hmm.HiddenMarkovModel>` with
    discrete output model from an already existing MSM over observable states. The procedure is described in
    :footcite:`noe2013projected` and uses PCCA+ :footcite:`roblitz2013fuzzy` for
    coarse-graining the transition matrix and obtaining membership assignments.

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
    hmm_init : HiddenMarkovModel
        An initial guess for the HMM

    See Also
    --------
    deeptime.markov.hmm.DiscreteOutputModel
        The type of output model this heuristic uses.

    :func:`metastable_from_data`
        Initial guess from data if no MSM is available yet.

    :func:`deeptime.markov.hmm.init.gaussian.from_data`
        Initial guess with :class:`Gaussian output model <deeptime.markov.hmm.GaussianOutputModel>`.

    References
    ----------
    .. footbibliography::
    """
    from deeptime.markov._transition_matrix import stationary_distribution
    from deeptime.markov._transition_matrix import estimate_P
    from deeptime.markov.msm import MarkovStateModel
    from deeptime.markov import PCCAModel

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
        separate_states = None

    hidden_transition_matrix = _coarse_grain_transition_matrix(msm.transition_matrix, memberships)
    if reversible:
        from deeptime.markov._transition_matrix import enforce_reversible_on_closed
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
                                                             stationary=stationary, count_matrix=hidden_counts, eps=eps_a)
    eps_b = 0.01 / msm.n_states if regularize else 0.
    output_probabilities = _regularize_pobs(output_probabilities, nonempty=None, separate=separate_symbols, eps=eps_b)
    from deeptime.markov.hmm import HiddenMarkovModel
    return HiddenMarkovModel(transition_model=hidden_transition_matrix, output_model=output_probabilities,
                             initial_distribution=hidden_pi)


def metastable_from_data(dtrajs, n_hidden_states, lagtime, stride=1, mode='largest-regularized',
                         reversible: bool = True, stationary: bool = False,
                         separate_symbols=None, states: Optional[np.ndarray] = None,
                         regularize: bool = True, connectivity_threshold: Union[str, float] = 0.):
    r"""Estimates an initial guess :class:`HMM <deeptime.markov.hmm.HiddenMarkovModel>` from given
    discrete trajectories.

    Following the procedure described in :footcite:`noe2013projected`: First
    a :class:`MSM <deeptime.markov.msm.MarkovStateModel>` is estimated, which is then subsequently
    coarse-grained with PCCA+ :footcite:`roblitz2013fuzzy`. After estimation of the MSM, this
    method calls :meth:`metastable_from_msm`.

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
          :meth:`TransitionCountModel.submodel_largest <deeptime.markov.TransitionCountModel.submodel_largest>`.
        * populus: the connected set with the largest population in the data, see
          :meth:`TransitionCountModel.submodel_largest <deeptime.markov.TransitionCountModel.submodel_largest>`.

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
    hmm_init : HiddenMarkovModel
        An initial guess for the HMM

    See Also
    --------
    DiscreteOutputModel
        The type of output model this heuristic uses.

    :func:`metastable_from_msm`
        Initial guess from an already existing :class:`MSM <deeptime.markov.msm.MarkovStateModel>`.

    :func:`deeptime.markov.hmm.init.gaussian.from_data`
        Initial guess with :class:`Gaussian output model <deeptime.markov.hmm.GaussianOutputModel>`.


    References
    ----------
    .. footbibliography::
    """
    if mode not in metastable_from_data.VALID_MODES \
            + [m + "-regularized" for m in metastable_from_data.VALID_MODES]:
        raise ValueError("mode can only be one of [{}]".format(", ".join(metastable_from_data.VALID_MODES)))

    from deeptime.markov import compute_dtrajs_effective, TransitionCountEstimator

    dtrajs = ensure_dtraj_list(dtrajs)
    dtrajs = compute_dtrajs_effective(dtrajs, lagtime=lagtime, n_states=n_hidden_states, stride=stride)
    counts = TransitionCountEstimator(1, 'sliding', sparse=False).fit(dtrajs).fetch_model()
    if states is not None:
        counts = counts.submodel(states)
    if '-regularized' in mode:
        import deeptime.markov.tools.estimation as memest
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
    from deeptime.markov.msm import MaximumLikelihoodMSM
    msm = MaximumLikelihoodMSM(reversible=True, allow_disconnected=True, maxerr=1e-3,
                               maxiter=10000).fit(counts).fetch_model()
    return metastable_from_msm(msm, n_hidden_states, reversible, stationary, separate_symbols, regularize)


metastable_from_data.VALID_MODES = ['all', 'largest', 'populous']


def random_guess(n_observation_states: int, n_hidden_states: int, seed: Optional[int] = None):
    r"""Initializes a :class:`HMM <deeptime.markov.hmm.HiddenMarkovModel>` with a set number of hidden and
    observable states by setting the transition matrix uniform and drawing a random row-stochastic matrix as
    output probabilities.

    Parameters
    ----------
    n_observation_states : int
        The number of states in observable space.
    n_hidden_states : int
        The number of hidden states.
    seed : int, optional, default=None
        The random seed.

    Returns
    -------
    init_hmm : HiddenMarkovModel
        A randomly initialized hidden markov state model.
    """
    state = np.random.RandomState(seed=seed)
    P = np.empty((n_hidden_states, n_hidden_states))
    P.fill(1. / n_hidden_states)
    B = state.uniform(size=(n_hidden_states, n_observation_states))
    B /= B.sum(axis=-1, keepdims=True)
    from deeptime.markov.hmm import HiddenMarkovModel
    return HiddenMarkovModel(transition_model=P, output_model=B)
