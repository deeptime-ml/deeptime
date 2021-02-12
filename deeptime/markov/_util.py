from typing import Union

import numpy as np


def visited_set(dtrajs):
    r"""returns the set of states that have at least one count

    Parameters
    ----------
    dtrajs : array_like or list of array_like
        Discretized trajectory or list of discretized trajectories

    Returns
    -------
    vis : ndarray((n), dtype=int)
        the set of states that have at least one count.
    """
    from deeptime.util.types import ensure_dtraj_list
    dtrajs = ensure_dtraj_list(dtrajs)
    hist = count_states(dtrajs)
    return np.argwhere(hist > 0)[:, 0]


def count_states(dtrajs, ignore_negative: bool = False):
    r"""Computes a histogram over the visited states in one or multiple discretized trajectories.

    Parameters
    ----------
    dtrajs : array_like or list of array_like
        Discretized trajectory or list of discretized trajectories
    ignore_negative : bool, default=False
        Ignore negative elements. By default, a negative element will cause an
        exception

    Returns
    -------
    count : ndarray((n), dtype=int)
        the number of occurrences of each state. n=max+1 where max is the largest state index found.

    """
    from deeptime.util.types import ensure_dtraj_list
    dtrajs = ensure_dtraj_list(dtrajs)

    max_n_states = 0
    histograms = []
    for discrete_trajectory in dtrajs:
        if ignore_negative:
            discrete_trajectory = discrete_trajectory[np.where(discrete_trajectory >= 0)]
        trajectory_histogram = np.bincount(discrete_trajectory)
        max_n_states = max(max_n_states, trajectory_histogram.shape[0])
        histograms.append(trajectory_histogram)
    # allocate space for histogram
    res = np.zeros(max_n_states, dtype=int)
    # aggregate histograms over trajectories
    for trajectory_histogram in histograms:
        res[:trajectory_histogram.shape[0]] += trajectory_histogram
    return res


def compute_effective_stride(dtrajs, lagtime, n_states) -> int:
    r"""
    Computes the effective stride which is an estimate of the striding required to produce uncorrelated samples.
    By default this is the lagtime (lag sampling). A nonreversible MSM is estimated, if its number of states is larger
    than the number of states provided to this method, stride is set to the minimum of lagtime and two times the
    correlation time of the next neglected timescale.

    Parameters
    ----------
    dtrajs : array_like or list of array_like
        Discretized trajectory or list of discretized trajectories
    lagtime : int
        Lagtime
    n_states : int
        Number of resolved states

    Returns
    -------
    stride : int
        Estimated effective stride to produce approximately uncorrelated samples
    """
    from deeptime.util.types import ensure_dtraj_list
    dtrajs = ensure_dtraj_list(dtrajs)
    # by default use lag as stride (=lag sampling), because we currently have no better theory for deciding
    # how many uncorrelated counts we can make
    stride = lagtime
    # get a quick fit from the spectral radius of the non-reversible
    from deeptime.markov import TransitionCountEstimator
    count_model = TransitionCountEstimator(lagtime=lagtime, count_mode="sliding").fit(dtrajs).fetch_model()
    count_model = count_model.submodel_largest()
    from deeptime.markov.msm import MaximumLikelihoodMSM
    msm_non_rev = MaximumLikelihoodMSM(reversible=False, sparse=False).fit(count_model).fetch_model()
    # if we have more than n_states timescales in our MSM, we use the next (neglected) timescale as an
    # fit of the de-correlation time
    if msm_non_rev.n_states > n_states:
        # because we use non-reversible msm, we want to silence the ImaginaryEigenvalueWarning
        import warnings
        with warnings.catch_warnings():
            from deeptime.util.exceptions import ImaginaryEigenValueWarning
            warnings.filterwarnings('ignore', category=ImaginaryEigenValueWarning)
            correlation_time = max(1, msm_non_rev.timescales()[n_states - 1])
        # use the smaller of these two pessimistic estimates
        stride = int(min(lagtime, 2 * correlation_time))

    return stride


def lag_observations(observations, lag, stride=1):
    r""" Create new trajectories that are subsampled at lag but shifted

    Given a trajectory (s0, s1, s2, s3, s4, ...) and lag 3, this function will generate 3 trajectories
    (s0, s3, s6, ...), (s1, s4, s7, ...) and (s2, s5, s8, ...). Use this function in order to parametrize a MLE
    at lag times larger than 1 without discarding data. Do not use this function for Bayesian estimators, where
    data must be given such that subsequent transitions are uncorrelated.

    Parameters
    ----------
    observations : array_like or list of array_like
        observation trajectories
    lag : int
        lag time
    stride : int, default=1
        will return only one trajectory for every stride. Use this for Bayesian analysis.

    """
    # todo cppify
    from deeptime.util.types import ensure_timeseries_data
    observations = ensure_timeseries_data(observations)
    obsnew = []
    for obs in observations:
        for shift in range(0, lag, stride):
            obs_lagged = obs[shift::lag]
            if len(obs_lagged) > 1:
                obsnew.append(obs_lagged)
    return obsnew


def compute_dtrajs_effective(dtrajs, lagtime: int, n_states: int, stride: Union[int, str]):
    r"""
    Takes discrete trajectories as input and strides these with an effective stride. See methods
    `compute_effective_stride` and `lag_observations`.

    Parameters
    ----------
    dtrajs : array_like or list of array_like
        discrete trajectories
    lagtime : int
        lagtime
    n_states : int
        number of resolved states
    stride : int or str
        if 'effective', computes effective stride, otherwise uses int value

    Returns
    -------
    Lagged and stridden observations.
    """
    lagtime = int(lagtime)
    # EVALUATE STRIDE
    if stride == 'effective':
        stride = compute_effective_stride(dtrajs, lagtime, n_states)

    # LAG AND STRIDE DATA
    dtrajs_lagged_strided = lag_observations(dtrajs, lagtime, stride=stride)
    return dtrajs_lagged_strided


def compute_connected_sets(count_matrix, connectivity_threshold: float = 0, directed=True):
    """ Computes the connected sets of a count matrix C.

    C : (N, N) np.ndarray
        count matrix
    mincount_connectivity : float
        Minimum count required to be included in the connected set computation.
    directed : boolean
        True: Seek connected sets in the directed graph. False: Seek connected sets in the undirected graph.
    Returns
    -------
    A list of arrays, each array representing a connected set by enumerating the respective states. The list is in
    descending order by size of connected set.
    """
    import deeptime.markov.tools.estimation as msmest
    import scipy.sparse as scs
    if connectivity_threshold > 0:
        if scs.issparse(count_matrix):
            Cconn = count_matrix.tocsr(copy=True)
            Cconn.data[Cconn.data < connectivity_threshold] = 0
            Cconn.eliminate_zeros()
        else:
            Cconn = count_matrix.copy()
            Cconn[np.where(Cconn < connectivity_threshold)] = 0
    else:
        Cconn = count_matrix
    # treat each connected set separately
    S = msmest.connected_sets(Cconn, directed=directed)
    return S


def number_of_states(dtrajs, only_used=False) -> int:
    r"""Returns the number of states in the given trajectories.

    Parameters
    ----------
    dtrajs : array_like or list of array_like
        Discretized trajectory or list of discretized trajectories
    only_used : bool, default=False
        If False, will return max+1, where max is the largest index used.
        If True, will return the number of states that occur at least once.

    Returns
    -------
    count : int
        The number of states
    """
    from deeptime.util.types import ensure_dtraj_list
    dtrajs = ensure_dtraj_list(dtrajs)
    if only_used:
        # only states with counts > 0 wanted. Make a bincount and count nonzeros
        bc = count_states(dtrajs)
        return np.count_nonzero(bc)
    else:
        # all states wanted, included nonpopulated ones. return max + 1
        return max(np.max(dtraj) for dtraj in dtrajs) + 1


def closed_sets(count_matrix, connectivity_threshold: float = 0):
    r"""
    todo
    Computes the strongly connected closed sets of C
    Parameters
    ----------
    count_matrix
    connectivity_threshold

    Returns
    -------

    """
    n = np.shape(count_matrix)[0]
    S = compute_connected_sets(count_matrix, connectivity_threshold=connectivity_threshold, directed=True)
    closed = []
    for s in S:
        mask = np.zeros(n, dtype=bool)
        mask[s] = True
        if count_matrix[np.ix_(mask, ~mask)].sum() == 0:
            # closed set, take it
            closed.append(s)
    return closed


def is_connected(count_matrix, connectivity_threshold: float = 0, directed=True):
    r"""
    todo
    Parameters
    ----------
    count_matrix
    connectivity_threshold
    directed

    Returns
    -------

    """
    sets = compute_connected_sets(count_matrix, connectivity_threshold=connectivity_threshold, directed=directed)
    return len(sets) == 1
