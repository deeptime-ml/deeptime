from typing import List

import numpy as np

from deeptime.util.types import ensure_dtraj_list


def compute_index_states(dtrajs, subset=None) -> List[np.ndarray]:
    """Generates a trajectory/time indices for the given list of states

    Parameters
    ----------
    dtrajs : array_like or list of array_like
        Discretized trajectory or list of discretized trajectories. Negative elements will be ignored
    subset : ndarray((n)), optional, default = None
        array of states to be indexed. By default all states in dtrajs will be used

    Returns
    -------
    indices : List[np.ndarray]
        For each state, all trajectory and time indices where this state occurs.
        Each matrix has a number of rows equal to the number of occurrences of the corresponding state,
        with rows consisting of a tuple (i, t), where i is the index of the trajectory and t is the time index
        within the trajectory.

    """
    # check input
    from .. import _markov_bindings as bd
    dtrajs = ensure_dtraj_list(dtrajs)
    return bd.sample.index_states(dtrajs, subset)


################################################################################
# sampling from state indices
################################################################################


def indices_by_sequence(indices: List[np.ndarray], sequence):
    r"""Samples trajectory/time indices according to the given sequence of states.

    Notes
    -----
    Returns -1 indices for states that are not observed in the current sample.

    Parameters
    ----------
    indices : List[np.ndarray]
        For each state, all trajectory and time indices where this state occurs.
        Each matrix has a number of rows equal to the number of occurrences of the corresponding state,
        with rows consisting of a tuple (i, t), where i is the index of the trajectory and t is the time index
        within the trajectory.
    sequence : array_like of integers
        A sequence of discrete states. For each state, a trajectory/time index will be sampled at which dtrajs
        have an occurrences of this state

    Returns
    -------
    indices : np.ndarray
        The sampled index sequence of shape `(N, 2)`.
        Index array with a number of rows equal to N=len(sequence), with rows consisting of a tuple (i, t),
        where i is the index of the trajectory and t is the time index within the trajectory.

    """
    N = len(sequence)
    res = np.zeros((N, 2), dtype=int)
    for t in range(N):
        s = sequence[t]
        ind = indices[s]
        res[t, :] = indices[s][np.random.randint(len(ind)), :] if len(ind) > 0 else -1

    return res


def indices_by_state(indices, nsample, subset=None, replace=True):
    """Samples trajectory/time indices according to the given sequence of states

    Parameters
    ----------
    indices : List[np.ndarray]
        For each state, all trajectory and time indices where this state occurs.
        Each matrix has a number of rows equal to the number of occurrences of the corresponding state,
        with rows consisting of a tuple (i, t), where i is the index of the trajectory and t is the time index
        within the trajectory.
    nsample : int
        Number of samples per state. If replace = False, the number of returned samples per state could be smaller
        if less than nsample indices are available for a state.
    subset : ndarray((n)), optional, default = None
        array of states to be indexed. By default all states in dtrajs will be used
    replace : boolean, optional
        Whether the sample is with or without replacement

    Returns
    -------
    indices : List[np.ndarray]
        List of the sampled indices by state, each state corresponding to an ndarray of shape (N, 2).
        Each element is an index array with a number of rows equal to N=len(sequence), with rows consisting of a
        tuple (i, t), where i is the index of the trajectory and t is the time index within the trajectory.

    """
    # how many states in total?
    n = len(indices)
    # define set of states to work on
    if subset is None:
        subset = np.arange(n)

    # list of states
    res = np.ndarray(len(subset), dtype=object)
    for i, s in enumerate(subset):
        # how many indices are available?
        m_available = indices[s].shape[0]
        # do we have no indices for this state? Then insert empty array.
        if m_available == 0:
            res[i] = np.zeros((0, 2), dtype=int)
        elif replace:
            I = np.random.choice(m_available, nsample, replace=True)
            res[i] = indices[s][I, :]
        else:
            I = np.random.choice(m_available, min(m_available, nsample), replace=False)
            res[i] = indices[s][I, :]

    return res


def indices_by_distribution(indices: List[np.ndarray], distributions, nsample):
    """Samples trajectory/time indices according to the given probability distributions

    Parameters
    ----------
    indices : list of ndarray( (N_i, 2) )
        For each state, all trajectory and time indices where this state occurs.
        Each matrix has a number of rows equal to the number of occurrences of the corresponding state,
        with rows consisting of a tuple (i, t), where i is the index of the trajectory and t is the time index
        within the trajectory.
    distributions : list or array of ndarray ( (n) )
        m distributions over states. Each distribution must be of length n and must sum up to 1.0
    nsample : int
        Number of samples per distribution. If replace = False, the number of returned samples per state could be smaller
        if less than nsample indices are available for a state.

    Returns
    -------
    indices : length m list of ndarray( (nsample, 2) )
        List of the sampled indices by distribution.
        Each element is an index array with a number of rows equal to nsample, with rows consisting of a
        tuple (i, t), where i is the index of the trajectory and t is the time index within the trajectory.

    """
    # how many states in total?
    n = len(indices)
    for dist in distributions:
        if len(dist) != n:
            raise ValueError('Size error: Distributions must all be of length n (number of states).')

    # list of states
    res = np.ndarray(len(distributions), dtype=object)
    for i, dist in enumerate(distributions):
        # sample states by distribution
        sequence = np.random.choice(n, size=nsample, p=dist)
        res[i] = indices_by_sequence(indices, sequence)
    return res


def by_state(dtrajs, n_samples, subset=None, replace=True):
    """Generates samples of the connected states.

    For each state in the active set of states, generates nsample samples with trajectory/time indices.

    Parameters
    ----------
    dtrajs : List[np.ndarray]
        underlying discrete trajectories
    n_samples : int
        Number of samples per state. If replace = False, the number of returned samples per state could be smaller
        if less than nsample indices are available for a state.
    subset : ndarray((n)), optional, default = None
        array of states to be indexed. By default all states in the connected set will be used
    replace : boolean, optional
        Whether the sample is with or without replacement

    Returns
    -------
    indices : list of ndarray( (N, 2) )
        list of trajectory/time index arrays with an array for each state.
        Within each index array, each row consist of a tuple (i, t), where i is
        the index of the trajectory and t is the time index within the trajectory.
    """
    # generate connected state indices
    indices = compute_index_states(dtrajs, subset=subset)
    return indices_by_state(indices, n_samples, subset=subset, replace=replace)
