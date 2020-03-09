import typing

import numpy as np

from sktime.util import ensure_dtraj_list


def compute_index_states(dtrajs, subset=None) -> typing.List[np.ndarray]:
    """Generates a trajectory/time indices for the given list of states

    Parameters
    ----------
    dtraj : array_like or list of array_like
        Discretized trajectory or list of discretized trajectories. Negative elements will be ignored
    subset : ndarray((n)), optional, default = None
        array of states to be indexed. By default all states in dtrajs will be used

    Returns
    -------
    indices : list of ndarray( (N_i, 2) )
        For each state, all trajectory and time indices where this state occurs.
        Each matrix has a number of rows equal to the number of occurrences of the corresponding state,
        with rows consisting of a tuple (i, t), where i is the index of the trajectory and t is the time index
        within the trajectory.

    """
    # check input
    from . import _markov_bindings as bd
    dtrajs = ensure_dtraj_list(dtrajs)
    return bd.sample.index_states(dtrajs, subset)

################################################################################
# sampling from state indices
################################################################################


def indices_by_sequence(indices: typing.List[np.ndarray], sequence):
    """Samples trajectory/time indices according to the given sequence of states

    Parameters
    ----------
    indices : list of (N,2) ndarray
        For each state, all trajectory and time indices where this state occurs.
        Each matrix has a number of rows equal to the number of occurrences of the corresponding state,
        with rows consisting of a tuple (i, t), where i is the index of the trajectory and t is the time index
        within the trajectory.
    sequence : array of integers
        A sequence of discrete states. For each state, a trajectory/time index will be sampled at which dtrajs
        have an occurrences of this state

    Returns
    -------
    indices : ndarray( (N, 2) )
        The sampled index sequence.
        Index array with a number of rows equal to N=len(sequence), with rows consisting of a tuple (i, t),
        where i is the index of the trajectory and t is the time index within the trajectory.

    """
    N = len(sequence)
    res = np.zeros((N,2), dtype=int)
    for t in range(N):
        s = sequence[t]
        i = np.random.randint(indices[s].shape[0])
        res[t,:] = indices[s][i, :]

    return res


def indices_by_state(indices, nsample, subset=None, replace=True):
    """Samples trajectory/time indices according to the given sequence of states

    Parameters
    ----------
    indices : list of ndarray( (N_i, 2) )
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
    indices : list of ndarray( (N, 2) )
        List of the sampled indices by state.
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
            res[i] = indices[s][I,:]
        else:
            I = np.random.choice(m_available, min(m_available,nsample), replace=False)
            res[i] = indices[s][I,:]

    return res


def indices_by_distribution(indices: typing.List[np.ndarray], distributions, nsample):
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
    #
    return res



def by_sequence(dtrajs, sequence, N, start=None, stop=None, stride=1):
    """Generates a synthetic discrete trajectory of length N and simulation time stride * lag time * N

    This information can be used
    in order to generate a synthetic molecular dynamics trajectory - see
    :func:`pyemma.coordinates.save_traj`

    Note that the time different between two samples is the Markov model lag time tau. When comparing
    quantities computing from this synthetic trajectory and from the input trajectories, the time points of this
    trajectory must be scaled by the lag time in order to have them on the same time scale.

    Parameters
    ----------
    dtrajs : List[np.ndarray]
        underlying discrete trajectories
    sequence : np.ndarray
        sequence of states that are being remapped to samples from dtrajs
    N : int
        Number of time steps in the output trajectory. The total simulation time is stride * lag time * N
    start : int, optional, default = None
        starting state. If not given, will sample from the stationary distribution of P
    stop : int or int-array-like, optional, default = None
        stopping set. If given, the trajectory will be stopped before N steps
        once a state of the stop set is reached
    stride : int, optional, default = 1
        Multiple of lag time used as a time step. By default, the time step is equal to the lag time

    Returns
    -------
    indices : ndarray( (N, 2) )
        trajectory and time indices of the simulated trajectory. Each row consist of a tuple (i, t), where i is
        the index of the trajectory and t is the time index within the trajectory.
        Note that the time different between two samples is the Markov model lag time tau

    See also
    --------
    pyemma.coordinates.save_traj
        in order to save this synthetic trajectory as a trajectory file with molecular structures

    """
    return indices_by_sequence(self.active_state_indices, sequence)


def by_state(dtrajs, nsample, subset=None, replace=True):
    """Generates samples of the connected states.

    For each state in the active set of states, generates nsample samples with trajectory/time indices.
    This information can be used in order to generate a trajectory of length nsample * nconnected using
    :func:`pyemma.coordinates.save_traj` or nconnected trajectories of length nsample each using
    :func:`pyemma.coordinates.save_traj`

    Parameters
    ----------
    dtrajs : List[np.ndarray]
        underlying discrete trajectories
    nsample : int
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

    See also
    --------
    pyemma.coordinates.save_traj
        in order to save the sampled frames sequentially in a trajectory file with molecular structures
    pyemma.coordinates.save_trajs
        in order to save the sampled frames in nconnected trajectory files with molecular structures

    """
    # generate connected state indices
    return indices_by_state(self.active_state_indices, nsample, subset=subset, replace=replace)


# TODO: add sample_metastable() for sampling from metastable (pcca or hmm) states.
def by_distributions(self, distributions, nsample):
    """Generates samples according to given probability distributions

    Parameters
    ----------
    dtrajs : List[np.ndarray]
        underlying discrete trajectories
    distributions : list or array of ndarray ( (n) )
        m distributions over states. Each distribution must be of length n and must sum up to 1.0
    nsample : int
        Number of samples per distribution. If replace = False, the number of returned samples per state could be
        smaller if less than nsample indices are available for a state.

    Returns
    -------
    indices : length m list of ndarray( (nsample, 2) )
        List of the sampled indices by distribution.
        Each element is an index array with a number of rows equal to nsample, with rows consisting of a
        tuple (i, t), where i is the index of the trajectory and t is the time index within the trajectory.

    """
    # generate connected state indices
    return indices_by_distribution(self.active_state_indices, distributions, nsample)
