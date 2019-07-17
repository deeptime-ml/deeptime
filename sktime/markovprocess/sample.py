import numpy as np

def _ensure_dtraj_list(dtrajs):
    r"""Makes sure that dtrajs is a list of discrete trajectories (array of int)

    """
    from sktime.util import ensure_ndarray

    return [ensure_ndarray(t, dtype=int) for t in dtrajs]


def _index_states(dtrajs, subset=None):
    """Generates a trajectory/time indexes for the given list of states

    Parameters
    ----------
    dtraj : array_like or list of array_like
        Discretized trajectory or list of discretized trajectories. Negative elements will be ignored
    subset : ndarray((n)), optional, default = None
        array of states to be indexed. By default all states in dtrajs will be used

    Returns
    -------
    indexes : list of ndarray( (N_i, 2) )
        For each state, all trajectory and time indexes where this state occurs.
        Each matrix has a number of rows equal to the number of occurrences of the corresponding state,
        with rows consisting of a tuple (i, t), where i is the index of the trajectory and t is the time index
        within the trajectory.

    """
    # check input
    dtrajs = _ensure_dtraj_list(dtrajs)
    # select subset unless given
    n_states = max(np.max(dtraj) for dtraj in dtrajs)
    if subset is None:
        subset = np.arange(n_states)
    else:
        if np.max(subset) >= n_states:
            raise ValueError('Selected subset is not a subset of the states in dtrajs.')
    # histogram states
    hist = count_states(dtrajs, ignore_negative=True)
    # efficient access to which state are accessible
    is_requested = np.ndarray((n_states), dtype=bool)
    is_requested[:] = False
    is_requested[subset] = True
    # efficient access to requested state indexes
    full2states = np.zeros((n_states), dtype=int)
    full2states[subset] = range(len(subset))
    # initialize results
    res    = np.ndarray(len(subset), dtype=object)
    counts = np.zeros((len(subset)), dtype=int)
    for i,s in enumerate(subset):
        res[i] = np.zeros((hist[s],2), dtype=int)
    # walk through trajectories and remember requested state indexes
    for i,dtraj in enumerate(dtrajs):
        for t,s in enumerate(dtraj):
            # only index nonnegative state indexes
            if s >= 0 and is_requested[s]:
                k = full2states[s]
                res[k][counts[k],0] = i
                res[k][counts[k],1] = t
                counts[k] += 1
    return res

################################################################################
# sampling from state indexes
################################################################################


def sample_indexes_by_sequence(indexes, sequence):
    """Samples trajectory/time indexes according to the given sequence of states

    Parameters
    ----------
    indexes : list of ndarray( (N_i, 2) )
        For each state, all trajectory and time indexes where this state occurs.
        Each matrix has a number of rows equal to the number of occurrences of the corresponding state,
        with rows consisting of a tuple (i, t), where i is the index of the trajectory and t is the time index
        within the trajectory.
    sequence : array of integers
        A sequence of discrete states. For each state, a trajectory/time index will be sampled at which dtrajs
        have an occurrences of this state

    Returns
    -------
    indexes : ndarray( (N, 2) )
        The sampled index sequence.
        Index array with a number of rows equal to N=len(sequence), with rows consisting of a tuple (i, t),
        where i is the index of the trajectory and t is the time index within the trajectory.

    """
    N = len(sequence)
    res = np.zeros((N,2), dtype=int)
    for t in range(N):
        s = sequence[t]
        i = np.random.randint(indexes[s].shape[0])
        res[t,:] = indexes[s][i,:]

    return res


def sample_indexes_by_state(indexes, nsample, subset=None, replace=True):
    """Samples trajectory/time indexes according to the given sequence of states

    Parameters
    ----------
    indexes : list of ndarray( (N_i, 2) )
        For each state, all trajectory and time indexes where this state occurs.
        Each matrix has a number of rows equal to the number of occurrences of the corresponding state,
        with rows consisting of a tuple (i, t), where i is the index of the trajectory and t is the time index
        within the trajectory.
    nsample : int
        Number of samples per state. If replace = False, the number of returned samples per state could be smaller
        if less than nsample indexes are available for a state.
    subset : ndarray((n)), optional, default = None
        array of states to be indexed. By default all states in dtrajs will be used
    replace : boolean, optional
        Whether the sample is with or without replacement

    Returns
    -------
    indexes : list of ndarray( (N, 2) )
        List of the sampled indices by state.
        Each element is an index array with a number of rows equal to N=len(sequence), with rows consisting of a
        tuple (i, t), where i is the index of the trajectory and t is the time index within the trajectory.

    """
    # how many states in total?
    n = len(indexes)
    # define set of states to work on
    if subset is None:
        subset = np.arange(n)

    # list of states
    res = np.ndarray(len(subset), dtype=object)
    for i, s in enumerate(subset):
        # how many indexes are available?
        m_available = indexes[s].shape[0]
        # do we have no indexes for this state? Then insert empty array.
        if m_available == 0:
            res[i] = np.zeros((0, 2), dtype=int)
        elif replace:
            I = np.random.choice(m_available, nsample, replace=True)
            res[i] = indexes[s][I,:]
        else:
            I = np.random.choice(m_available, min(m_available,nsample), replace=False)
            res[i] = indexes[s][I,:]

    return res


def sample_indexes_by_distribution(indexes, distributions, nsample):
    """Samples trajectory/time indexes according to the given probability distributions

    Parameters
    ----------
    indexes : list of ndarray( (N_i, 2) )
        For each state, all trajectory and time indexes where this state occurs.
        Each matrix has a number of rows equal to the number of occurrences of the corresponding state,
        with rows consisting of a tuple (i, t), where i is the index of the trajectory and t is the time index
        within the trajectory.
    distributions : list or array of ndarray ( (n) )
        m distributions over states. Each distribution must be of length n and must sum up to 1.0
    nsample : int
        Number of samples per distribution. If replace = False, the number of returned samples per state could be smaller
        if less than nsample indexes are available for a state.

    Returns
    -------
    indexes : length m list of ndarray( (nsample, 2) )
        List of the sampled indices by distribution.
        Each element is an index array with a number of rows equal to nsample, with rows consisting of a
        tuple (i, t), where i is the index of the trajectory and t is the time index within the trajectory.

    """
    # how many states in total?
    n = len(indexes)
    for dist in distributions:
        if len(dist) != n:
            raise ValueError('Size error: Distributions must all be of length n (number of states).')

    # list of states
    res = np.ndarray(len(distributions), dtype=object)
    for i, dist in enumerate(distributions):
        # sample states by distribution
        sequence = np.random.choice(n, size=nsample, p=dist)
        res[i] = sample_indexes_by_sequence(indexes, sequence)
    #
    return res



def sample_by_sequence(dtrajs, sequence, N, start=None, stop=None, stride=1):
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
    indexes : ndarray( (N, 2) )
        trajectory and time indexes of the simulated trajectory. Each row consist of a tuple (i, t), where i is
        the index of the trajectory and t is the time index within the trajectory.
        Note that the time different between two samples is the Markov model lag time tau

    See also
    --------
    pyemma.coordinates.save_traj
        in order to save this synthetic trajectory as a trajectory file with molecular structures

    """
    return sample_indexes_by_sequence(self.active_state_indexes, sequence)


def sample_by_state(dtrajs, nsample, subset=None, replace=True):
    """Generates samples of the connected states.

    For each state in the active set of states, generates nsample samples with trajectory/time indexes.
    This information can be used in order to generate a trajectory of length nsample * nconnected using
    :func:`pyemma.coordinates.save_traj` or nconnected trajectories of length nsample each using
    :func:`pyemma.coordinates.save_traj`

    Parameters
    ----------
    dtrajs : List[np.ndarray]
        underlying discrete trajectories
    nsample : int
        Number of samples per state. If replace = False, the number of returned samples per state could be smaller
        if less than nsample indexes are available for a state.
    subset : ndarray((n)), optional, default = None
        array of states to be indexed. By default all states in the connected set will be used
    replace : boolean, optional
        Whether the sample is with or without replacement

    Returns
    -------
    indexes : list of ndarray( (N, 2) )
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
    # generate connected state indexes
    return sample_indexes_by_state(self.active_state_indexes, nsample, subset=subset, replace=replace)


# TODO: add sample_metastable() for sampling from metastable (pcca or hmm) states.
def sample_by_distributions(self, distributions, nsample):
    """Generates samples according to given probability distributions

    Parameters
    ----------
    dtrajs : List[np.ndarray]
        underlying discrete trajectories
    distributions : list or array of ndarray ( (n) )
        m distributions over states. Each distribution must be of length n and must sum up to 1.0
    nsample : int
        Number of samples per distribution. If replace = False, the number of returned samples per state could be
        smaller if less than nsample indexes are available for a state.

    Returns
    -------
    indexes : length m list of ndarray( (nsample, 2) )
        List of the sampled indices by distribution.
        Each element is an index array with a number of rows equal to nsample, with rows consisting of a
        tuple (i, t), where i is the index of the trajectory and t is the time index within the trajectory.

    """
    # generate connected state indexes
    return sample_indexes_by_distribution(self.active_state_indexes, distributions, nsample)
