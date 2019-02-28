import numpy as np


def visited_set(dtrajs):
    r"""returns the set of states that have at least one count

    Parameters
    ----------
    dtraj : array_like or list of array_like
        Discretized trajectory or list of discretized trajectories

    Returns
    -------
    vis : ndarray((n), dtype=int)
        the set of states that have at least one count.
    """
    hist = count_states(dtrajs)
    return np.argwhere(hist > 0)[:, 0]


def count_states(dtrajs, ignore_negative=False):
    r"""returns a histogram count

    Parameters
    ----------
    dtrajs : array_like or list of array_like
        Discretized trajectory or list of discretized trajectories
    ignore_negative, bool, default=False
        Ignore negative elements. By default, a negative element will cause an
        exception

    Returns
    -------
    count : ndarray((n), dtype=int)
        the number of occurrences of each state. n=max+1 where max is the largest state index found.

    """
    # make bincounts for each input trajectory
    nmax = 0
    bcs = []
    for dtraj in dtrajs:
        if ignore_negative:
            dtraj = dtraj[np.where(dtraj >= 0)]
        bc = np.bincount(dtraj)
        nmax = max(nmax, bc.shape[0])
        bcs.append(bc)
    # construct total bincount
    res = np.zeros(nmax, dtype=int)
    # add up individual bincounts
    for i, bc in enumerate(bcs):
        res[:bc.shape[0]] += bc
    return res
