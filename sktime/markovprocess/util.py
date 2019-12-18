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


def compute_effective_stride(dtrajs, lagtime, nstates):
    # by default use lag as stride (=lag sampling), because we currently have no better theory for deciding
    # how many uncorrelated counts we can make
    stride = lagtime
    # get a quick fit from the spectral radius of the non-reversible
    from sktime.markovprocess import MaximumLikelihoodMSM
    msm_non_rev = MaximumLikelihoodMSM(lagtime=lagtime, reversible=False, sparse=False).fit(dtrajs).fetch_model()
    # if we have more than nstates timescales in our MSM, we use the next (neglected) timescale as an
    # fit of the de-correlation time
    if msm_non_rev.nstates > nstates:
        # because we use non-reversible msm, we want to silence the ImaginaryEigenvalueWarning
        import warnings
        from msmtools.util.exceptions import ImaginaryEigenValueWarning
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ImaginaryEigenValueWarning,
                                    module='msmtools.analysis.dense.decomposition')
            correlation_time = max(1, msm_non_rev.timescales()[nstates - 1])
        # use the smaller of these two pessimistic estimates
        stride = int(min(lagtime, 2 * correlation_time))

    return stride
