import numpy as np

from ._util import compute_connected_sets, closed_sets, is_connected


def estimate_P(C, reversible=True, fixed_statdist=None, maxiter=1000000, maxerr=1e-8, mincount_connectivity=0):
    """ Estimates full transition matrix for general connectivity structure

    Parameters
    ----------
    C : ndarray
        count matrix
    reversible : bool
        estimate reversible?
    fixed_statdist : ndarray or None
        estimate with given stationary distribution
    maxiter : int
        Maximum number of reversible iterations.
    maxerr : float
        Stopping criterion for reversible iteration: Will stop when infinity
        norm  of difference vector of two subsequent equilibrium distributions
        is below maxerr.
    mincount_connectivity : float
        Minimum count which counts as a connection.

    """
    import deeptime.markov.tools.estimation as msmest
    n = np.shape(C)[0]
    # output matrix. Set initially to Identity matrix in order to handle empty states
    P = np.eye(n, dtype=np.float64)
    # decide if we need to proceed by weakly or strongly connected sets
    if reversible and fixed_statdist is None:  # reversible to unknown eq. dist. - use strongly connected sets.
        S = compute_connected_sets(C, connectivity_threshold=mincount_connectivity, directed=True)
        for s in S:
            mask = np.zeros(n, dtype=bool)
            mask[s] = True
            if C[np.ix_(mask, ~mask)].sum() > np.finfo(C.dtype).eps:  # outgoing transitions - use partial rev algo.
                transition_matrix_partial_rev(C, P, mask, maxiter=maxiter, maxerr=maxerr)
            else:  # closed set - use standard estimator
                indices = np.ix_(mask, mask)
                if s.size > 1:  # leave diagonal 1 if single closed state.
                    P[indices] = msmest.transition_matrix(C[indices], reversible=True, warn_not_converged=False,
                                                    maxiter=maxiter, maxerr=maxerr)
    else:  # nonreversible or given equilibrium distribution - weakly connected sets
        S = compute_connected_sets(C, connectivity_threshold=mincount_connectivity, directed=False)
        for s in S:
            indices = np.ix_(s, s)
            if not reversible:
                Csub = C[indices]
                # any zero rows? must set Cii = 1 to avoid dividing by zero
                zero_rows = np.where(Csub.sum(axis=1) == 0)[0]
                Csub[zero_rows, zero_rows] = 1.0
                P[indices] = msmest.transition_matrix(Csub, reversible=False)
            elif reversible and fixed_statdist is not None:
                P[indices] = msmest.transition_matrix(C[indices], reversible=True, fixed_statdist=fixed_statdist,
                                                maxiter=maxiter, maxerr=maxerr)
            else:  # unknown case
                raise NotImplementedError('Transition estimation for the case reversible=' + str(reversible) +
                                          ' fixed_statdist=' + str(fixed_statdist is not None) + ' not implemented.')
    # done
    return P


def transition_matrix_partial_rev(C, P, S, maxiter=1000000, maxerr=1e-8):
    r"""Maximum likelihood estimation of transition matrix which is reversible on parts

    Partially-reversible estimation of transition matrix. Maximizes the likelihood:

    .. math:
        P_S &=& arg max prod_{S, :} (p_ij)^c_ij \\
        \\Pi_S P_{S,S} &=& \\Pi_S P_{S,S}

    where the product runs over all elements of the rows S, and detailed balance only
    acts on the block with rows and columns S. :math:`\Pi_S` is the diagonal matrix of
    equilibrium probabilities restricted to set S.

    Note that this formulation

    Parameters
    ----------
    C : ndarray
        full count matrix
    P : ndarray
        full transition matrix to write to. Will overwrite P[S]
    S : ndarray, bool
        boolean selection of reversible set with outgoing transitions
    maxiter : int, optional, default = 1000000
        Maximum number of iterations, iteration termination condition.
    maxerr : float
        maximum difference in matrix sums between iterations (infinity norm) in order to stop.
    """
    # test input
    assert np.array_equal(C.shape, P.shape)
    # constants
    A = C[S][:, S]
    B = C[S][:, ~S]
    ATA = A + A.T
    countsums = C[S].sum(axis=1)
    # initialize
    X = 0.5 * ATA
    Y = C[S][:, ~S]
    # normalize X, Y
    totalsum = X.sum() + Y.sum()
    X /= totalsum
    Y /= totalsum
    # rowsums
    rowsums = X.sum(axis=1) + Y.sum(axis=1)
    err = 1.0
    it = 0
    while err > maxerr and it < maxiter:
        # update
        d = countsums / rowsums
        X = ATA / (d[:, None] + d)
        Y = B / d[:, None]
        # normalize X, Y
        totalsum = X.sum() + Y.sum()
        X /= totalsum
        Y /= totalsum
        # update sums
        rowsums_new = X.sum(axis=1) + Y.sum(axis=1)
        # compute error
        err = np.max(np.abs(rowsums_new - rowsums))
        # update
        rowsums = rowsums_new
        it += 1
    # write to P
    P[np.ix_(S, S)] = X
    P[np.ix_(S, ~S)] = Y
    P[S] /= P[S].sum(axis=1)[:, None]


def enforce_reversible_on_closed(P):
    """ Enforces transition matrix P to be reversible on its closed sets. """
    import deeptime.markov.tools.analysis as msmana
    Prev = P.copy()
    # treat each weakly connected set separately
    sets = closed_sets(P)
    for s in sets:
        indices = np.ix_(s, s)
        # compute stationary probability
        pi_s = msmana.stationary_distribution(P[indices])
        # symmetrize
        X_s = pi_s[:, None] * P[indices]
        X_s = 0.5 * (X_s + X_s.T)
        # normalize
        Prev[indices] = X_s / X_s.sum(axis=1)[:, None]
    return Prev


def is_reversible(P):
    """ Returns if P is reversible on its weakly connected sets """
    import deeptime.markov.tools.analysis as msmana
    # treat each weakly connected set separately
    sets = compute_connected_sets(P, directed=False)
    for s in sets:
        Ps = P[s, :][:, s]
        if not msmana.is_transition_matrix(Ps):
            return False  # isn't even a transition matrix!
        pi = msmana.stationary_distribution(Ps)
        X = pi[:, None] * Ps
        if not np.allclose(X, X.T):
            return False
    # survived.
    return True


def stationary_distribution(P, C=None, mincount_connectivity=0):
    """ Simple estimator for stationary distribution for multiple strongly connected sets """
    # can be replaced by deeptime.markov.tools.analysis.stationary_distribution in next msmtools release
    from deeptime.markov.tools.analysis import stationary_distribution as msmstatdist
    if C is None:
        if is_connected(P, directed=True):
            return msmstatdist(P, check_inputs=False)
        else:
            raise ValueError('Computing stationary distribution for disconnected matrix. Need count matrix.')

    # disconnected sets
    n = np.shape(C)[0]
    ctot = np.sum(C)
    pi = np.zeros(n)
    # treat each weakly connected set separately
    sets = compute_connected_sets(C, connectivity_threshold=mincount_connectivity, directed=False)
    for s in sets:
        # compute weight
        w = np.sum(C[s, :]) / ctot
        pi[s] = w * msmstatdist(P[s, :][:, s], check_inputs=False)
    # reinforce normalization
    pi /= np.sum(pi)
    return pi
