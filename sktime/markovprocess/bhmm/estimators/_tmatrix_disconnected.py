
# This file is part of BHMM (Bayesian Hidden Markov Models).
#
# Copyright (c) 2016 Frank Noe (Freie Universitaet Berlin)
# and John D. Chodera (Memorial Sloan-Kettering Cancer Center, New York)
#
# BHMM is free software: you can redistribute it and/or modify
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

import numpy as np


def is_connected(C, mincount_connectivity=0, strong=True):
    S = connected_sets(C, mincount_connectivity=mincount_connectivity, strong=strong)
    return len(S) == 1


def connected_sets(C, mincount_connectivity=0, strong=True):
    """ Computes the connected sets of C.

    C : count matrix
    mincount_connectivity : float
        Minimum count which counts as a connection.
    strong : boolean
        True: Seek strongly connected sets. False: Seek weakly connected sets.

    """
    import msmtools.estimation as msmest
    Cconn = C.copy()
    Cconn[np.where(C <= mincount_connectivity)] = 0
    # treat each connected set separately
    S = msmest.connected_sets(Cconn, directed=strong)
    return S


def closed_sets(C, mincount_connectivity=0):
    """ Computes the strongly connected closed sets of C """
    n = np.shape(C)[0]
    S = connected_sets(C, mincount_connectivity=mincount_connectivity, strong=True)
    closed = []
    for s in S:
        mask = np.zeros(n, dtype=bool)
        mask[s] = True
        if C[np.ix_(mask, ~mask)].sum() == 0:  # closed set, take it
            closed.append(s)
    return closed


def nonempty_set(C, mincount_connectivity=0):
    """ Returns the set of states that have at least one incoming or outgoing count """
    # truncate to states with at least one observed incoming or outgoing count.
    if mincount_connectivity > 0:
        C = C.copy()
        C[np.where(C < mincount_connectivity)] = 0
    return np.where(C.sum(axis=0) + C.sum(axis=1) > 0)[0]


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
    import msmtools.estimation as msmest
    n = np.shape(C)[0]
    # output matrix. Set initially to Identity matrix in order to handle empty states
    P = np.eye(n, dtype=np.float64)
    # decide if we need to proceed by weakly or strongly connected sets
    if reversible and fixed_statdist is None:  # reversible to unknown eq. dist. - use strongly connected sets.
        S = connected_sets(C, mincount_connectivity=mincount_connectivity, strong=True)
        for s in S:
            mask = np.zeros(n, dtype=bool)
            mask[s] = True
            if C[np.ix_(mask, ~mask)].sum() > np.finfo(C.dtype).eps:  # outgoing transitions - use partial rev algo.
                transition_matrix_partial_rev(C, P, mask, maxiter=maxiter, maxerr=maxerr)
            else:  # closed set - use standard estimator
                I = np.ix_(mask, mask)
                if s.size > 1:  # leave diagonal 1 if single closed state.
                    P[I] = msmest.transition_matrix(C[I], reversible=True, warn_not_converged=False,
                                                    maxiter=maxiter, maxerr=maxerr)
    else:  # nonreversible or given equilibrium distribution - weakly connected sets
        S = connected_sets(C, mincount_connectivity=mincount_connectivity, strong=False)
        for s in S:
            I = np.ix_(s, s)
            if not reversible:
                Csub = C[I]
                # any zero rows? must set Cii = 1 to avoid dividing by zero
                zero_rows = np.where(Csub.sum(axis=1) == 0)[0]
                Csub[zero_rows, zero_rows] = 1.0
                P[I] = msmest.transition_matrix(Csub, reversible=False)
            elif reversible and fixed_statdist is not None:
                P[I] = msmest.transition_matrix(C[I], reversible=True, fixed_statdist=fixed_statdist,
                                                maxiter=maxiter, maxerr=maxerr)
            else:  # unknown case
                raise NotImplementedError('Transition estimation for the case reversible=' + str(reversible) +
                                          ' fixed_statdist=' + str(fixed_statdist is not None) + ' not implemented.')
    # done
    return P


def transition_matrix_partial_rev(C, P, S, maxiter=1000000, maxerr=1e-8):
    """Maximum likelihood estimation of transition matrix which is reversible on parts

    Partially-reversible estimation of transition matrix. Maximizes the likelihood:

    .. math:
        P_S &=& arg max prod_{S, :} (p_ij)^c_ij \\
        \Pi_S P_{S,S} &=& \Pi_S P_{S,S}

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
    import msmtools.analysis as msmana
    n = np.shape(P)[0]
    Prev = P.copy()
    # treat each weakly connected set separately
    sets = closed_sets(P)
    for s in sets:
        I = np.ix_(s, s)
        # compute stationary probability
        pi_s = msmana.stationary_distribution(P[I])
        # symmetrize
        X_s = pi_s[:, None] * P[I]
        X_s = 0.5 * (X_s + X_s.T)
        # normalize
        Prev[I] = X_s / X_s.sum(axis=1)[:, None]
    return Prev


def is_reversible(P):
    """ Returns if P is reversible on its weakly connected sets """
    import msmtools.analysis as msmana
    # treat each weakly connected set separately
    sets = connected_sets(P, strong=False)
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
    # can be replaced by msmtools.analysis.stationary_distribution in next msmtools release
    from msmtools.analysis.dense.stationary_vector import stationary_distribution as msmstatdist
    if C is None:
        if is_connected(P, strong=True):
            return msmstatdist(P)
        else:
            raise ValueError('Computing stationary distribution for disconnected matrix. Need count matrix.')

    # disconnected sets
    n = np.shape(C)[0]
    ctot = np.sum(C)
    pi = np.zeros(n)
    # treat each weakly connected set separately
    sets = connected_sets(C, mincount_connectivity=mincount_connectivity, strong=False)
    for s in sets:
        # compute weight
        w = np.sum(C[s, :]) / ctot
        pi[s] = w * msmstatdist(P[s, :][:, s])
    # reinforce normalization
    pi /= np.sum(pi)
    return pi


def rdl_decomposition(P, reversible=True):
    # TODO: this treatment is probably not meaningful for weakly connected matrices.
    import msmtools.estimation as msmest
    import msmtools.analysis as msmana
    # output matrices
    n = np.shape(P)[0]
    if reversible:
        dtype = np.float64
    else:
        dtype = complex
    R = np.zeros((n, n), dtype=dtype)
    D = np.zeros((n, n), dtype=dtype)
    L = np.zeros((n, n), dtype=dtype)
    # treat each strongly connected set separately
    S = msmest.connected_sets(P)
    for s in S:
        I = np.ix_(s, s)
        if len(s) > 1:
            if reversible:
                r, d, l = msmana.rdl_decomposition(P[s, :][:, s], norm='reversible')
                # everything must be real-valued - this should rather be handled by msmtools
                R[I] = r.real
                D[I] = d.real
                L[I] = l.real
            else:
                r, d, l = msmana.rdl_decomposition(P[s, :][:, s], norm='standard')
                # write to full
                R[I] = r
                D[I] = d
                L[I] = l
        else:  # just one element. Write 1's
            R[I] = 1
            D[I] = 1
            L[I] = 1
    # done
    return R, D, L
