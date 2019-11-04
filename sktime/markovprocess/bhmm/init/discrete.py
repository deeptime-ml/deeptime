
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

from bhmm.util.logger import logger
from bhmm.estimators import _tmatrix_disconnected


def coarse_grain_transition_matrix(P, M):
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


def regularize_hidden(p0, P, reversible=True, stationary=False, C=None, eps=None):
    """ Regularizes the hidden initial distribution and transition matrix.

    Makes sure that the hidden initial distribution and transition matrix have
    nonzero probabilities by setting them to eps and then renormalizing.
    Avoids zeros that would cause estimation algorithms to crash or get stuck
    in suboptimal states.

    Parameters
    ----------
    p0 : ndarray(n)
        Initial hidden distribution of the HMM
    P : ndarray(n, n)
        Hidden transition matrix
    reversible : bool
        HMM is reversible. Will make sure it is still reversible after modification.
    stationary : bool
        p0 is the stationary distribution of P. In this case, will not regularize
        p0 separately. If stationary=False, the regularization will be applied to p0.
    C : ndarray(n, n)
        Hidden count matrix. Only needed for stationary=True and P disconnected.
    epsilon : float or None
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
    n = P.shape[0]
    if eps is None:  # default output probability, in order to avoid zero columns
        eps = 0.01 / n

    # REGULARIZE P
    P = np.maximum(P, eps)
    # and renormalize
    P /= P.sum(axis=1)[:, None]
    # ensure reversibility
    if reversible:
        P = _tmatrix_disconnected.enforce_reversible_on_closed(P)

    # REGULARIZE p0
    if stationary:
        _tmatrix_disconnected.stationary_distribution(P, C=C)
    else:
        p0 = np.maximum(p0, eps)
        p0 /= p0.sum()

    return p0, P


def regularize_pobs(B, nonempty=None, separate=None, eps=None):
    """ Regularizes the output probabilities.

    Makes sure that the output probability distributions has
    nonzero probabilities by setting them to eps and then renormalizing.
    Avoids zeros that would cause estimation algorithms to crash or get stuck
    in suboptimal states.

    Parameters
    ----------
    B : ndarray(n, m)
        HMM output probabilities
    nonempty : None or iterable of int
        Nonempty set. Only regularize on this subset.
    separate : None or iterable of int
        Force the given set of observed states to stay in a separate hidden state.
        The remaining nstates-1 states will be assigned by a metastable decomposition.
    reversible : bool
        HMM is reversible. Will make sure it is still reversible after modification.

    Returns
    -------
    B : ndarray(n, m)
        Regularized output probabilities

    """
    # input
    B = B.copy()  # modify copy
    n, m = B.shape  # number of hidden / observable states
    if eps is None:  # default output probability, in order to avoid zero columns
        eps = 0.01 / m
    # observable sets
    if nonempty is None:
        nonempty = np.arange(m)

    if separate is None:
        B[:, nonempty] = np.maximum(B[:, nonempty], eps)
    else:
        nonempty_nonseparate = np.array(list(set(nonempty) - set(separate)), dtype=int)
        nonempty_separate = np.array(list(set(nonempty).intersection(set(separate))), dtype=int)
        B[:n-1, nonempty_nonseparate] = np.maximum(B[:n-1, nonempty_nonseparate], eps)
        B[n-1, nonempty_separate] = np.maximum(B[n-1, nonempty_separate], eps)

    # renormalize and return copy
    B /= B.sum(axis=1)[:, None]
    return B


def init_discrete_hmm_spectral(C_full, nstates, reversible=True, stationary=True, active_set=None, P=None,
                               eps_A=None, eps_B=None, separate=None):
    """Initializes discrete HMM using spectral clustering of observation counts

    Initializes HMM as described in [1]_. First estimates a Markov state model
    on the given observations, then uses PCCA+ to coarse-grain the transition
    matrix [2]_ which initializes the HMM transition matrix. The HMM output
    probabilities are given by Bayesian inversion from the PCCA+ memberships [1]_.

    The regularization parameters eps_A and eps_B are used
    to guarantee that the hidden transition matrix and output probability matrix
    have no zeros. HMM estimation algorithms such as the EM algorithm and the
    Bayesian sampling algorithm cannot recover from zero entries, i.e. once they
    are zero, they will stay zero.

    Parameters
    ----------
    C_full : ndarray(N, N)
        Transition count matrix on the full observable state space
    nstates : int
        The number of hidden states.
    reversible : bool
        Estimate reversible HMM transition matrix.
    stationary : bool
        p0 is the stationary distribution of P. In this case, will not
    active_set : ndarray(n, dtype=int) or None
        Index area. Will estimate kinetics only on the given subset of C
    P : ndarray(n, n)
        Transition matrix estimated from C (with option reversible). Use this
        option if P has already been estimated to avoid estimating it twice.
    eps_A : float or None
        Minimum transition probability. Default: 0.01 / nstates
    eps_B : float or None
        Minimum output probability. Default: 0.01 / nfull
    separate : None or iterable of int
        Force the given set of observed states to stay in a separate hidden state.
        The remaining nstates-1 states will be assigned by a metastable decomposition.

    Returns
    -------
    p0 : ndarray(n)
        Hidden state initial distribution
    A : ndarray(n, n)
        Hidden state transition matrix
    B : ndarray(n, N)
        Hidden-to-observable state output probabilities

    Raises
    ------
    ValueError
        If the given active set is illegal.
    NotImplementedError
        If the number of hidden states exceeds the number of observed states.

    Examples
    --------
    Generate initial model for a discrete output model.

    >>> import numpy as np
    >>> C = np.array([[0.5, 0.5, 0.0], [0.4, 0.5, 0.1], [0.0, 0.1, 0.9]])
    >>> initial_model = init_discrete_hmm_spectral(C, 2)

    References
    ----------
    .. [1] F. Noe, H. Wu, J.-H. Prinz and N. Plattner: Projected and hidden
        Markov models for calculating kinetics and  metastable states of
        complex molecules. J. Chem. Phys. 139, 184114 (2013)
    .. [2] S. Kube and M. Weber: A coarse graining method for the identification
        of transition rates between molecular conformations.
        J. Chem. Phys. 126, 024103 (2007)

    """
    # MICROSTATE COUNT MATRIX
    nfull = C_full.shape[0]

    # INPUTS
    if eps_A is None:  # default transition probability, in order to avoid zero columns
        eps_A = 0.01 / nstates
    if eps_B is None:  # default output probability, in order to avoid zero columns
        eps_B = 0.01 / nfull
    # Manage sets
    symsum = C_full.sum(axis=0) + C_full.sum(axis=1)
    nonempty = np.where(symsum > 0)[0]
    if active_set is None:
        active_set = nonempty
    else:
        if np.any(symsum[active_set] == 0):
            raise ValueError('Given active set has empty states')  # don't tolerate empty states
    if P is not None:
        if np.shape(P)[0] != active_set.size:  # needs to fit to active
            raise ValueError('Given initial transition matrix P has shape ' + str(np.shape(P))
                             + 'while active set has size ' + str(active_set.size))
    # when using separate states, only keep the nonempty ones (the others don't matter)
    if separate is None:
        active_nonseparate = active_set.copy()
        nmeta = nstates
    else:
        if np.max(separate) >= nfull:
            raise ValueError('Separate set has indexes that do not exist in full state space: '
                             + str(np.max(separate)))
        active_nonseparate = np.array(list(set(active_set) - set(separate)))
        nmeta = nstates - 1
    # check if we can proceed
    if active_nonseparate.size < nmeta:
        raise NotImplementedError('Trying to initialize ' + str(nmeta) + '-state HMM from smaller '
                                  + str(active_nonseparate.size) + '-state MSM.')

    # MICROSTATE TRANSITION MATRIX (MSM).
    C_active = C_full[np.ix_(active_set, active_set)]
    if P is None:  # This matrix may be disconnected and have transient states
        P_active = _tmatrix_disconnected.estimate_P(C_active, reversible=reversible, maxiter=10000)  # short iteration
    else:
        P_active = P

    # MICROSTATE EQUILIBRIUM DISTRIBUTION
    pi_active = _tmatrix_disconnected.stationary_distribution(P_active, C=C_active)
    pi_full = np.zeros(nfull)
    pi_full[active_set] = pi_active

    # NONSEPARATE TRANSITION MATRIX FOR PCCA+
    C_active_nonseparate = C_full[np.ix_(active_nonseparate, active_nonseparate)]
    if reversible and separate is None:  # in this case we already have a reversible estimate with the right size
        P_active_nonseparate = P_active
    else:  # not yet reversible. re-estimate
        P_active_nonseparate = _tmatrix_disconnected.estimate_P(C_active_nonseparate, reversible=True)

    # COARSE-GRAINING WITH PCCA+
    if active_nonseparate.size > nmeta:
        from msmtools.analysis.dense.pcca import PCCA
        pcca_obj = PCCA(P_active_nonseparate, nmeta)
        M_active_nonseparate = pcca_obj.memberships  # memberships
        B_active_nonseparate = pcca_obj.output_probabilities  # output probabilities
    else:  # equal size
        M_active_nonseparate = np.eye(nmeta)
        B_active_nonseparate = np.eye(nmeta)

    # ADD SEPARATE STATE IF NEEDED
    if separate is None:
        M_active = M_active_nonseparate
    else:
        M_full = np.zeros((nfull, nstates))
        M_full[active_nonseparate, :nmeta] = M_active_nonseparate
        M_full[separate, -1] = 1
        M_active = M_full[active_set]

    # COARSE-GRAINED TRANSITION MATRIX
    P_hmm = coarse_grain_transition_matrix(P_active, M_active)
    if reversible:
        P_hmm = _tmatrix_disconnected.enforce_reversible_on_closed(P_hmm)
    C_hmm = M_active.T.dot(C_active).dot(M_active)
    pi_hmm = _tmatrix_disconnected.stationary_distribution(P_hmm, C=C_hmm)  # need C_hmm in case if A is disconnected

    # COARSE-GRAINED OUTPUT DISTRIBUTION
    B_hmm = np.zeros((nstates, nfull))
    B_hmm[:nmeta, active_nonseparate] = B_active_nonseparate
    if separate is not None:  # add separate states
        B_hmm[-1, separate] = pi_full[separate]

    # REGULARIZE SOLUTION
    pi_hmm, P_hmm = regularize_hidden(pi_hmm, P_hmm, reversible=reversible, stationary=stationary, C=C_hmm, eps=eps_A)
    B_hmm = regularize_pobs(B_hmm, nonempty=nonempty, separate=separate, eps=eps_B)

    # print 'cg pi: ', pi_hmm
    # print 'cg A:\n ', P_hmm
    # print 'cg B:\n ', B_hmm

    logger().info('Initial model: ')
    logger().info('initial distribution = \n'+str(pi_hmm))
    logger().info('transition matrix = \n'+str(P_hmm))
    logger().info('output matrix = \n'+str(B_hmm.T))

    return pi_hmm, P_hmm, B_hmm


# Markers for future functions
def init_discrete_hmm_ml(C_full, nstates, reversible=True, stationary=True, active_set=None, P=None,
                         eps_A=None, eps_B=None, separate=None):
    """Initializes discrete HMM using maximum likelihood of observation counts"""
    raise NotImplementedError('ML-initialization not yet implemented')


def init_discrete_hmm_random(nhidden, nobs, lifetimes=None):
    """Initializes discrete HMM randomly"""
    raise NotImplementedError('Random initialization not yet implemented')
