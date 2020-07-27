
# -*- coding: utf-8 -*-

# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# MSMTools is free software: you can redistribute it and/or modify
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

r"""
=========================
 Estimation API
=========================

"""

__docformat__ = "restructuredtext en"

import warnings

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from scipy.sparse.sputils import isdense

from . import dense
from . import sparse
from ..dtraj.api import count_states as _count_states
from ..dtraj.api import number_of_states as _number_of_states
from ..util.annotators import shortcut
from ..util.types import ensure_dtraj_list as _ensure_dtraj_list

__author__ = "Benjamin Trendelkamp-Schroer, Martin Scherer, Frank Noe"
__copyright__ = "Copyright 2014, Computational Molecular Biology Group, FU-Berlin"
__credits__ = ["Benjamin Trendelkamp-Schroer", "Martin Scherer", "Fabian Paul", "Frank Noe"]

__version__ = "2.0.0"
__maintainer__ = "Martin Scherer"
__email__ = "m.scherer AT fu-berlin DOT de"

__all__ = ['bootstrap_trajectories',
           'bootstrap_counts',
           'count_matrix',
           'count_states',
           'connected_sets',
           'effective_count_matrix',
           'error_perturbation',
           'is_connected',
           'largest_connected_set',
           'largest_connected_submatrix',
           'log_likelihood',
           'number_of_states',
           'prior_const',
           'prior_neighbor',
           'prior_rev',
           'rate_matrix',
           'sample_tmatrix',
           'tmatrix_cov',
           'tmatrix_sampler',
           'transition_matrix'
           ]

# append shortcuts separately in order to avoid complaints by syntax checker
__all__.append('histogram')
__all__.append('nstates')
__all__.append('cmatrix')
__all__.append('effective_cmatrix')
__all__.append('connected_cmatrix')
__all__.append('tmatrix')

################################################################################
# Basic counting
################################################################################


@shortcut('histogram')
def count_states(dtrajs):
    r"""returns a histogram count

    Parameters
    ----------
    dtraj : array_like or list of array_like
        Discretized trajectory or list of discretized trajectories

    Returns
    -------
    count : ndarray((n), dtype=int)
        the number of occurrances of each state. n=max+1 where max is the largest state index found.
    """
    return _count_states(dtrajs)


@shortcut('nstates')
def number_of_states(dtrajs, only_used=False):
    r"""returns the number of states in the given trajectories.

    Parameters
    ----------
    dtraj : array_like or list of array_like
        Discretized trajectory or list of discretized trajectories
    only_used = False : boolean
        If False, will return max+1, where max is the largest index used.
        If True, will return the number of states that occur at least once.
    """
    return _number_of_states(dtrajs, only_used=only_used)


################################################################################
# Count matrix
################################################################################

@shortcut('cmatrix')
def count_matrix(dtraj, lag, sliding=True, sparse_return=True, nstates=None):
    r"""Generate a count matrix from given microstate trajectory.

    Parameters
    ----------
    dtraj : array_like or list of array_like
        Discretized trajectory or list of discretized trajectories
    lag : int
        Lagtime in trajectory steps
    sliding : bool, optional
        If true the sliding window approach
        is used for transition counting.
    sparse_return : bool (optional)
        Whether to return a dense or a sparse matrix.
    nstates : int, optional
        Enforce a count-matrix with shape=(nstates, nstates)

    Returns
    -------
    C : scipy.sparse.coo_matrix
        The count matrix at given lag in coordinate list format.

    Notes
    -----
    Transition counts can be obtained from microstate trajectory using
    two methods. Couning at lag and slidingwindow counting.

    **Lag**

    This approach will skip all points in the trajectory that are
    seperated form the last point by less than the given lagtime
    :math:`\tau`.

    Transition counts :math:`c_{ij}(\tau)` are generated according to

    .. math:: c_{ij}(\tau) = \sum_{k=0}^{\left \lfloor \frac{N}{\tau} \right \rfloor -2}
                                        \chi_{i}(X_{k\tau})\chi_{j}(X_{(k+1)\tau}).

    :math:`\chi_{i}(x)` is the indicator function of :math:`i`, i.e
    :math:`\chi_{i}(x)=1` for :math:`x=i` and :math:`\chi_{i}(x)=0` for
    :math:`x \neq i`.

    **Sliding**

    The sliding approach slides along the trajectory and counts all
    transitions sperated by the lagtime :math:`\tau`.

    Transition counts :math:`c_{ij}(\tau)` are generated according to

    .. math:: c_{ij}(\tau)=\sum_{k=0}^{N-\tau-1} \chi_{i}(X_{k}) \chi_{j}(X_{k+\tau}).

    References
    ----------
    .. [1] Prinz, J H, H Wu, M Sarich, B Keller, M Senne, M Held, J D
        Chodera, C Schuette and F Noe. 2011. Markov models of
        molecular kinetics: Generation and validation. J Chem Phys
        134: 174105

    Examples
    --------

    >>> import numpy as np
    >>> from msmtools.estimation import count_matrix

    >>> dtraj = np.array([0, 0, 1, 0, 1, 1, 0])
    >>> tau = 2

    Use the sliding approach first

    >>> C_sliding = count_matrix(dtraj, tau)

    The generated matrix is a sparse matrix in CSR-format. For
    convenient printing we convert it to a dense ndarray.

    >>> C_sliding.toarray()
    array([[ 1.,  2.],
           [ 1.,  1.]])

    Let us compare to the count-matrix we obtain using the lag
    approach

    >>> C_lag = count_matrix(dtraj, tau, sliding=False)
    >>> C_lag.toarray()
    array([[ 0.,  1.],
           [ 1.,  1.]])

    """
    # convert dtraj input, if it contains out of nested python lists to
    # a list of int ndarrays.
    dtraj = _ensure_dtraj_list(dtraj)
    return sparse.count_matrix.count_matrix_coo2_mult(dtraj, lag, sliding=sliding,
                                                      sparse=sparse_return, nstates=nstates)


@shortcut('effective_cmatrix')
def effective_count_matrix(dtrajs, lag, average='row', mact=1.0, n_jobs=1, callback=None):
    r""" Computes the statistically effective transition count matrix

    Given a list of discrete trajectories, compute the effective number of statistically uncorrelated transition
    counts at the given lag time. First computes the full sliding-window counts :math:`c_{ij}(tau)`. Then uses
    :func:`statistical_inefficiencies` to compute statistical inefficiencies :math:`I_{ij}(tau)`. The number of
    effective counts in a row is then computed as

    .. math:
        c_i^{\mathrm{eff}}(tau) = \sum_j I_{ij}(tau) c_{ij}(tau)

    and the effective transition counts are obtained by scaling the rows accordingly:

    .. math:
        c_{ij}^{\mathrm{eff}}(tau) = \frac{c_i^{\mathrm{eff}}(tau)}{c_i(tau)} c_{ij}(tau)

    This procedure is not yet published, but a manuscript is in preparation [1]_.

    Parameters
    ----------
    dtrajs : list of int-iterables
        discrete trajectories
    lag : int
        lag time
    average : str, default='row'
        Use either of 'row', 'all', 'none', with the following consequences:
        'none': the statistical inefficiency is applied separately to each
            transition count (not recommended)
        'row': the statistical inefficiency is averaged (weighted) by row
            (recommended).
        'all': the statistical inefficiency is averaged (weighted) over all
            transition counts (not recommended).
    mact : float, default=1.0
        multiplier for the autocorrelation time. We tend to underestimate the
        autocorrelation time (and thus overestimate effective counts)
        because the autocorrelation function is truncated when it passes
        through 0 in order to avoid numerical instabilities.
        This is a purely heuristic factor trying to compensate this effect.
        This parameter might be removed in the future when a more robust
        estimation method of the autocorrelation time is used.
    n_jobs : int, default=1
        If greater one, the function will be evaluated with multiple processes.
    callback : callable, default=None
        will be called for every statistical inefficiency computed (number of nonzero elements in count matrix).
        If n_jobs is greater one, the callback will be invoked per finished batch.

    See also
    --------
    statistical_inefficiencies
        is used for computing the statistical inefficiencies of sliding window transition counts

    References
    ----------
    .. [1] Noe, F. and H. Wu: in preparation (2015)

    """

    dtrajs = _ensure_dtraj_list(dtrajs)
    import os
    # enforce one job on windows.
    if os.name == 'nt':
        n_jobs = 1
    return sparse.effective_counts.effective_count_matrix(dtrajs, lag, average=average, mact=mact, n_jobs=n_jobs, callback=callback)


################################################################################
# Bootstrapping data
################################################################################

def bootstrap_trajectories(trajs, correlation_length):
    r"""Generates a randomly resampled trajectory segments.

    Parameters
    ----------
    trajs : array-like or array-like of array-like
        single or multiple trajectories. Every trajectory is assumed to be
        a statistically independent realization. Note that this is often not true and
        is a weakness with the present bootstrapping approach.

    correlation_length : int
        Correlation length (also known as the or statistical inefficiency) of the data.
        If set to < 1 or > L, where L is the longest trajectory length, the
        bootstrapping will sample full trajectories.
        We suggest to select the largest implied timescale or relaxation timescale as a
        conservative estimate of the correlation length. If this timescale is unknown,
        it's suggested to use full trajectories (set timescale to < 1) or come up with
        a rough estimate. For computing the error on specific observables, one may use
        shorter timescales, because the relevant correlation length is the integral of
        the autocorrelation function of the observables of interest [3]. The slowest
        implied timescale is an upper bound for that correlation length, and therefore
        a conservative estimate [4].

    Notes
    -----
    This function can be called multiple times in order to generate randomly
    resampled trajectory data. In order to compute error bars on your observable
    of interest, call this function to generate resampled trajectories, and
    put them into your estimator. The standard deviation of such a sample of
    the observable is a model for the standard error.

    Implements a moving block bootstrapping procedure [1] for generation of
    randomly resampled count matrixes from discrete trajectories. The corrlation length
    determines the size of trajectory blocks that will remain contiguous.
    For a single trajectory N with correlation length t_corr < N,
    we will sample floor(N/t_corr) subtrajectories of length t_corr using starting time t.
    t is a uniform random number in [0, N-t_corr-1].
    When multiple trajectories are available, N is the total number of timesteps
    over all trajectories, the algorithm will generate resampled data with a total number
    of N (or slightly larger) time steps. Each trajectory of length n_i has a probability
    of n_i to be selected. Trajectories of length n_i <= t_corr are returned completely.
    For longer trajectories, segments of length t_corr are randomly generated.

    Note that like all error models for correlated time series data, Bootstrapping
    just gives you a model for the error given a number of assumptions [2]. The most
    critical decisions are: (1) is this approach meaningful at all (only if the
    trajectories are statistically independent realizations), and (2) select
    an appropriate timescale of the correlation length (see below).
    Note that transition matrix sampling from the Dirichlet distribution is a
    much better option from a theoretical point of view, but may also be
    computationally more demanding.

    References
    ----------
    .. [1] H. R. Kuensch. The jackknife and the bootstrap for general
        stationary observations, Ann. Stat. 3, 1217-41 (1989).
    .. [2] B. Efron. Bootstrap methods: Another look at the jackknife.
        Ann. Statist. 7 1-26 (1979).
    .. [3] T.W. Anderson. The Statistical Analysis of Time Series
        Wiley, New York (1971).
    .. [4] F. Noe and F. Nueske: A variational approach to modeling
        slow processes in stochastic dynamical systems.  SIAM
        Multiscale Model. Simul., 11 . pp. 635-655 (2013)

    """
    return dense.bootstrapping.bootstrap_trajectories(trajs, correlation_length)


def bootstrap_counts(dtrajs, lagtime, corrlength=None):
    r"""Generates a randomly resampled count matrix given the input coordinates.

    Parameters
    ----------
    dtrajs : array-like or array-like of array-like
        single or multiple discrete trajectories. Every trajectory is assumed to be
        a statistically independent realization. Note that this is often not true and
        is a weakness with the present bootstrapping approach.

    lagtime : int
        the lag time at which the count matrix will be evaluated

    corrlength : int, optional, default=None
        the correlation length of the discrete trajectory. N / corrlength counts will be generated,
        where N is the total number of frames. If set to None (default), corrlength = lagtime will be used.

    Notes
    -----
    This function can be called multiple times in order to generate randomly
    resampled realizations of count matrices. For each of these realizations
    you can estimate a transition matrix, and from each of them computing the
    observables of your interest. The standard deviation of such a sample of
    the observable is a model for the standard error.

    The bootstrap will be generated by sampling N/corrlength counts at time tuples (t, t+lagtime),
    where t is uniformly sampled over all trajectory time frames in [0,n_i-lagtime].
    Here, n_i is the length of trajectory i and N = sum_i n_i is the total number of frames.

    See also
    --------
    bootstrap_trajectories

    """
    dtrajs = _ensure_dtraj_list(dtrajs)
    return dense.bootstrapping.bootstrap_counts(dtrajs, lagtime, corrlength=corrlength)


################################################################################
# Connectivity
################################################################################

def connected_sets(C, directed=True):
    r"""Compute connected sets of microstates.

    Connected components for a directed graph with edge-weights
    given by the count matrix.

    Parameters
    ----------
    C : scipy.sparse matrix
        Count matrix specifying edge weights.
    directed : bool, optional
       Whether to compute connected components for a directed  or
       undirected graph. Default is True.

    Returns
    -------
    cc : list of arrays of integers
        Each entry is an array containing all vertices (states) in the
        corresponding connected component. The list is sorted
        according to the size of the individual components. The
        largest connected set is the first entry in the list, lcc=cc[0].

    Notes
    -----
    Viewing the count matrix as the adjacency matrix of a (directed) graph
    the connected components are given by the connected components of that
    graph. Connected components of a graph can be efficiently computed
    using Tarjan's algorithm.

    References
    ----------
    .. [1] Tarjan, R E. 1972. Depth-first search and linear graph
        algorithms. SIAM Journal on Computing 1 (2): 146-160.

    Examples
    --------

    >>> import numpy as np
    >>> from msmtools.estimation import connected_sets

    >>> C = np.array([[10, 1, 0], [2, 0, 3], [0, 0, 4]])
    >>> cc_directed = connected_sets(C)
    >>> cc_directed
    [array([0, 1]), array([2])]

    >>> cc_undirected = connected_sets(C, directed=False)
    >>> cc_undirected
    [array([0, 1, 2])]

    """
    if isdense(C):
        return sparse.connectivity.connected_sets(csr_matrix(C), directed=directed)
    else:
        return sparse.connectivity.connected_sets(C, directed=directed)


def largest_connected_set(C, directed=True):
    r"""Largest connected component for a directed graph with edge-weights
    given by the count matrix.

    Parameters
    ----------
    C : scipy.sparse matrix
        Count matrix specifying edge weights.
    directed : bool, optional
       Whether to compute connected components for a directed  or
       undirected graph. Default is True.

    Returns
    -------
    lcc : array of integers
        The largest connected component of the directed graph.

    See also
    --------
    connected_sets

    Notes
    -----
    Viewing the count matrix as the adjacency matrix of a (directed)
    graph the largest connected set is the largest connected set of
    nodes of the corresponding graph. The largest connected set of a graph
    can be efficiently computed using Tarjan's algorithm.

    References
    ----------
    .. [1] Tarjan, R E. 1972. Depth-first search and linear graph
        algorithms. SIAM Journal on Computing 1 (2): 146-160.

    Examples
    --------

    >>> import numpy as np
    >>> from msmtools.estimation import largest_connected_set

    >>> C =  np.array([[10, 1, 0], [2, 0, 3], [0, 0, 4]])
    >>> lcc_directed = largest_connected_set(C)
    >>> lcc_directed
    array([0, 1])

    >>> lcc_undirected = largest_connected_set(C, directed=False)
    >>> lcc_undirected
    array([0, 1, 2])

    """
    if isdense(C):
        return sparse.connectivity.largest_connected_set(csr_matrix(C), directed=directed)
    else:
        return sparse.connectivity.largest_connected_set(C, directed=directed)


@shortcut('connected_cmatrix')
def largest_connected_submatrix(C, directed=True, lcc=None):
    r"""Compute the count matrix on the largest connected set.

    Parameters
    ----------
    C : scipy.sparse matrix
        Count matrix specifying edge weights.
    directed : bool, optional
       Whether to compute connected components for a directed or
       undirected graph. Default is True
    lcc : (M,) ndarray, optional
       The largest connected set

    Returns
    -------
    C_cc : scipy.sparse matrix
        Count matrix of largest completely
        connected set of vertices (states)

    See also
    --------
    largest_connected_set

    Notes
    -----
    Viewing the count matrix as the adjacency matrix of a (directed)
    graph the larest connected submatrix is the adjacency matrix of
    the largest connected set of the corresponding graph. The largest
    connected submatrix can be efficiently computed using Tarjan's algorithm.

    References
    ----------
    .. [1] Tarjan, R E. 1972. Depth-first search and linear graph
        algorithms. SIAM Journal on Computing 1 (2): 146-160.

    Examples
    --------

    >>> import numpy as np
    >>> from msmtools.estimation import largest_connected_submatrix

    >>> C = np.array([[10, 1, 0], [2, 0, 3], [0, 0, 4]])

    >>> C_cc_directed = largest_connected_submatrix(C)
    >>> C_cc_directed # doctest: +ELLIPSIS
    array([[10,  1],
           [ 2,  0]]...)

    >>> C_cc_undirected = largest_connected_submatrix(C, directed=False)
    >>> C_cc_undirected # doctest: +ELLIPSIS
    array([[10,  1,  0],
           [ 2,  0,  3],
           [ 0,  0,  4]]...)

    """
    if isdense(C):
        return sparse.connectivity.largest_connected_submatrix(csr_matrix(C), directed=directed, lcc=lcc).toarray()
    else:
        return sparse.connectivity.largest_connected_submatrix(C, directed=directed, lcc=lcc)


def is_connected(C, directed=True):
    """Check connectivity of the given matrix.

    Parameters
    ----------
    C : scipy.sparse matrix
        Count matrix specifying edge weights.
    directed : bool, optional
       Whether to compute connected components for a directed or
       undirected graph. Default is True.

    Returns
    -------
    is_connected: bool
        True if C is connected, False otherwise.

    See also
    --------
    largest_connected_submatrix

    Notes
    -----
    A count matrix is connected if the graph having the count matrix
    as adjacency matrix has a single connected component. Connectivity
    of a graph can be efficiently checked using Tarjan's algorithm.

    References
    ----------
    .. [1] Tarjan, R E. 1972. Depth-first search and linear graph
        algorithms. SIAM Journal on Computing 1 (2): 146-160.

    Examples
    --------

    >>> import numpy as np
    >>> from msmtools.estimation import is_connected

    >>> C = np.array([[10, 1, 0], [2, 0, 3], [0, 0, 4]])
    >>> is_connected(C)
    False

    >>> is_connected(C, directed=False)
    True

    """
    if isdense(C):
        return sparse.connectivity.is_connected(csr_matrix(C), directed=directed)
    else:
        return sparse.connectivity.is_connected(C, directed=directed)


################################################################################
# priors
################################################################################

def prior_neighbor(C, alpha=0.001):
    r"""Neighbor prior for the given count matrix.

    Parameters
    ----------
    C : (M, M) ndarray or scipy.sparse matrix
        Count matrix
    alpha : float (optional)
        Value of prior counts

    Returns
    -------
    B : (M, M) ndarray or scipy.sparse matrix
        Prior count matrix

    Notes
    ------
    The neighbor prior :math:`b_{ij}` is defined as

    .. math:: b_{ij}=\left \{ \begin{array}{rl}
                     \alpha & c_{ij}+c_{ji}>0 \\
                     0      & \text{else}
                     \end{array} \right .

    Examples
    --------

    >>> import numpy as np
    >>> from msmtools.estimation import prior_neighbor

    >>> C = np.array([[10, 1, 0], [2, 0, 3], [0, 1, 4]])
    >>> B = prior_neighbor(C)
    >>> B
    array([[ 0.001,  0.001,  0.   ],
           [ 0.001,  0.   ,  0.001],
           [ 0.   ,  0.001,  0.001]])

    """

    if isdense(C):
        B = sparse.prior.prior_neighbor(csr_matrix(C), alpha=alpha)
        return B.toarray()
    else:
        return sparse.prior.prior_neighbor(C, alpha=alpha)


def prior_const(C, alpha=0.001):
    r"""Constant prior for given count matrix.

    Parameters
    ----------
    C : (M, M) ndarray or scipy.sparse matrix
        Count matrix
    alpha : float (optional)
        Value of prior counts

    Returns
    -------
    B : (M, M) ndarray
        Prior count matrix

    Notes
    -----
    The prior is defined as

    .. math:: \begin{array}{rl} b_{ij}= \alpha & \forall i, j \end{array}

    Examples
    --------

    >>> import numpy as np
    >>> from msmtools.estimation import prior_const

    >>> C = np.array([[10, 1, 0], [2, 0, 3], [0, 1, 4]])
    >>> B = prior_const(C)
    >>> B
    array([[ 0.001,  0.001,  0.001],
           [ 0.001,  0.001,  0.001],
           [ 0.001,  0.001,  0.001]])

    """
    if isdense(C):
        return sparse.prior.prior_const(C, alpha=alpha)
    else:
        warnings.warn("Prior will be a dense matrix for sparse input")
        return sparse.prior.prior_const(C, alpha=alpha)


__all__.append('prior_const')


def prior_rev(C, alpha=-1.0):
    r"""Prior counts for sampling of reversible transition
    matrices.

    Prior is defined as

    b_ij= alpha if i<=j
    b_ij=0         else

    Parameters
    ----------
    C : (M, M) ndarray or scipy.sparse matrix
        Count matrix
    alpha : float (optional)
        Value of prior counts

    Returns
    -------
    B : (M, M) ndarray
        Matrix of prior counts

    Notes
    -----
    The reversible prior is a matrix with -1 on the upper triangle.
    Adding this prior respects the fact that
    for a reversible transition matrix the degrees of freedom
    correspond essentially to the upper triangular part of the matrix.

    The prior is defined as

    .. math:: b_{ij} = \left \{ \begin{array}{rl}
                       \alpha & i \leq j \\
                       0      & \text{elsewhere}
                       \end{array} \right .

    Examples
    --------

    >>> import numpy as np
    >>> from msmtools.estimation import prior_rev

    >>> C = np.array([[10, 1, 0], [2, 0, 3], [0, 1, 4]])
    >>> B = prior_rev(C)
    >>> B
    array([[-1., -1., -1.],
           [ 0., -1., -1.],
           [ 0.,  0., -1.]])

    """
    if isdense(C):
        return sparse.prior.prior_rev(C, alpha=alpha)
    else:
        warnings.warn("Prior will be a dense matrix for sparse input")
        return sparse.prior.prior_rev(C, alpha=alpha)


################################################################################
# Transition matrix
################################################################################

@shortcut('tmatrix')
def transition_matrix(C, reversible=False, mu=None, method='auto', **kwargs):
    r"""Estimate the transition matrix from the given countmatrix.

    Parameters
    ----------
    C : numpy ndarray or scipy.sparse matrix
        Count matrix
    reversible : bool (optional)
        If True restrict the ensemble of transition matrices
        to those having a detailed balance symmetry otherwise
        the likelihood optimization is carried out over the whole
        space of stochastic matrices.
    mu : array_like
        The stationary distribution of the MLE transition matrix.
    method : str
        Select which implementation to use for the estimation.
        One of 'auto', 'dense' and 'sparse', optional, default='auto'.
        'dense' always selects the dense implementation, 'sparse' always selects
        the sparse one.
        'auto' selects the most efficient implementation according to
        the sparsity structure of the matrix: if the occupation of the C
        matrix is less then one third, select sparse. Else select dense.
        The type of the T matrix returned always matches the type of the
        C matrix, irrespective of the method that was used to compute it.
    **kwargs: Optional algorithm-specific parameters. See below for special cases
    Xinit : (M, M) ndarray
        Optional parameter with reversible = True.
        initial value for the matrix of absolute transition probabilities. Unless set otherwise,
        will use X = diag(pi) t, where T is a nonreversible transition matrix estimated from C,
        i.e. T_ij = c_ij / sum_k c_ik, and pi is its stationary distribution.
    maxiter : 1000000 : int
        Optional parameter with reversible = True.
        maximum number of iterations before the method exits
    maxerr : 1e-8 : float
        Optional parameter with reversible = True.
        convergence tolerance for transition matrix estimation.
        This specifies the maximum change of the Euclidean norm of relative
        stationary probabilities (:math:`x_i = \sum_k x_{ik}`). The relative stationary probability changes
        :math:`e_i = (x_i^{(1)} - x_i^{(2)})/(x_i^{(1)} + x_i^{(2)})` are used in order to track changes in small
        probabilities. The Euclidean norm of the change vector, :math:`|e_i|_2`, is compared to maxerr.
    rev_pisym : bool, default=False
        Fast computation of reversible transition matrix by normalizing
        :math:`x_{ij} = \pi_i p_{ij} + \pi_j p_{ji}`. :math:`p_{ij}` is the direct
        (nonreversible) estimate and :math:`pi_i` is its stationary distribution.
        This estimator is asympotically unbiased but not maximum likelihood.
    return_statdist : bool, default=False
        Optional parameter with reversible = True.
        If set to true, the stationary distribution is also returned
    return_conv : bool, default=False
        Optional parameter with reversible = True.
        If set to true, the likelihood history and the pi_change history is returned.
    warn_not_converged : bool, default=True
        Prints a warning if not converged.
    sparse_newton : bool, default=False
        If True, use the experimental primal-dual interior-point solver for sparse input/computation method.

    Returns
    -------
    P : (M, M) ndarray or scipy.sparse matrix
       The MLE transition matrix. P has the same data type (dense or sparse)
       as the input matrix C.
    The reversible estimator returns by default only P, but may also return
    (P,pi) or (P,lhist,pi_changes) or (P,pi,lhist,pi_changes) depending on the return settings
    P : ndarray (n,n)
        transition matrix. This is the only return for return_statdist = False, return_conv = False
    (pi) : ndarray (n)
        stationary distribution. Only returned if return_statdist = True
    (lhist) : ndarray (k)
        likelihood history. Has the length of the number of iterations needed.
        Only returned if return_conv = True
    (pi_changes) : ndarray (k)
        history of likelihood history. Has the length of the number of iterations needed.
        Only returned if return_conv = True

    Notes
    -----
    The transition matrix is a maximum likelihood estimate (MLE) of
    the probability distribution of transition matrices with
    parameters given by the count matrix.

    References
    ----------
    .. [1] Prinz, J H, H Wu, M Sarich, B Keller, M Senne, M Held, J D
        Chodera, C Schuette and F Noe. 2011. Markov models of
        molecular kinetics: Generation and validation. J Chem Phys
        134: 174105
    .. [2] Bowman, G R, K A Beauchamp, G Boxer and V S Pande. 2009.
        Progress and challenges in the automated construction of Markov state models for full protein systems.
        J. Chem. Phys. 131: 124101
    .. [3] Trendelkamp-Schroer, B, H Wu, F Paul and F. Noe. 2015
        Estimation and uncertainty of reversible Markov models.
        J. Chem. Phys. 143: 174101

    Examples
    --------

    >>> import numpy as np
    >>> from msmtools.estimation import transition_matrix

    >>> C = np.array([[10, 1, 1], [2, 0, 3], [0, 1, 4]])

    Non-reversible estimate

    >>> T_nrev = transition_matrix(C)
    >>> T_nrev
    array([[ 0.83333333,  0.08333333,  0.08333333],
           [ 0.4       ,  0.        ,  0.6       ],
           [ 0.        ,  0.2       ,  0.8       ]])

    Reversible estimate

    >>> T_rev = transition_matrix(C, reversible=True)
    >>> T_rev
    array([[ 0.83333333,  0.10385551,  0.06281115],
           [ 0.35074677,  0.        ,  0.64925323],
           [ 0.04925323,  0.15074677,  0.8       ]])

    Reversible estimate with given stationary vector

    >>> mu = np.array([0.7, 0.01, 0.29])
    >>> T_mu = transition_matrix(C, reversible=True, mu=mu)
    >>> T_mu
    array([[ 0.94771371,  0.00612645,  0.04615984],
           [ 0.42885157,  0.        ,  0.57114843],
           [ 0.11142031,  0.01969477,  0.86888491]])

    """
    if issparse(C):
        sparse_input_type = True
    elif isdense(C):
        sparse_input_type = False
    else:
        raise NotImplementedError('C has an unknown type.')

    if method == 'dense':
        sparse_computation = False
    elif method == 'sparse':
        sparse_computation = True
    elif method == 'auto':
        # heuristically determine whether is't more efficient to do a dense of sparse computation
        if sparse_input_type:
            dof = C.getnnz()
        else:
            dof = np.count_nonzero(C)
        dimension = C.shape[0]
        if dimension*dimension < 3*dof:
            sparse_computation = False
        else:
            sparse_computation = True
    else:
        raise ValueError(('method="%s" is no valid choice. It should be one of'
                          '"dense", "sparse" or "auto".') % method)

    # convert input type
    if sparse_computation and not sparse_input_type:
        C = coo_matrix(C)
    if not sparse_computation and sparse_input_type:
        C = C.toarray()

    return_statdist = 'return_statdist' in kwargs

    if not return_statdist:
        kwargs['return_statdist'] = False

    sparse_newton = kwargs.pop('sparse_newton', False)

    if reversible:
        rev_pisym = kwargs.pop('rev_pisym', False)

        if mu is None:
            if sparse_computation:
                if rev_pisym:
                    result = sparse.transition_matrix.transition_matrix_reversible_pisym(C, **kwargs)
                elif sparse_newton:
                    from .sparse.mle.newton.mle_rev import solve_mle_rev
                    result = solve_mle_rev(C, **kwargs)
                else:
                    result = sparse.mle.mle_trev.mle_trev(C, **kwargs)
            else:
                if rev_pisym:
                    result = dense.transition_matrix.transition_matrix_reversible_pisym(C, **kwargs)
                else:
                    result = dense.mle.mle_trev.mle_trev(C, **kwargs)
        else:
            kwargs.pop('return_statdist') # pi given, keyword unknown by estimators.
            if sparse_computation:
                # Sparse, reversible, fixed pi (currently using dense with sparse conversion)
                result = sparse.mle.mle_trev_given_pi.mle_trev_given_pi(C, mu, **kwargs)
            else:
                result = dense.mle.mle_trev_given_pi.mle_trev_given_pi(C, mu, **kwargs)
    else:  # nonreversible estimation
        if mu is None:
            if sparse_computation:
                # Sparse,  nonreversible
                result = sparse.transition_matrix.transition_matrix_non_reversible(C)
            else:
                # Dense,  nonreversible
                result = dense.transition_matrix.transition_matrix_non_reversible(C)
            # Both methods currently do not have an iterate of pi, so we compute it here for consistency.
            if return_statdist:
                from ..analysis import stationary_distribution
                mu = stationary_distribution(result)
        else:
            raise NotImplementedError('nonreversible mle with fixed stationary distribution not implemented.')

    if return_statdist and isinstance(result, tuple):
        T, mu = result
    else:
        T = result

    # convert return type
    if sparse_computation and not sparse_input_type:
        T = T.toarray()
    elif not sparse_computation and sparse_input_type:
        T = csr_matrix(T)

    if return_statdist:
        return T, mu
    return T


def log_likelihood(C, T):
    r"""Log-likelihood of the count matrix given a transition matrix.

    Parameters
    ----------
    C : (M, M) ndarray or scipy.sparse matrix
        Count matrix
    T : (M, M) ndarray orscipy.sparse matrix
        Transition matrix

    Returns
    -------
    logL : float
        Log-likelihood of the count matrix

    Notes
    -----

    The likelihood of a set of observed transition counts
    :math:`C=(c_{ij})` for a given matrix of transition counts
    :math:`T=(t_{ij})` is given by

    .. math:: L(C|P)=\prod_{i=1}^{M} \left( \prod_{j=1}^{M} p_{ij}^{c_{ij}} \right)

    The log-likelihood is given by

    .. math:: l(C|P)=\sum_{i,j=1}^{M}c_{ij} \log p_{ij}.

    The likelihood describes the probability of making an observation
    :math:`C` for a given model :math:`P`.

    Examples
    --------

    >>> import numpy as np
    >>> from msmtools.estimation import log_likelihood

    >>> T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])

    >>> C = np.array([[58, 7, 0], [6, 0, 4], [0, 3, 21]])
    >>> logL = log_likelihood(C, T)
    >>> logL # doctest: +ELLIPSIS
    -38.2808034725...

    >>> C = np.array([[58, 20, 0], [6, 0, 4], [0, 3, 21]])
    >>> logL = log_likelihood(C, T)
    >>> logL # doctest: +ELLIPSIS
    -68.2144096814...

    References
    ----------
    .. [1] Prinz, J H, H Wu, M Sarich, B Keller, M Senne, M Held, J D
        Chodera, C Schuette and F Noe. 2011. Markov models of
        molecular kinetics: Generation and validation. J Chem Phys
        134: 174105

    """
    if issparse(C) and issparse(T):
        return sparse.likelihood.log_likelihood(C, T)
    else:
        # use the dense likelihood calculator for all other cases
        # if a mix of dense/sparse C/T matrices is used, then both
        # will be converted to ndarrays.
        if not isinstance(C, np.ndarray):
            C = np.array(C)
        if not isinstance(T, np.ndarray):
            T = np.array(T)
        # computation is still efficient, because we only use terms
        # for nonzero elements of T
        nz = np.nonzero(T)
        return np.dot(C[nz], np.log(T[nz]))


def tmatrix_cov(C, k=None):
    r"""Covariance tensor for non-reversible transition matrix posterior.

    Parameters
    ----------
    C : (M, M) ndarray or scipy.sparse matrix
        Count matrix
    k : int (optional)
        Return only covariance matrix for entires in the k-th row of
        the transition matrix

    Returns
    -------
    cov : (M, M, M) ndarray
        Covariance tensor for transition matrix posterior

    Notes
    -----
    The posterior of non-reversible transition matrices is

    .. math:: \mathbb{P}(T|C) \propto \prod_{i=1}^{M} \left( \prod_{j=1}^{M} p_{ij}^{c_{ij}} \right)

    Each row in the transition matrix is distributed according to a
    Dirichlet distribution with parameters given by the observed
    transition counts :math:`c_{ij}`.

    The covariance tensor
    :math:`\text{cov}[p_{ij},p_{kl}]=\Sigma_{i,j,k,l}` is zero
    whenever :math:`i \neq k` so that only :math:`\Sigma_{i,j,i,l}` is
    returned.

    """
    if issparse(C):
        warnings.warn("Covariance matrix will be dense for sparse input")
        C = C.toarray()
    return dense.covariance.tmatrix_cov(C, row=k)


def error_perturbation(C, S):
    r"""Error perturbation for given sensitivity matrix.

    Parameters
    ----------
    C : (M, M) ndarray
        Count matrix
    S : (M, M) ndarray or (K, M, M) ndarray
        Sensitivity matrix (for scalar observable) or sensitivity
        tensor for vector observable

    Returns
    -------
    X : float or (K, K) ndarray
        error-perturbation (for scalar observables) or covariance matrix
        (for vector-valued observable)

    Notes
    -----

    **Scalar observable**

    The sensitivity matrix :math:`S=(s_{ij})` of a scalar observable
    :math:`f(T)` is defined as

    .. math:: S= \left(\left. \frac{\partial f(T)}{\partial t_{ij}} \right \rvert_{T_0} \right)

    evaluated at a suitable transition matrix :math:`T_0`.

    The sensitivity is the variance of the observable

    .. math:: \mathbb{V}(f)=\sum_{i,j,k,l} s_{ij} \text{cov}[t_{ij}, t_{kl}] s_{kl}

    **Vector valued observable**

    The sensitivity tensor :math:`S=(s_{ijk})` for a vector
    valued observable :math:`(f_1(T),\dots,f_K(T))` is defined as

    .. math:: S= \left( \left. \frac{\partial f_i(T)}{\partial t_{jk}} \right\rvert_{T_0} \right)
    evaluated at a suitable transition matrix :math:`T_0`.

    The sensitivity is the covariance matrix for the observable

    .. math:: \text{cov}[f_{\alpha}(T),f_{\beta}(T)] = \sum_{i,j,k,l} s_{\alpha i j}
                                                       \text{cov}[t_{ij}, t_{kl}] s_{\beta kl}

    """
    if issparse(C):
        warnings.warn("Error-perturbation will be dense for sparse input")
        C = C.toarray()
    return dense.covariance.error_perturbation(C, S)


def _showSparseConversionWarning():
    warnings.warn('Converting input to dense, since method is '
                  'currently only implemented for dense matrices.', UserWarning)


def sample_tmatrix(C, nsample=1, nsteps=None, reversible=False, mu=None, T0=None, return_statdist=False):
    r"""samples transition matrices from the posterior distribution

    Parameters
    ----------
    C : (M, M) ndarray or scipy.sparse matrix
        Count matrix
    nsample : int
        number of samples to be drawn
    nstep : int, default=None
        number of full Gibbs sampling sweeps internally done for each sample
        returned. This option is meant to ensure approximately uncorrelated
        samples for every call to sample(). If None, the number of steps will
        be automatically determined based on the other options and  the matrix
        size. nstep>1 will only be used for reversible sampling, because
        nonreversible sampling generates statistically independent transition
        matrices every step.
    reversible : bool
        If true sample from the ensemble of transition matrices
        restricted to those obeying a detailed balance condition,
        else draw from the whole ensemble of stochastic matrices.
    mu : array_like
        A fixed stationary distribution. Transition matrices with that stationary distribution will be sampled
    T0 : ndarray, shape=(n, n) or scipy.sparse matrix
        Starting point of the MC chain of the sampling algorithm.
        Has to obey the required constraints.
    return_statdist : bool, optional, default = False
        if true, will also return the stationary distribution.

    Returns
    -------
    P : ndarray(n,n) or array of ndarray(n,n)
        sampled transition matrix (or multiple matrices if nsample > 1)

    Notes
    -----
    The transition matrix sampler generates transition matrices from
    the posterior distribution. The posterior distribution is given as
    a product of Dirichlet distributions

    .. math:: \mathbb{P}(T|C) \propto \prod_{i=1}^{M} \left( \prod_{j=1}^{M} p_{ij}^{c_{ij}} \right)

    See also
    --------
    tmatrix_sampler

    """
    if issparse(C):
        _showSparseConversionWarning()
        C = C.toarray()

    sampler = tmatrix_sampler(C, reversible=reversible, mu=mu, T0=T0, nsteps=nsteps)
    return sampler.sample(nsamples=nsample, return_statdist=return_statdist)


def tmatrix_sampler(C, reversible=False, mu=None, T0=None, nsteps=None, prior='sparse'):
    r"""Generate transition matrix sampler object.

    Parameters
    ----------
    C : (M, M) ndarray or scipy.sparse matrix
        Count matrix
    reversible : bool
        If true sample from the ensemble of transition matrices
        restricted to those obeying a detailed balance condition,
        else draw from the whole ensemble of stochastic matrices.
    mu : array_like
        A fixed stationary distribution. Transition matrices with that
        stationary distribution will be sampled
    T0 : ndarray, shape=(n, n) or scipy.sparse matrix
        Starting point of the MC chain of the sampling algorithm.
        Has to obey the required constraints.
    nstep : int, default=None
        number of full Gibbs sampling sweeps per sample. This option is meant
        to ensure approximately uncorrelated samples for every call to
        sample(). If None, the number of steps will be automatically determined
        based on the other options and  the matrix size. nstep>1 will only be
        used for reversible sampling, because nonreversible sampling generates
        statistically independent transition matrices every step.

    Returns
    -------
    sampler : A :py:class:dense.tmatrix_sampler.TransitionMatrixSampler object that can be used to generate samples.

    Notes
    -----
    The transition matrix sampler generates transition matrices from
    the posterior distribution. The posterior distribution is given as
    a product of Dirichlet distributions

    .. math:: \mathbb{P}(T|C) \propto \prod_{i=1}^{M} \left( \prod_{j=1}^{M} p_{ij}^{c_{ij}} \right)

    The method can generate samples from the posterior under the following constraints

    **Reversible sampling**

    Using a MCMC sampler outlined in .. [1] it is ensured that samples
    from the posterior are reversible, i.e. there is a probability
    vector :math:`(\mu_i)` such that :math:`\mu_i t_{ij} = \mu_j
    t_{ji}` holds for all :math:`i,j`.

    **Reversible sampling with fixed stationary vector**

    Using a MCMC sampler outlined in .. [2] it is ensured that samples
    from the posterior fulfill detailed balance with respect to a given
    probability vector :math:`(\mu_i)`.

    References
    ----------
    .. [1] Noe, F. Probability distributions of molecular observables
        computed from Markov state models. J Chem Phys 128: 244103 (2008)

    .. [2] Trendelkamp-Schroer, B., H. Wu, F. Paul and F. Noe: Estimation and
        uncertainty of reversible Markov models. J. Chem. Phys. (submitted)

    """
    if issparse(C):
        _showSparseConversionWarning()
        C = C.toarray()

    from .dense.tmat_sampling.tmatrix_sampler import TransitionMatrixSampler
    sampler = TransitionMatrixSampler(C, reversible=reversible, mu=mu, P0=T0,
                                      nsteps=nsteps, prior=prior)
    return sampler


def rate_matrix(C, dt=1.0, method='KL', sparsity=None,
                t_agg=None, pi=None, tol=1.0E7, K0=None,
                maxiter=100000, on_error='raise'):
    r"""Estimate a reversible rate matrix from a count matrix.

    Parameters
    ----------
    C : (N,N) ndarray
        count matrix at a lag time dt
    dt : float, optional, default=1.0
        lag time that was used to estimate C
    method : str, one of {'KL', 'CVE', 'pseudo', 'truncated_log'}
        Method to use for estimation of the rate matrix.

        * 'pseudo' selects the pseudo-generator. A reversible transition
          matrix T is estimated and :math:`(T-Id)/d` is returned as the rate matrix.

        * 'truncated_log' selects the truncated logarithm [3]_. A
          reversible transition matrix T is estimated and :math:`max(logm(T*T)/(2dt),0)`
          is returned as the rate matrix. logm is the matrix logarithm and
          the maximum is taken element-wise.

        * 'CVE' selects the algorithm of Crommelin and Vanden-Eijnden [1]_.
          It consists of minimizing the following objective function:

          .. math:: f(K)=\sum_{ij}\left(\sum_{kl} U_{ik}^{-1}K_{kl}U_{lj}-L_{ij}\right)^2 \left|\Lambda_{i}\Lambda_{j}\right|

          where :math:`\Lambda_i` are the eigenvalues of :math:`T` and :math:`U`
          is the matrix of its (right) eigenvectors; :math:`L_{ij}=\delta_{ij}\frac{1}{\tau}\log\left|\Lambda_i\right|`.
          :math:`T` is computed from C using the reversible maximum likelihood
          estimator.

        * 'KL' selects the algorihtm of Kalbfleisch and Lawless [2]_.
          It consists of maximizing the following log-likelihood:

          .. math:: f(K)=\log L=\sum_{ij}C_{ij}\log(e^{K\Delta t})_{ij}

          where :math:`C_{ij}` are the transition counts at a lag-time :math:`\Delta t`.
          Here :math:`e` is the matrix exponential and the logarithm is taken
          element-wise.

    sparsity : (N,N) ndarray or None, optional, default=None
        If sparsity is None, a fully occupied rate matrix will be estimated.
        Alternatively, with the methods 'CVE' and 'KL' a ndarray of the
        same shape as C can be supplied. If sparsity[i,j]=0 and sparsity[j,i]=0
        the rate matrix elements :math:`K_{ij}` and :math:`K_{ji}` will be
        constrained to zero.
    t_agg : float, optional
        the aggregated simulation time;
        by default this is the total number of transition counts times
        the lag time (no sliding window counting). This value is used
        to compute the lower bound on the transition rate (that are not zero).
        If sparsity is None, this value is ignored.
    pi : (N) ndarray, optional
        the stationary vector of the desired rate matrix K.
        If no pi is given, the function takes the stationary vector
        of the MLE reversible T matrix that is computed from C.
    tol : float, optional, default = 1.0E7
        Tolerance of the quasi-Newton algorithm that is used to minimize
        the objective function. This is passed as the `factr` parameter to
        `scipy.optimize.fmin_l_bfgs_b`.
        Typical values for factr are: 1e12 for low accuracy; 1e7
        for moderate accuracy; 10.0 for extremely high accuracy.
    maxiter : int, optional, default = 100000
        Minimization of the objective function will do at most this number
        of steps.
    on_error : string, optional, default = 'raise'
        What to do then an error happend. When 'raise' is given, raise
        an exception. When 'warn' is given, produce a (Python) warning.

    Returns
    -------
    K : (N,N) ndarray
        the optimal rate matrix

    Notes
    -----
    In this implementation the algorithm of Crommelin and Vanden-Eijnden
    (CVE) is initialized with the pseudo-generator estimate. The
    algorithm of Kalbfleisch and Lawless (KL) is initialized using the
    CVE result.

    Example
    -------
    >>> import numpy as np
    >>> from msmtools.estimation import rate_matrix
    >>> C = np.array([[100,1],[50,50]])
    >>> rate_matrix(C)
    array([[-0.01384753,  0.01384753],
           [ 0.69930032, -0.69930032]])

    References
    ----------
    .. [1] D. Crommelin and E. Vanden-Eijnden. Data-based inference of
        generators for markov jump processes using convex optimization.
        Multiscale. Model. Sim., 7(4):1751-1778, 2009.
    .. [2] J. D. Kalbfleisch and J. F. Lawless. The analysis of panel
        data under a markov assumption. J. Am. Stat. Assoc.,
        80(392):863-871, 1985.
    .. [3] E. B. Davies. Embeddable Markov Matrices. Electron. J. Probab.
        15:1474, 2010.
    """

    from .dense.ratematrix import estimate_rate_matrix
    return estimate_rate_matrix(C, dt=dt, method=method, sparsity=sparsity,
                         t_agg=t_agg, pi=pi, tol=tol, K0=K0,
                         maxiter=maxiter, on_error=on_error)

