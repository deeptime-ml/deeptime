r"""

=======================
 Analysis API
=======================

"""

import warnings

import numpy as _np
from scipy.sparse import issparse as _issparse
from deeptime.util.types import ensure_number_array, ensure_integer_array, ensure_floating_array

from ._stationary_vector import stationary_distribution
from ._decomposition import timescales_from_eigenvalues

from . import dense
from . import _assessment
from . import _mean_first_passage_time
from . import _decomposition
from . import _fingerprints
from . import _committor
from . import _expectations

__docformat__ = "restructuredtext en"
__authors__ = __author__ = "Benjamin Trendelkamp-Schroer, Martin Scherer, Jan-Hendrik Prinz, Frank Noe"
__copyright__ = "Copyright 2014, Computational Molecular Biology Group, FU-Berlin"
__credits__ = ["Benjamin Trendelkamp-Schroer", "Martin Scherer", "Jan-Hendrik Prinz", "Frank Noe"]

__version__ = "2.0.0"
__maintainer__ = "Martin Scherer"
__email__ = "m.scherer AT fu-berlin DOT de"


_type_not_supported = \
    TypeError("T is not a numpy.ndarray or a scipy.sparse matrix.")


################################################################################
# Assessment tools
################################################################################


def is_transition_matrix(T, tol=1e-12):
    r"""Check if the given matrix is a transition matrix.

    Parameters
    ----------
    T : (M, M) ndarray or scipy.sparse matrix
        Matrix to check
    tol : float (optional)
        Floating point tolerance to check with

    Returns
    -------
    is_transition_matrix : bool
        True, if T is a valid transition matrix, False otherwise

    Notes
    -----
    A valid transition matrix :math:`P=(p_{ij})` has non-negative
    elements, :math:`p_{ij} \geq 0`, and elements of each row sum up
    to one, :math:`\sum_j p_{ij} = 1`. Matrices wit this property are
    also called stochastic matrices.

    Examples
    --------
    >>> import numpy as np
    >>> from deeptime.markov.tools.analysis import is_transition_matrix

    >>> A = np.array([[0.4, 0.5, 0.3], [0.2, 0.4, 0.4], [-1, 1, 1]])
    >>> is_transition_matrix(A)
    False

    >>> T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])
    >>> is_transition_matrix(T)
    True

    """
    T = ensure_number_array(T, ndim=2)
    return _assessment.is_transition_matrix(T, tol)


def is_rate_matrix(K, tol=1e-12):
    r"""Check if the given matrix is a rate matrix.

    Parameters
    ----------
    K : (M, M) ndarray or scipy.sparse matrix
        Matrix to check
    tol : float (optional)
        Floating point tolerance to check with

    Returns
    -------
    is_rate_matrix : bool
        True, if K is a valid rate matrix, False otherwise

    Notes
    -----
    A valid rate matrix :math:`K=(k_{ij})` has non-negative off
    diagonal elements, :math:`k_{ij} \leq 0`, for :math:`i \neq j`,
    and elements of each row sum up to zero, :math:`\sum_{j}
    k_{ij}=0`.

    Examples
    --------
    >>> import numpy as np
    >>> from deeptime.markov.tools.analysis import is_rate_matrix

    >>> A = np.array([[0.5, -0.5, -0.2], [-0.3, 0.6, -0.3], [-0.2, 0.2, 0.0]])
    >>> is_rate_matrix(A)
    False

    >>> K = np.array([[-0.3, 0.2, 0.1], [0.5, -0.5, 0.0], [0.1, 0.1, -0.2]])
    >>> is_rate_matrix(K)
    True

    """
    K = ensure_number_array(K, ndim=2)
    return _assessment.is_rate_matrix(K, tol)


def is_connected(T, directed=True):
    r"""Check connectivity of the given matrix.

    Parameters
    ----------
    T : (M, M) ndarray or scipy.sparse matrix
        Matrix to check
    directed : bool (optional)
       If True respect direction of transitions, if False do not
       distinguish between forward and backward transitions

    Returns
    -------
    is_connected : bool
        True, if T is connected, False otherwise

    Notes
    -----
    A transition matrix :math:`T=(t_{ij})` is connected if for any pair
    of states :math:`(i, j)` one can reach state :math:`j` from state
    :math:`i` in a finite number of steps.

    In more precise terms: For any pair of states :math:`(i, j)` there
    exists a number :math:`N=N(i, j)`, so that the probability of
    going from state :math:`i` to state :math:`j` in :math:`N` steps
    is positive, :math:`\mathbb{P}(X_{N}=j|X_{0}=i)>0`.

    A transition matrix with this property is also called irreducible.

    Viewing the transition matrix as the adjency matrix of a
    (directed) graph the transition matrix is irreducible if and only
    if the corresponding graph has a single connected
    component. :footcite:`hoel1986introduction`.

    Connectivity of a graph can be efficiently checked
    using Tarjan's algorithm. :footcite:`tarjan1972depth`.

    References
    ----------
    .. footbibliography::

    Examples
    --------

    >>> import numpy as np
    >>> from deeptime.markov.tools.analysis import is_connected

    >>> A = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.0, 1.0]])
    >>> is_connected(A)
    False

    >>> T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])
    >>> is_connected(T)
    True

    """
    T = ensure_number_array(T, ndim=2)
    return _assessment.is_connected(T, directed=directed)


def is_reversible(T, mu=None, tol=1e-12):
    r"""Check reversibility of the given transition matrix.

    Parameters
    ----------
    T : (M, M) ndarray or scipy.sparse matrix
        Transition matrix
    mu : (M,) ndarray (optional)
         Test reversibility with respect to this vector
    tol : float (optional)
        Floating point tolerance to check with

    Returns
    -------
    is_reversible : bool
        True, if T is reversible, False otherwise

    Notes
    -----
    A transition matrix :math:`T=(t_{ij})` is reversible with respect
    to a probability vector :math:`\mu=(\mu_i)` if the follwing holds,

    .. math:: \mu_i \, t_{ij}= \mu_j \, t_{ji}.

    In this case :math:`\mu` is the stationary vector for :math:`T`,
    so that :math:`\mu^T T = \mu^T`.

    If the stationary vector is unknown it is computed from :math:`T`
    before reversibility is checked.

    A reversible transition matrix has purely real eigenvalues. The
    left eigenvectors :math:`(l_i)` can be computed from right
    eigenvectors :math:`(r_i)` via :math:`l_i=\mu_i r_i`.

    Examples
    --------

    >>> import numpy as np
    >>> from deeptime.markov.tools.analysis import is_reversible

    >>> P = np.array([[0.8, 0.1, 0.1], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])
    >>> is_reversible(P)
    False

    >>> T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])
    >>> is_reversible(T)
    True

    """
    # check input
    T = ensure_number_array(T, ndim=2)
    if mu is not None:
        mu = ensure_number_array(mu, ndim=1)
    return _assessment.is_reversible(T, mu, tol)


################################################################################
# Eigenvalues and eigenvectors
################################################################################

def _check_k(T, k):
    # ensure k is not exceeding shape of transition matrix
    if k is None:
        return
    n = T.shape[0]
    if _issparse(T):
        # for sparse matrices we can compute an eigendecomposition of n - 1
        new_k = min(n - 1, k)
    else:
        new_k = min(n, k)
    if new_k < k:
        warnings.warn('truncated eigendecomposition to contain %s components' % new_k, category=UserWarning)
    return new_k


def eigenvalues(T, k=None, ncv=None, reversible=False, mu=None):
    r"""Find eigenvalues of the transition matrix.

    Parameters
    ----------
    T : (M, M) ndarray or sparse matrix
        Transition matrix
    k : int (optional)
        Compute the first `k` eigenvalues of `T`
    ncv : int (optional)
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k
    reversible : bool, optional
        Indicate that transition matrix is reversible
    mu : (M,) ndarray, optional
        Stationary distribution of T

    Returns
    -------
    w : (M,) ndarray
        Eigenvalues of `T`. If `k` is specified, `w` has
        shape (k,)

    Notes
    -----
    Eigenvalues are returned in order of decreasing magnitude.

    If reversible=True the the eigenvalues of the similar symmetric
    matrix `\sqrt(\mu_i / \mu_j) p_{ij}` will be computed.

    The precomputed stationary distribution will only be used if
    reversible=True.

    Examples
    --------

    >>> import numpy as np
    >>> from deeptime.markov.tools.analysis import eigenvalues

    >>> T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])
    >>> w = eigenvalues(T)
    >>> w
    array([ 1. +0.j,  0.9+0.j, -0.1+0.j])

    """
    T = ensure_number_array(T, ndim=2)
    k = _check_k(T, k)
    return _decomposition.eigenvalues(T, k, ncv=ncv, reversible=reversible, mu=mu)


def timescales(T, tau=1, k=None, ncv=None, reversible=False, mu=None):
    r"""Compute implied time scales of given transition matrix.

    Parameters
    ----------
    T : (M, M) ndarray or scipy.sparse matrix
        Transition matrix
    tau : int (optional)
        The time-lag (in elementary time steps of the microstate
        trajectory) at which the given transition matrix was
        constructed.
    k : int (optional)
        Compute the first `k` implied time scales.
    ncv : int (optional, for sparse T only)
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k
    reversible : bool, optional
        Indicate that transition matrix is reversible
    mu : (M,) ndarray, optional
        Stationary distribution of T

    Returns
    -------
    ts : (M,) ndarray
        The implied time scales of the transition matrix.  If `k` is
        not None then the shape of `ts` is (k,).

    Notes
    -----
    The implied time scale :math:`t_i` is defined as

    .. math:: t_i=-\frac{\tau}{\log \lvert \lambda_i \rvert}

    If reversible=True the the eigenvalues of the similar symmetric
    matrix `\sqrt(\mu_i / \mu_j) p_{ij}` will be computed.

    The precomputed stationary distribution will only be used if
    reversible=True.

    Examples
    --------

    >>> import numpy as np
    >>> from deeptime.markov.tools.analysis import timescales

    >>> T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])
    >>> ts = timescales(T)
    >>> ts
    array([       inf, 9.49122158, 0.43429448])

    """
    T = ensure_number_array(T, ndim=2)
    k = _check_k(T, k)
    return _decomposition.timescales(T, tau=tau, k=k, ncv=ncv, reversible=reversible, mu=mu)


def eigenvectors(T, k=None, right=True, ncv=None, reversible=False, mu=None):
    r"""Compute eigenvectors of given transition matrix.

    Parameters
    ----------
    T : numpy.ndarray, shape(d,d) or scipy.sparse matrix
        Transition matrix (stochastic matrix)
    k : int (optional)
        Compute the first k eigenvectors
    ncv : int (optional)
        The number of Lanczos vectors generated, `ncv` must be greater than `k`;
        it is recommended that `ncv > 2*k`
    right : bool, optional
        If right=True compute right eigenvectors, left eigenvectors
        otherwise
    reversible : bool, optional
        Indicate that transition matrix is reversible
    mu : (M,) ndarray, optional
        Stationary distribution of T


    Returns
    -------
    eigvec : numpy.ndarray, shape=(d, n)
        The eigenvectors of T ordered with decreasing absolute value of
        the corresponding eigenvalue. If k is None then n=d, if k is
        int then n=k.

    See also
    --------
    rdl_decomposition

    Notes
    -----
    Eigenvectors are computed using the scipy interface
    to the corresponding LAPACK/ARPACK routines.

    If reversible=False, the returned eigenvectors :math:`v_i` are
    normalized such that

    ..  math::

        \langle v_i, v_i \rangle = 1

    This is the case for right eigenvectors :math:`r_i` as well as
    for left eigenvectors :math:`l_i`.

    If you desire orthonormal left and right eigenvectors please use the
    rdl_decomposition method.

    If reversible=True the the eigenvectors of the similar symmetric
    matrix `\sqrt(\mu_i / \mu_j) p_{ij}` will be used to compute the
    eigenvectors of T.

    The precomputed stationary distribution will only be used if
    reversible=True.

    Examples
    --------

    >>> import numpy as np
    >>> from deeptime.markov.tools.analysis import eigenvectors

    >>> T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])
    >>> R = eigenvectors(T)

    Matrix with right eigenvectors as columns

    >>> R  # doctest: +ELLIPSIS
    array([[ 5.77350269e-01,  7.07106781e-01,  9.90147543e-02]...
    """

    T = ensure_number_array(T, ndim=2)
    k = _check_k(T, k)
    ev = _decomposition.eigenvectors(T, k=k, right=right, ncv=ncv, reversible=reversible, mu=mu)
    if not right:
        ev = ev.T
    return ev


def rdl_decomposition(T, k=None, norm='auto', ncv=None, reversible=False, mu=None):
    r"""Compute the decomposition into eigenvalues, left and right
    eigenvectors.

    Parameters
    ----------
    T : (M, M) ndarray or sparse matrix
        Transition matrix
    k : int (optional)
        Number of eigenvector/eigenvalue pairs
    norm: {'standard', 'reversible', 'auto'}, optional
        which normalization convention to use

        ============ =============================================
        norm
        ============ =============================================
        'standard'   LR = Id, is a probability\
                     distribution, the stationary distribution\
                     of `T`. Right eigenvectors `R`\
                     have a 2-norm of 1
        'reversible' `R` and `L` are related via ``L[0, :]*R``
        'auto'       reversible if T is reversible, else standard.
        ============ =============================================

    ncv : int (optional)
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k
    reversible : bool, optional
        Indicate that transition matrix is reversible
    mu : (M,) ndarray, optional
        Stationary distribution of T

    Returns
    -------
    R : (M, M) ndarray
        The normalized ("unit length") right eigenvectors, such that the
        column ``R[:,i]`` is the right eigenvector corresponding to the eigenvalue
        ``w[i]``, ``dot(T,R[:,i])``=``w[i]*R[:,i]``
    D : (M, M) ndarray
        A diagonal matrix containing the eigenvalues, each repeated
        according to its multiplicity
    L : (M, M) ndarray
        The normalized (with respect to `R`) left eigenvectors, such that the
        row ``L[i, :]`` is the left eigenvector corresponding to the eigenvalue
        ``w[i]``, ``dot(L[i, :], T)``=``w[i]*L[i, :]``

    Examples
    --------

    >>> import numpy as np
    >>> from deeptime.markov.tools.analysis import rdl_decomposition

    >>> T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])
    >>> R, D, L = rdl_decomposition(T)

    Matrix with right eigenvectors as columns

    >>> R  # doctest: +ELLIPSIS
    array([[ 1.00000000e+00,  1.04880885e+00,  3.16227766e-01]...


    Diagonal matrix with eigenvalues

    >>> D
    array([[ 1. ,  0. ,  0. ],
           [ 0. ,  0.9,  0. ],
           [ 0. ,  0. , -0.1]])

    Matrix with left eigenvectors as rows

    >>> L # doctest: +ELLIPSIS
    array([[ 4.54545455e-01,  9.09090909e-02,  4.54545455e-01]...


    """
    T = ensure_number_array(T, ndim=2)
    k = _check_k(T, k)
    return _decomposition.rdl_decomposition(T, k=k, norm=norm, ncv=ncv,
                                            reversible=reversible, mu=mu)


def mfpt(T, target, origin=None, tau=1, mu=None):
    r"""Mean first passage times (from a set of starting states - optional)
    to a set of target states.

    Parameters
    ----------
    T : ndarray or scipy.sparse matrix, shape=(n,n)
        Transition matrix.
    target : int or list of int
        Target states for mfpt calculation.
    origin : int or list of int (optional)
        Set of starting states.
    tau : int (optional)
        The time-lag (in elementary time steps of the microstate
        trajectory) at which the given transition matrix was
        constructed.
    mu : (n,) ndarray (optional)
        The stationary distribution of the transition matrix T.

    Returns
    -------
    m_t : ndarray, shape=(n,) or shape(1,)
        Mean first passage time or vector of mean first passage times.

    Notes
    -----
    The mean first passage time :math:`\mathbf{E}_x[T_Y]` is the expected
    hitting time of one state :math:`y` in :math:`Y` when starting
    in state :math:`x`. :footcite:`hoel1986introduction`.

    For a fixed target state :math:`y` it is given by

    .. math :: \mathbb{E}_x[T_y] = \left \{  \begin{array}{cc}
                                             0 & x=y \\
                                             1+\sum_{z} T_{x,z} \mathbb{E}_z[T_y] & x \neq y
                                             \end{array}  \right.

    For a set of target states :math:`Y` it is given by

    .. math :: \mathbb{E}_x[T_Y] = \left \{  \begin{array}{cc}
                                             0 & x \in Y \\
                                             1+\sum_{z} T_{x,z} \mathbb{E}_z[T_Y] & x \notin Y
                                             \end{array}  \right.

    The mean first passage time between sets, :math:`\mathbf{E}_X[T_Y]`, is given by

    .. math :: \mathbb{E}_X[T_Y] = \sum_{x \in X}
                \frac{\mu_x \mathbb{E}_x[T_Y]}{\sum_{z \in X} \mu_z}

    References
    ----------
    .. footbibliography::

    Examples
    --------

    >>> import numpy as np
    >>> from deeptime.markov.tools.analysis import mfpt

    >>> T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])
    >>> m_t = mfpt(T, 0)
    >>> m_t
    array([ 0., 12., 22.])

    """
    # check inputs
    T = ensure_number_array(T, ndim=2)
    target = ensure_integer_array(target, ndim=1)
    if origin is not None:
        origin = ensure_integer_array(origin, ndim=1)
    if origin is None:
        t_tau = _mean_first_passage_time.mfpt(T, target)
    else:
        t_tau = _mean_first_passage_time.mfpt_between_sets(T, target, origin, mu=mu)
    # scale answer by lag time used.
    return tau * t_tau


def hitting_probability(T, target):
    r"""
    Computes the hitting probabilities for all states to the target states.

    The hitting probability of state i to the target set A is defined as the minimal,
    non-negative solution of:

    .. math::
        h_i^A &= 1                    \:\:\:\:  i\in A \\
        h_i^A &= \sum_j p_{ij} h_i^A  \:\:\:\:  i \notin A

    Parameters
    ----------
    T : (M, M) ndarray or scipy.sparse matrix
        Transition matrix
    target: array_like
        List of integer state labels for the target set

    Returns
    -------
    h : ndarray(n)
        a vector with hitting probabilities
    """
    T = ensure_number_array(T, ndim=2)
    target = ensure_integer_array(target, ndim=1)
    if _issparse(T):
        _showSparseConversionWarning()  # currently no sparse implementation!
        return dense.hitting_probability.hitting_probability(T.toarray(), target)
    else:
        return dense.hitting_probability.hitting_probability(T, target)


################################################################################
# Transition path theory
################################################################################

def committor(T, A, B, forward=True, mu=None):
    r"""Compute the committor between sets of microstates.

    The committor assigns to each microstate a probability that being
    at this state, the set B will be hit next, rather than set A
    (forward committor), or that the set A has been hit previously
    rather than set B (backward committor). See :footcite:`metzner2009transition` for a
    detailed mathematical description. The present implementation
    uses the equations given in :footcite:`noe2009constructing`.

    Parameters
    ----------
    T : (M, M) ndarray or scipy.sparse matrix
        Transition matrix
    A : array_like
        List of integer state labels for set A
    B : array_like
        List of integer state labels for set B
    forward : bool
        If True compute the forward committor, else
        compute the backward committor.

    Returns
    -------
    q : (M,) ndarray
        Vector of comittor probabilities.

    Notes
    -----
    Committor functions are used to characterize microstates in terms
    of their probability to being visited during a reaction/transition
    between two disjoint regions of state space A, B.

    **Forward committor**

    The forward committor :math:`q^{(+)}_i` is defined as the probability
    that the process starting in `i` will reach `B` first, rather than `A`.

    Using the first hitting time of a set :math:`S`,

    .. math:: T_{S}=\inf\{t \geq 0 | X_t \in S \}

    the forward committor :math:`q^{(+)}_i` can be fromally defined as

    .. math:: q^{(+)}_i=\mathbb{P}_{i}(T_{A}<T_{B}).

    The forward committor solves to the following boundary value problem

    .. math::  \begin{array}{rl} \sum_j L_{ij} q^{(+)}_{j}=0 & i \in X \setminus{(A \cup B)} \\
                q_{i}^{(+)}=0 & i \in A \\
                q_{i}^{(+)}=1 & i \in B
                \end{array}

    :math:`L=T-I` denotes the generator matrix.

    **Backward committor**

    The backward committor is defined as the probability that the process
    starting in :math:`x` came from :math:`A` rather than from :math:`B`.

    Using the last exit time of a set :math:`S`,

    .. math:: t_{S}=\sup\{t \geq 0 | X_t \notin S \}

    the backward committor can be formally defined as

    .. math:: q^{(-)}_i=\mathbb{P}_{i}(t_{A}<t_{B}).

    The backward comittor solves another boundary value problem

    .. math::  \begin{array}{rl}
                \sum_j K_{ij} q^{(-)}_{j}=0 & i \in X \setminus{(A \cup B)} \\
                q_{i}^{(-)}=1 & i \in A \\
                q_{i}^{(-)}=0 & i \in B
                \end{array}

    :math:`K=(D_{\pi}L)^{T}` denotes the adjoint generator matrix.

    References
    ----------
    .. footbibliography::

    Examples
    --------

    >>> import numpy as np
    >>> from deeptime.markov.tools.analysis import committor
    >>> T = np.array([[0.89, 0.1, 0.01], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])
    >>> A = [0]
    >>> B = [2]

    >>> u_plus = committor(T, A, B)
    >>> u_plus
    array([0. , 0.5, 1. ])

    >>> u_minus = committor(T, A, B, forward=False)
    >>> u_minus
    array([1.        , 0.45454545, 0.        ])
    """
    T = ensure_number_array(T, ndim=2)
    A = ensure_integer_array(A, ndim=1)
    B = ensure_integer_array(B, ndim=1)
    if forward:
        return _committor.forward_committor(T, A, B)
    else:
        """ if P is time reversible backward commitor is equal 1 - q+"""
        if is_reversible(T, mu=mu):
            return 1.0 - _committor.forward_committor(T, A, B)

        else:
            return _committor.backward_committor(T, A, B)


################################################################################
# Expectations
################################################################################

def expected_counts(T, p0, N):
    r"""Compute expected transition counts for Markov chain with n steps.

    Parameters
    ----------
    T : (M, M) ndarray or sparse matrix
        Transition matrix
    p0 : (M,) ndarray
        Initial (probability) vector
    N : int
        Number of steps to take

    Returns
    --------
    EC : (M, M) ndarray or sparse matrix
        Expected value for transition counts after N steps

    Notes
    -----
    Expected counts can be computed via the following expression

    .. math::

        \mathbb{E}[C^{(N)}]=\sum_{k=0}^{N-1} \text{diag}(p^{T} T^{k}) T

    Examples
    --------

    >>> import numpy as np
    >>> from deeptime.markov.tools.analysis import expected_counts

    >>> T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])
    >>> p0 = np.array([1.0, 0.0, 0.0])
    >>> N = 100
    >>> EC = expected_counts(T, p0, N)

    >>> EC
    array([[45.44616147,  5.0495735 ,  0.        ],
           [ 4.50413223,  0.        ,  4.50413223],
           [ 0.        ,  4.04960006, 36.44640052]])

    """
    # check input
    T = ensure_number_array(T, ndim=2)
    p0 = ensure_floating_array(p0, ndim=1)
    # go
    return _expectations.expected_counts(p0, T, N)


def expected_counts_stationary(T, N, mu=None):
    r"""Expected transition counts for Markov chain in equilibrium.

    Parameters
    ----------
    T : (M, M) ndarray or sparse matrix
        Transition matrix.
    N : int
        Number of steps for chain.
    mu : (M,) ndarray (optional)
        Stationary distribution for T. If mu is not specified it will be
        computed from T.

    Returns
    -------
    EC : (M, M) ndarray or sparse matrix
        Expected value for transition counts after N steps.

    Notes
    -----
    Since :math:`\mu` is stationary for :math:`T` we have

    .. math::

        \mathbb{E}[C^{(N)}]=N D_{\mu}T.

    :math:`D_{\mu}` is a diagonal matrix. Elements on the diagonal are
    given by the stationary vector :math:`\mu`

    Examples
    --------

    >>> import numpy as np
    >>> from deeptime.markov.tools.analysis import expected_counts_stationary

    >>> T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])
    >>> N = 100
    >>> EC = expected_counts_stationary(T, N)

    >>> EC
    array([[40.90909091,  4.54545455,  0.        ],
           [ 4.54545455,  0.        ,  4.54545455],
           [ 0.        ,  4.54545455, 40.90909091]])

    """
    # check input
    T = ensure_number_array(T, ndim=2)
    if mu is not None:
        mu = ensure_floating_array(mu, ndim=1)
    # go
    return _expectations.expected_counts_stationary(T, N, mu=mu)


################################################################################
# Fingerprints
################################################################################

def fingerprint_correlation(T, obs1, obs2=None, tau=1, k=None, ncv=None):
    r"""Dynamical fingerprint for equilibrium correlation
    experiment. :footcite:`noe2011dynamical`.

    Parameters
    ----------
    T : (M, M) ndarray or scipy.sparse matrix
        Transition matrix
    obs1 : (M,) ndarray
        Observable, represented as vector on state space
    obs2 : (M,) ndarray (optional)
        Second observable, for cross-correlations
    k : int (optional)
        Number of time-scales and amplitudes to compute
    tau : int (optional)
        Lag time of given transition matrix, for correct time-scales
    ncv : int (optional)
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k

    Returns
    -------
    timescales : (N,) ndarray
        Time-scales of the transition matrix
    amplitudes : (N,) ndarray
        Amplitudes for the correlation experiment

    See also
    --------
    correlation, fingerprint_relaxation

    References
    ----------
    .. footbibliography::

    Notes
    -----
    Fingerprints are a combination of time-scale and amplitude spectrum for
    a equilibrium correlation or a non-equilibrium relaxation experiment.

    **Auto-correlation**

    The auto-correlation of an observable :math:`a(x)` for a system in
    equilibrium is

    .. math:: \mathbb{E}_{\mu}[a(x,0)a(x,t)]=\sum_x \mu(x) a(x, 0) a(x, t)

    :math:`a(x,0)=a(x)` is the observable at time :math:`t=0`.  It can
    be propagated forward in time using the t-step transition matrix
    :math:`p^{t}(x, y)`.

    The propagated observable at time :math:`t` is :math:`a(x,
    t)=\sum_y p^t(x, y)a(y, 0)`.

    Using the eigenvlaues and eigenvectors of the transition matrix the autocorrelation
    can be written as

    .. math:: \mathbb{E}_{\mu}[a(x,0)a(x,t)]=\sum_i \lambda_i^t \langle a, r_i\rangle_{\mu} \langle l_i, a \rangle.

    The fingerprint amplitudes :math:`\gamma_i` are given by

    .. math:: \gamma_i=\langle a, r_i\rangle_{\mu} \langle l_i, a \rangle.

    And the fingerprint time scales :math:`t_i` are given by

    .. math:: t_i=-\frac{\tau}{\log \lvert \lambda_i \rvert}.

    **Cross-correlation**

    The cross-correlation of two observables :math:`a(x)`, :math:`b(x)` is similarly given

    .. math:: \mathbb{E}_{\mu}[a(x,0)b(x,t)]=\sum_x \mu(x) a(x, 0) b(x, t)

    The fingerprint amplitudes :math:`\gamma_i` are similarly given in terms of the eigenvectors

    .. math:: \gamma_i=\langle a, r_i\rangle_{\mu} \langle l_i, b \rangle.

    Examples
    --------

    >>> import numpy as np
    >>> from deeptime.markov.tools.analysis import fingerprint_correlation

    >>> T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])
    >>> a = np.array([1.0, 0.0, 0.0])
    >>> ts, amp = fingerprint_correlation(T, a)

    >>> ts
    array([       inf, 9.49122158, 0.43429448])

    >>> amp
    array([0.20661157, 0.22727273, 0.02066116])

    """
    # check if square matrix and remember size
    T = ensure_number_array(T, ndim=2)
    n = T.shape[0]
    # will not do fingerprint analysis for nonreversible matrices
    if not is_reversible(T):
        raise ValueError('Fingerprint calculation is not supported for nonreversible transition matrices. ')
    obs1 = ensure_number_array(obs1, ndim=1, size=n)
    if obs2 is not None:
        obs2 = ensure_number_array(obs2, ndim=1, size=n)
    # go
    return _fingerprints.fingerprint_correlation(T, obs1, obs2=obs2, tau=tau, k=k, ncv=ncv)


def fingerprint_relaxation(T, p0, obs, tau=1, k=None, ncv=None):
    r"""Dynamical fingerprint for relaxation experiment. :footcite:`noe2011dynamical`

    The dynamical fingerprint is given by the implied time-scale
    spectrum together with the corresponding amplitudes.

    Parameters
    ----------
    T : (M, M) ndarray or scipy.sparse matrix
        Transition matrix
    p0 : (M,) ndarray
        Starting distribution.
    obs : (M,) ndarray
        Observable, represented as vector on state space
    k : int (optional)
        Number of time-scales and amplitudes to compute
    tau : int (optional)
        Lag time of given transition matrix, for correct time-scales
    ncv : int (optional)
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k

    Returns
    -------
    timescales : (N,) ndarray
        Time-scales of the transition matrix
    amplitudes : (N,) ndarray
        Amplitudes for the relaxation experiment

    See also
    --------
    relaxation, fingerprint_correlation

    References
    ----------
    .. footbibliography::

    Notes
    -----
    Fingerprints are a combination of time-scale and amplitude spectrum for
    a equilibrium correlation or a non-equilibrium relaxation experiment.

    **Relaxation**

    A relaxation experiment looks at the time dependent expectation
    value of an observable for a system out of equilibrium

    .. math:: \mathbb{E}_{w_{0}}[a(x, t)]=\sum_x w_0(x) a(x, t)=\sum_x w_0(x) \sum_y p^t(x, y) a(y).

    The fingerprint amplitudes :math:`\gamma_i` are given by

    .. math:: \gamma_i=\langle w_0, r_i\rangle \langle l_i, a \rangle.

    And the fingerprint time scales :math:`t_i` are given by

    .. math:: t_i=-\frac{\tau}{\log \lvert \lambda_i \rvert}.

    Examples
    --------

    >>> import numpy as np
    >>> from deeptime.markov.tools.analysis import fingerprint_relaxation

    >>> T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])
    >>> p0 = np.array([1.0, 0.0, 0.0])
    >>> a = np.array([1.0, 0.0, 0.0])
    >>> ts, amp = fingerprint_relaxation(T, p0, a)

    >>> ts
    array([       inf, 9.49122158, 0.43429448])

    >>> amp
    array([0.45454545, 0.5       , 0.04545455])

    """
    # check if square matrix and remember size
    T = ensure_number_array(T, ndim=2)
    n = T.shape[0]
    # will not do fingerprint analysis for nonreversible matrices
    if not is_reversible(T):
        raise ValueError('Fingerprint calculation is not supported for nonreversible transition matrices. ')
    p0 = ensure_number_array(p0, ndim=1, size=n)
    obs = ensure_number_array(obs, ndim=1, size=n)
    # go
    return _fingerprints.fingerprint_relaxation(T, p0, obs, tau=tau, k=k, ncv=ncv)


def expectation(T, a, mu=None):
    r"""Equilibrium expectation value of a given observable.

    Parameters
    ----------
    T : (M, M) ndarray or scipy.sparse matrix
        Transition matrix
    a : (M,) ndarray
        Observable vector
    mu : (M,) ndarray (optional)
        The stationary distribution of T.  If given, the stationary
        distribution will not be recalculated (saving lots of time)

    Returns
    -------
    val: float
        Equilibrium expectation value fo the given observable

    Notes
    -----
    The equilibrium expectation value of an observable a is defined as follows

    .. math::

        \mathbb{E}_{\mu}[a] = \sum_i \mu_i a_i

    :math:`\mu=(\mu_i)` is the stationary vector of the transition matrix :math:`T`.

    Examples
    --------

    >>> import numpy as np
    >>> from deeptime.markov.tools.analysis import expectation

    >>> T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])
    >>> a = np.array([1.0, 0.0, 1.0])
    >>> m_a = expectation(T, a)
    >>> m_a # doctest: +ELLIPSIS
    0.909090909...

    """
    # check if square matrix and remember size
    T = ensure_number_array(T, ndim=2)
    n = T.shape[0]
    a = ensure_number_array(a, ndim=1, size=n)
    if mu is not None:
        mu = ensure_number_array(mu, ndim=1, size=n)
    # go
    if not mu:
        mu = stationary_distribution(T)
    return _np.dot(mu, a)


def correlation(T, obs1, obs2=None, times=(1,), k=None, ncv=None):
    r"""Time-correlation for equilibrium experiment. :footcite:`noe2011dynamical`

    Parameters
    ----------
    T : (M, M) ndarray or scipy.sparse matrix
        Transition matrix
    obs1 : (M,) ndarray
        Observable, represented as vector on state space
    obs2 : (M,) ndarray (optional)
        Second observable, for cross-correlations
    times : array-like of int (optional), default=(1)
        List of times (in tau) at which to compute correlation
    k : int (optional)
        Number of eigenvalues and eigenvectors to use for computation
    ncv : int (optional)
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k

    Returns
    -------
    correlations : ndarray
        Correlation values at given times
    times : ndarray, optional
        time points at which the correlation was computed (if return_times=True)

    References
    ----------
    .. footbibliography::

    Notes
    -----

    **Auto-correlation**

    The auto-correlation of an observable :math:`a(x)` for a system in
    equilibrium is

    .. math:: \mathbb{E}_{\mu}[a(x,0)a(x,t)]=\sum_x \mu(x) a(x, 0) a(x, t)

    :math:`a(x,0)=a(x)` is the observable at time :math:`t=0`.  It can
    be propagated forward in time using the t-step transition matrix
    :math:`p^{t}(x, y)`.

    The propagated observable at time :math:`t` is :math:`a(x,
    t)=\sum_y p^t(x, y)a(y, 0)`.

    Using the eigenvlaues and eigenvectors of the transition matrix
    the autocorrelation can be written as

    .. math:: \mathbb{E}_{\mu}[a(x,0)a(x,t)]=\sum_i \lambda_i^t \langle a, r_i\rangle_{\mu} \langle l_i, a \rangle.

    **Cross-correlation**

    The cross-correlation of two observables :math:`a(x)`,
    :math:`b(x)` is similarly given

    .. math:: \mathbb{E}_{\mu}[a(x,0)b(x,t)]=\sum_x \mu(x) a(x, 0) b(x, t)

    Examples
    --------

    >>> import numpy as np
    >>> from deeptime.markov.tools.analysis import correlation

    >>> T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])
    >>> a = np.array([1.0, 0.0, 0.0])
    >>> times = np.array([1, 5, 10, 20])

    >>> corr = correlation(T, a, times=times)
    >>> corr
    array([0.40909091, 0.34081364, 0.28585667, 0.23424263])

    """
    # check if square matrix and remember size
    T = ensure_number_array(T, ndim=2)
    n = T.shape[0]
    obs1 = ensure_number_array(obs1, ndim=1, size=n)
    if obs2 is not None:
        obs2 = ensure_number_array(obs2, ndim=1, size=n)
    times = ensure_integer_array(times, ndim=1)

    # check input
    # go
    return _fingerprints.correlation(T, obs1, obs2=obs2, times=times, k=k, ncv=ncv)


def relaxation(T, p0, obs, times=(1,), k=None):
    r"""Relaxation experiment. :footcite:`noe2011dynamical`

    The relaxation experiment describes the time-evolution
    of an expectation value starting in a non-equilibrium
    situation.

    Parameters
    ----------
    T : (M, M) ndarray or scipy.sparse matrix
        Transition matrix
    p0 : (M,) ndarray (optional)
        Initial distribution for a relaxation experiment
    obs : (M,) ndarray
        Observable, represented as vector on state space
    times : list of int (optional)
        List of times at which to compute expectation
    k : int (optional)
        Number of eigenvalues and eigenvectors to use for computation

    Returns
    -------
    res : ndarray
        Array of expectation value at given times

    References
    ----------
    .. footbibliography::

    Notes
    -----

    **Relaxation**

    A relaxation experiment looks at the time dependent expectation
    value of an observable for a system out of equilibrium

    .. math:: \mathbb{E}_{w_{0}}[a(x, t)]=\sum_x w_0(x) a(x, t)=\sum_x w_0(x) \sum_y p^t(x, y) a(y).

    Examples
    --------

    >>> import numpy as np
    >>> from deeptime.markov.tools.analysis import relaxation

    >>> T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])
    >>> p0 = np.array([1.0, 0.0, 0.0])
    >>> a = np.array([1.0, 1.0, 0.0])
    >>> times = np.array([1, 5, 10, 20])

    >>> rel = relaxation(T, p0, a, times=times)
    >>> rel
    array([1.        , 0.8407    , 0.71979377, 0.60624287])

    """
    # check if square matrix and remember size
    T = ensure_number_array(T, ndim=2)
    n = T.shape[0]
    p0 = ensure_number_array(p0, ndim=1, size=n)
    obs = ensure_number_array(obs, ndim=1, size=n)
    times = ensure_integer_array(times, ndim=1)
    # go
    return _fingerprints.relaxation(T, p0, obs, k=k, times=times)


# ========================
# PCCA
# ========================

def _pcca_object(T, m):
    """
    Constructs the pcca object from dense or sparse

    Parameters
    ----------
    T : (n, n) ndarray or scipy.sparse matrix
        Transition matrix
    m : int
        Number of metastable sets

    Returns
    -------
    pcca : PCCA
        PCCA object
    """
    if _issparse(T):
        _showSparseConversionWarning()
        T = T.toarray()
    T = ensure_number_array(T, ndim=2, accept_sparse=False)
    return dense.pcca.PCCA(T, m)


def pcca_memberships(T, m):
    r"""Compute meta-stable sets using PCCA++. :footcite:`roblitz2013fuzzy`
    Returns the membership of all states to these sets.

    Parameters
    ----------
    T : (n, n) ndarray or scipy.sparse matrix
        Transition matrix
    m : int
        Number of metastable sets

    Returns
    -------
    clusters : (n, m) ndarray
        Membership vectors. clusters[i, j] contains the membership of state i to metastable state j

    Notes
    -----
    Perron cluster center analysis assigns each microstate a vector of
    membership probabilities. This assignement is performed using the
    right eigenvectors of the transition matrix. Membership
    probabilities are computed via numerical optimization of the
    entries of a membership matrix.

    References
    ----------
    .. footbibliography::
    """
    return _pcca_object(T, m).memberships


def pcca_sets(T, m):
    r""" Computes the metastable sets given transition matrix T using the PCCA++ method _[1]

    This is only recommended for visualization purposes. You *cannot* compute any
    actual quantity of the coarse-grained kinetics without employing the fuzzy memberships!

    Parameters
    ----------
    T : (n, n) ndarray or scipy.sparse matrix
        Transition matrix
    m : int
        Number of metastable sets

    Returns
    -------
    A list of length equal to metastable states. Each element is an array with microstate indexes contained in it

    References
    ----------
    .. [1] Roeblitz, S and M Weber. 2013. Fuzzy spectral clustering by
        PCCA+: application to Markov state models and data
        classification. Advances in Data Analysis and Classification 7
        (2): 147-179
    """
    return _pcca_object(T, m).metastable_sets


def pcca_assignments(T, m):
    """ Computes the assignment to metastable sets for active set states using the PCCA++ method _[1]

    This is only recommended for visualization purposes. You *cannot* compute any
    actual quantity of the coarse-grained kinetics without employing the fuzzy memberships!

    Parameters
    ----------
    m : int
        Number of metastable sets

    Returns
    -------
    For each active set state, the metastable state it is located in.

    References
    ----------
    .. [1] Roeblitz, S and M Weber. 2013. Fuzzy spectral clustering by
        PCCA+: application to Markov state models and data
        classification. Advances in Data Analysis and Classification 7
        (2): 147-179
    """
    return _pcca_object(T, m).metastable_assignment


def pcca_distributions(T, m):
    """ Computes the probability distributions of active set states within each metastable set using the PCCA++ method _[1]
    using Bayesian inversion as described in _[2].

    Parameters
    ----------
    m : int
        Number of metastable sets

    Returns
    -------
    p_out : ndarray( (m, n) )
        A matrix containing the probability distribution of each active set state, given that we are in a
        metastable set.
        i.e. p(state | metastable). The row sums of p_out are 1.

    References
    ----------
    .. [1] Roeblitz, S and M Weber. 2013. Fuzzy spectral clustering by
        PCCA+: application to Markov state models and data
        classification. Advances in Data Analysis and Classification 7
        (2): 147-179
    .. [2] F. Noe, H. Wu, J.-H. Prinz and N. Plattner:
        Projected and hidden Markov models for calculating kinetics and metastable states of complex molecules
        J. Chem. Phys. 139, 184114 (2013)
    """
    return _pcca_object(T, m).output_probabilities


def coarsegrain(P, m):
    """Coarse-grains transition matrix P to n sets using PCCA++ _[1]

    Coarse-grains transition matrix P such that the dominant eigenvalues are preserved, using:

    ..math:
        \tilde{P} = M^T P M (M^T M)^{-1}

    where :math:`M` is the membership probability matrix and P is the full transition matrix.
    See _[2] and _[3] for the theory. The results of the coarse-graining can be interpreted as a hidden markov model
    where the states of the coarse-grained transition matrix are the hidden states. Therefore we additionally return
    the stationary probability of the coarse-grained transition matrix as well as the output probability matrix from
    metastable states to states in order to provide all objects needed for an HMM.

    Returns
    -------
    pi_c : ndarray( (m) )
        Equilibrium probability vector of the coarse-grained transition matrix
    P_c : ndarray( (m, m) )
        Coarse-grained transition matrix
    p_out : ndarray( (m, n) )
        A matrix containing the probability distribution of each active set state, given that we are in a
        metastable set.
        i.e. p(state | metastable). The row sums of p_out are 1.

    References
    ----------
    .. [1] Roeblitz, S and M Weber. 2013. Fuzzy spectral clustering by
        PCCA+: application to Markov state models and data
        classification. Advances in Data Analysis and Classification 7
        (2): 147-179
    .. [2] Kube, S and M Weber.
        A coarse-graining method for the identification of transition rates between molecular conformations
        J. Chem. Phys. 126, 024103 (2007)
    .. [2] F. Noe, H. Wu, J.-H. Prinz and N. Plattner:
        Projected and hidden Markov models for calculating kinetics and metastable states of complex molecules
        J. Chem. Phys. 139, 184114 (2013)
    """
    return _pcca_object(P, m).coarse_grained_transition_matrix


################################################################################
# Sensitivities
################################################################################

def _showSparseConversionWarning():
    msg = ("Converting input to dense, since this method is currently only implemented for dense arrays")
    warnings.warn(msg, UserWarning)


def eigenvalue_sensitivity(T, k):
    r"""Sensitivity matrix of a specified eigenvalue.

    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix
    k : int
        Compute sensitivity matrix for k-th eigenvalue

    Returns
    -------
    S : (M, M) ndarray
        Sensitivity matrix for k-th eigenvalue.

    """
    T = ensure_number_array(T, ndim=2)
    if _issparse(T):
        _showSparseConversionWarning()
        eigenvalue_sensitivity(T.todense(), k)
    else:
        return dense.sensitivity.eigenvalue_sensitivity(T, k)


def timescale_sensitivity(T, k):
    r"""Sensitivity matrix of a specified time-scale.

    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix
    k : int
        Compute sensitivity matrix for the k-th time-scale.

    Returns
    -------
    S : (M, M) ndarray
        Sensitivity matrix for the k-th time-scale.

    """
    T = ensure_number_array(T, ndim=2)
    if _issparse(T):
        _showSparseConversionWarning()
        timescale_sensitivity(T.todense(), k)
    else:
        return dense.sensitivity.timescale_sensitivity(T, k)


def eigenvector_sensitivity(T, k, j, right=True):
    r"""Sensitivity matrix of a selected eigenvector element.

    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix (stochastic matrix).
    k : int
        Eigenvector index
    j : int
        Element index
    right : bool
        If True compute for right eigenvector, otherwise compute for left eigenvector.

    Returns
    -------
    S : (M, M) ndarray
        Sensitivity matrix for the j-th element of the k-th eigenvector.

    """
    T = ensure_number_array(T, ndim=2)
    if _issparse(T):
        _showSparseConversionWarning()
        eigenvector_sensitivity(T.todense(), k, j, right=right)
    else:
        return dense.sensitivity.eigenvector_sensitivity(T, k, j, right=right)


def stationary_distribution_sensitivity(T, j):
    r"""Sensitivity matrix of a stationary distribution element.

    Parameters
    ----------
    T : (M, M) ndarray
       Transition matrix (stochastic matrix).
    j : int
        Index of stationary distribution element
        for which sensitivity matrix is computed.


    Returns
    -------
    S : (M, M) ndarray
        Sensitivity matrix for the specified element
        of the stationary distribution.

    """
    T = ensure_number_array(T, ndim=2)
    if _issparse(T):
        _showSparseConversionWarning()
        stationary_distribution_sensitivity(T.todense(), j)
    else:
        return dense.sensitivity.stationary_distribution_sensitivity(T, j)


def mfpt_sensitivity(T, target, i):
    r"""Sensitivity matrix of the mean first-passage time from specified state.

    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix
    target : int or list
        Target state or set for mfpt computation
    i : int
        Compute the sensitivity for state `i`

    Returns
    -------
    S : (M, M) ndarray
        Sensitivity matrix for specified state

    """
    # check input
    T = ensure_number_array(T, ndim=2)
    target = ensure_integer_array(target, ndim=1)
    # go
    if _issparse(T):
        _showSparseConversionWarning()
        mfpt_sensitivity(T.todense(), target, i)
    else:
        return dense.sensitivity.mfpt_sensitivity(T, target, i)


def committor_sensitivity(T, A, B, i, forward=True):
    r"""Sensitivity matrix of a specified committor entry.

    Parameters
    ----------

    T : (M, M) ndarray
        Transition matrix
    A : array_like
        List of integer state labels for set A
    B : array_like
        List of integer state labels for set B
    i : int
        Compute the sensitivity for committor entry `i`
    forward : bool (optional)
        Compute the forward committor. If forward
        is False compute the backward committor.

    Returns
    -------
    S : (M, M) ndarray
        Sensitivity matrix of the specified committor entry.

    """
    # check inputs
    T = ensure_number_array(T, ndim=2)
    A = ensure_integer_array(A, ndim=1)
    B = ensure_integer_array(B, ndim=1)
    if _issparse(T):
        _showSparseConversionWarning()
        committor_sensitivity(T.todense(), A, B, i, forward)
    else:
        if forward:
            return dense.sensitivity.forward_committor_sensitivity(T, A, B, i)
        else:
            return dense.sensitivity.backward_committor_sensitivity(T, A, B, i)


def expectation_sensitivity(T, a):
    r"""Sensitivity of expectation value of observable A=(a_i).

    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix
    a : (M,) ndarray
        Observable, a[i] is the value of the observable at state i.

    Returns
    -------
    S : (M, M) ndarray
        Sensitivity matrix of the expectation value.

    """
    # check input
    T = ensure_number_array(T, ndim=2)
    a = ensure_floating_array(a, ndim=1)
    # go
    if _issparse(T):
        _showSparseConversionWarning()
        return dense.sensitivity.expectation_sensitivity(T.toarray(), a)
    else:
        return dense.sensitivity.expectation_sensitivity(T, a)
