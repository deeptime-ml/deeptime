import numpy as np
import scipy.linalg as scl
import scipy.sparse

from deeptime.numeric import sort_eigs

__all__ = ['bootstrapping_count_matrix', 'bootstrapping_dtrajs', 'twostep_count_matrix', 'rank_decision',
           'oom_components', 'equilibrium_transition_matrix']

from deeptime.util.matrix import submatrix


def bootstrapping_dtrajs(dtrajs, lag, N_full, nbs=10000, active_set=None):
    """
    Perform trajectory based re-sampling.

    Parameters
    ----------
    dtrajs : list of discrete trajectories

    lag : int
        lag time

    N_full : int
        Number of states in discrete trajectories.
    nbs : int, optional
        Number of bootstrapping samples
    active_set : ndarray
        Indices of active set, all count matrices will be restricted
        to active set.

    Returns
    -------
    smean : ndarray(N,)
        mean values of singular values
    sdev : ndarray(N,)
        standard deviations of singular values
    """

    # Get the number of simulations:
    Q = len(dtrajs)
    # Get the number of states in the active set:
    if active_set is not None:
        N = active_set.size
    else:
        N = N_full
    # Build up a matrix of count matrices for each simulation. Size is Q*N^2:
    traj_ind = []
    state1 = []
    state2 = []
    q = 0
    for traj in dtrajs:
        traj_ind.append(q * np.ones(traj[:-lag].size))
        state1.append(traj[:-lag])
        state2.append(traj[lag:])
        q += 1
    traj_inds = np.concatenate(traj_ind)
    pairs = N_full * np.concatenate(state1) + np.concatenate(state2)
    data = np.ones(pairs.size)
    Ct_traj = scipy.sparse.coo_matrix((data, (traj_inds, pairs)), shape=(Q, N_full * N_full))
    Ct_traj = Ct_traj.tocsr()

    # Perform re-sampling:
    svals = np.zeros((nbs, N))
    for s in range(nbs):
        # Choose selection:
        sel = np.random.choice(Q, Q, replace=True)
        # Compute count matrix for selection:
        Ct_sel = Ct_traj[sel, :].sum(axis=0)
        Ct_sel = np.asarray(Ct_sel).reshape((N_full, N_full))
        if active_set is not None:
            Ct_sel = submatrix(Ct_sel, active_set)
        svals[s, :] = scl.svdvals(Ct_sel)
    # Compute mean and uncertainties:
    smean = np.mean(svals, axis=0)
    sdev = np.std(svals, axis=0)

    return smean, sdev


def bootstrapping_count_matrix(Ct, nbs=10000):
    """
    Perform bootstrapping on trajectories to estimate uncertainties for singular values of count matrices.

    Parameters
    ----------
    Ct : csr-matrix
        count matrix of the data.

    nbs : int, optional
        the number of re-samplings to be drawn from dtrajs

    Returns
    -------
    smean : ndarray(N,)
        mean values of singular values
    sdev : ndarray(N,)
        standard deviations of singular values
    """
    # Get the number of states:
    N = Ct.shape[0]
    # Get the number of transition pairs:
    T = Ct.sum()
    # Reshape and normalize the count matrix:
    p = Ct.toarray()
    p = np.reshape(p, (N * N,)).astype(np.float)
    p = p / T
    # Perform the bootstrapping:
    svals = np.zeros((nbs, N))
    for s in range(nbs):
        # Draw sample:
        sel = np.random.multinomial(T, p)
        # Compute the count-matrix:
        sC = np.reshape(sel, (N, N))
        # Compute singular values:
        svals[s, :] = scl.svdvals(sC)
    # Compute mean and uncertainties:
    smean = np.mean(svals, axis=0)
    sdev = np.std(svals, axis=0)

    return smean, sdev


def twostep_count_matrix(dtrajs, lag, N):
    """
    Compute all two-step count matrices from discrete trajectories.

    Parameters
    ----------
    dtrajs : list of discrete trajectories

    lag : int
        the lag time for count matrix estimation
    N : int
        the number of states in the discrete trajectories.

    Returns
    -------
    C2t : sparse csc-matrix (N, N, N)
        two-step count matrices for all states. C2t[:, n, :] is a count matrix for each n

    """
    # List all transition triples:
    rows = []
    cols = []
    states = []
    for dtraj in dtrajs:
        if dtraj.size > 2 * lag:
            rows.append(dtraj[0:-2 * lag])
            states.append(dtraj[lag:-lag])
            cols.append(dtraj[2 * lag:])
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    state = np.concatenate(states)
    data = np.ones(row.size)
    # Transform the rows and cols into a single list with N*+2 possible values:
    pair = N * row + col
    # Estimate sparse matrix:
    C2t = scipy.sparse.coo_matrix((data, (pair, state)), shape=(N * N, N))

    return C2t.tocsc()


def rank_decision(smean, sdev, tol=10.0):
    """
    Rank decision based on uncertainties of singular values.

    Parameters
    ----------
    smean : ndarray(N,)
        mean values of singular values for Ct
    sdev : ndarray(N,)
        standard errors of singular values for Ct
    tol : float, optional default=10.0
        accept singular values with signal-to-noise ratio >= tol_svd

    Returns
    -------
    ind : ndarray(N, dtype=bool)
        indicates which singular values are accepted.

    """
    # Determine signal-to-noise ratios of singular values:
    sratio = smean / sdev
    # Return Boolean array of accepted singular values:
    return sratio >= tol


def oom_components(Ct, C2t, rank_ind=None, lcc=None, tol_one=1e-2):
    """
    Compute OOM components and eigenvalues from count matrices:

    Parameters
    ----------
    Ct : ndarray(N, N)
        count matrix from data
    C2t : sparse csc-matrix (N*N, N)
        two-step count matrix from data for all states, columns enumerate
        intermediate steps.
    rank_ind : ndarray(N, dtype=bool), optional, default=None
        indicates which singular values are accepted. By default, all non-
        zero singular values are accepted.
    lcc : ndarray(N,), optional, default=None
        largest connected set of the count-matrix. Two step count matrix
        will be reduced to this set.
    tol_one : float, optional, default=1e-2
        keep eigenvalues of absolute value less or equal 1+tol_one.

    Returns
    -------
    Xi : ndarray(M, N, M)
        matrix of set-observable operators
    oom_information_state_vector: ndarray(M,)
        information state vector of OOM
    oom_evaluator : ndarray(M,)
        evaluator of OOM
    l : ndarray(M,)
        eigenvalues from OOM
    """
    import deeptime.markov.tools.estimation as me
    # Decompose count matrix by SVD:
    if lcc is not None:
        Ct_svd = me.largest_connected_submatrix(Ct, lcc=lcc)
        N1 = Ct.shape[0]
    else:
        Ct_svd = Ct
    V, s, W = scl.svd(Ct_svd, full_matrices=False)
    # Make rank decision:
    if rank_ind is None:
        ind = (s >= np.finfo(float).eps)
    V = V[:, rank_ind]
    s = s[rank_ind]
    W = W[rank_ind, :].T

    # Compute transformations:
    F1 = np.dot(V, np.diag(s ** -0.5))
    F2 = np.dot(W, np.diag(s ** -0.5))

    # Apply the transformations to C2t:
    N = Ct_svd.shape[0]
    M = F1.shape[1]
    Xi = np.zeros((M, N, M))
    for n in range(N):
        if lcc is not None:
            C2t_n = C2t[:, lcc[n]]
            C2t_n = _reshape_sparse(C2t_n, (N1, N1))
            C2t_n = me.largest_connected_submatrix(C2t_n, lcc=lcc)
        else:
            C2t_n = C2t[:, n]
            C2t_n = _reshape_sparse(C2t_n, (N, N))
        Xi[:, n, :] = np.dot(F1.T, C2t_n.dot(F2))

    # Compute sigma:
    c = np.sum(Ct_svd, axis=1)
    sigma = np.dot(F1.T, c)
    # Compute eigenvalues:
    Xi_S = np.sum(Xi, axis=1)
    eigenvalues, right_eigenvectors = scl.eig(Xi_S.T)
    # Restrict eigenvalues to reasonable range:
    ind = np.where(np.logical_and(np.abs(eigenvalues) <= (1 + tol_one), np.real(eigenvalues) >= 0.0))[0]
    eigenvalues = eigenvalues[ind]
    right_eigenvectors = right_eigenvectors[:, ind]
    # Sort and extract omega
    eigenvalues, right_eigenvectors = sort_eigs(eigenvalues, right_eigenvectors)
    omega = np.real(right_eigenvectors[:, 0])
    omega = omega / np.dot(omega, sigma)

    return Xi, omega, sigma, eigenvalues


def equilibrium_transition_matrix(Xi, omega, sigma, reversible=True, return_lcc=True):
    """
    Compute equilibrium transition matrix from OOM components:

    Parameters
    ----------
    Xi : ndarray(M, N, M)
        matrix of set-observable operators
    omega: ndarray(M,)
        information state vector of OOM
    sigma : ndarray(M,)
        evaluator of OOM
    reversible : bool, optional, default=True
        symmetrize corrected count matrix in order to obtain
        a reversible transition matrix.
    return_lcc: bool, optional, default=True
        return indices of largest connected set.

    Returns
    -------
    Tt_Eq : ndarray(N, N)
        equilibrium transition matrix
    lcc : ndarray(M,)
        the largest connected set of the transition matrix.
    """
    import deeptime.markov.tools.estimation as me

    # Compute equilibrium transition matrix:
    Ct_Eq = np.einsum('j,jkl,lmn,n->km', omega, Xi, Xi, sigma)
    # Remove negative entries:
    Ct_Eq[Ct_Eq < 0.0] = 0.0
    # Compute transition matrix after symmetrization:
    pi_r = np.sum(Ct_Eq, axis=1)
    if reversible:
        pi_c = np.sum(Ct_Eq, axis=0)
        pi_sym = pi_r + pi_c
        # Avoid zero row-sums. States with zero row-sums will be eliminated by active set update.
        ind0 = np.where(pi_sym == 0.0)[0]
        pi_sym[ind0] = 1.0
        Tt_Eq = (Ct_Eq + Ct_Eq.T) / pi_sym[:, None]
    else:
        # Avoid zero row-sums. States with zero row-sums will be eliminated by active set update.
        ind0 = np.where(pi_r == 0.0)[0]
        pi_r[ind0] = 1.0
        Tt_Eq = Ct_Eq / pi_r[:, None]

    # Perform active set update:
    lcc = me.largest_connected_set(Tt_Eq)
    Tt_Eq = me.largest_connected_submatrix(Tt_Eq, lcc=lcc)

    if return_lcc:
        return Tt_Eq, lcc
    else:
        return Tt_Eq


def _reshape_sparse(A, shape):
    nrows, ncols = A.shape
    if nrows * ncols != shape[0] * shape[1]:
        raise ValueError("Matrix dimensions must agree.")
    rows, cols = A.nonzero()
    flat_indices = rows * ncols + cols
    newrows, newcols = divmod(flat_indices, shape[1])
    data = np.array(A[A.nonzero()].tolist()).flatten()
    Anew = scipy.sparse.csc_matrix((data, (newrows, newcols)), shape=shape)

    return Anew
