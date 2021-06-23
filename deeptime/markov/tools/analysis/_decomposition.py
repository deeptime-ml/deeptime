r"""This module provides matrix-decomposition based functions for the
analysis of stochastic matrices

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>
.. moduleauthor:: M.Hoffmann
"""

import numpy as np
import scipy.sparse as sparse
import scipy.linalg as la
import scipy.sparse.linalg as sla
import warnings

from deeptime.util.exceptions import ImaginaryEigenValueWarning, SpectralWarning

from ._stationary_vector import stationary_distribution
from ._assessment import is_reversible


def _symmetrize(T, mu=None):
    if mu is None:
        mu = stationary_distribution(T)
    if np.any(mu <= 0):
        raise ValueError('Cannot symmetrize transition matrix')
    """ symmetrize T """
    smu = np.sqrt(mu)
    if sparse.issparse(T):
        D = sparse.diags(smu, 0)
        Dinv = sparse.diags(1.0 / smu, 0)
        S = (D.dot(T)).dot(Dinv)
    else:
        S = smu[:, None] * T / smu
    return smu, S


def eigenvalues(T, k=None, ncv=None, reversible=False, mu=None):
    r"""Compute the eigenvalues of a sparse transition matrix.

    Parameters
    ----------
    T : (M, M) scipy.sparse matrix or ndarray
        Transition matrix
    k : int, optional
        Number of eigenvalues to compute.
    ncv : int, optional
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k
    reversible : bool, optional
        Indicate that transition matrix is reversible
    mu : (M,) ndarray, optional
        Stationary distribution of T

    Returns
    -------
    v : (k,) ndarray
        Eigenvalues of T

    Notes
    -----
    The first k eigenvalues of largest magnitude are computed.

    If reversible=True the the eigenvalues of the similar symmetric
    matrix `\sqrt(\mu_i / \mu_j) p_{ij}` will be computed.

    The precomputed stationary distribution will only be used if
    reversible=True.

    """
    is_sparse = sparse.issparse(T)
    if is_sparse and k is None:
        raise ValueError("Number of eigenvalues required for decomposition of sparse matrix")
    else:
        if reversible:
            try:
                v = eigenvalues_rev(T, k, ncv=ncv, mu=mu)
            except ValueError:  # use fallback code, but cast to real
                if is_sparse:
                    v = sla.eigs(T, k=k, which='LM', return_eigenvectors=False, ncv=ncv).real
                else:
                    v = la.eigvals(T).real
        else:
            # nonreversible
            if is_sparse:
                v = sla.eigs(T, k=k, which='LM', return_eigenvectors=False, ncv=ncv)
            else:
                v = la.eigvals(T)

    ind = np.argsort(np.abs(v))[::-1][:k]  # top k
    return v[ind]


def eigenvalues_rev(T, k, ncv=None, mu=None):
    r"""Compute the eigenvalues of a reversible, sparse transition matrix.

    Parameters
    ----------
    T : (M, M) scipy.sparse matrix
        Transition matrix
    k : int
        Number of eigenvalues to compute.
    ncv : int, optional
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k
    mu : (M,) ndarray, optional
        Stationary distribution of T

    Returns
    -------
    v : (k,) ndarray
        Eigenvalues of T

    Raises
    ------
    ValueError
        If stationary distribution is nonpositive.

    Notes
    -----
    The first k eigenvalues of largest magnitude are computed.

    """

    _, S = _symmetrize(T, mu=mu)
    """Compute eigenvalues using a solver for symmetric/hermititan eigenproblems"""
    if sparse.issparse(T):
        evals = sla.eigsh(S, k=k, ncv=ncv, which='LM', return_eigenvectors=False)
    else:
        evals = la.eigvalsh(S)
    return evals


def eigenvectors(T, k=None, right=True, reversible=False, mu=None, ncv=None):
    r"""Compute eigenvectors of transition matrix.

    Parameters
    ----------
    T : (d, d) ndarray
        Transition matrix (stochastic matrix)
    k : int or tuple of ints, optional
        Compute the first k eigenvalues of T
    right : bool, optional
        If right=True compute right eigenvectors, left eigenvectors
        otherwise
    reversible : bool, optional
        Indicate that transition matrix is reversible
    mu : (d,) ndarray, optional
        Stationary distribution of T

    Returns
    -------
    eigvec : (d, n) ndarray
        The eigenvectors of T ordered with decreasing absolute value
        of the corresponding eigenvalue. If k is None then n=d, if k
        is int then n=k otherwise n is the length of the given tuple
        of eigenvector indices

    Notes
    -----
    Eigenvectors are computed using the numpy.linalg interface for the
    corresponding LAPACK routines.

    If reversible=True the the eigenvectors of the similar symmetric
    matrix `\sqrt(\mu_i / \mu_j) p_{ij}` will be used to compute the
    eigenvectors of T.

    The precomputed stationary distribution will only be used if
    reversible=True.

    """
    is_sparse = sparse.issparse(T)

    if is_sparse and k is None:
        raise ValueError("Number of eigenvectors required for decomposition of sparse matrix")

    if reversible:
        smu, S = _symmetrize(T, mu=mu)
        if is_sparse:
            val, vecs = sla.eigsh(S, k=k, ncv=ncv, which='LM', return_eigenvectors=True)
        else:
            val, vecs = la.eigh(S)
        """Sort eigenvectors"""
        perm = np.argsort(np.abs(val))[::-1][:k]
        eigvec = vecs[:, perm]
        if right:
            eigvec /= smu[:, None]
        else:
            eigvec *= smu[:, None]
    else:
        if not is_sparse:
            val, vecs = la.eig(T, left=not right, right=right)
        else:
            val, vecs = sla.eigs(T if right else T.transpose(), k=k, which='LM', ncv=ncv)
        perm = np.argsort(np.abs(val))[::-1][:k]
        eigvec = vecs[:, perm]

    return eigvec


def rdl_decomposition(T, k=None, ncv=None, reversible=False, norm='standard', mu=None):
    r"""Compute the decomposition into left and right eigenvectors.

    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix
    k : int (optional)
        Number of eigenvector/eigenvalue pairs
    norm: {'standard', 'reversible', 'auto'}
        standard: (L'R) = Id, L[:,0] is a probability distribution,
            the stationary distribution mu of T. Right eigenvectors
            R have a 2-norm of 1.
        reversible: R and L are related via L=L[:,0]*R.
        auto: will be reversible if T is reversible, otherwise standard
    reversible : bool, optional
        Indicate that transition matrix is reversible
    mu : (d,) ndarray, optional
        Stationary distribution of T

    Returns
    -------
    R : (M, M) ndarray
        The normalized (with respect to L) right eigenvectors, such that the
        column R[:,i] is the right eigenvector corresponding to the eigenvalue
        w[i], dot(T,R[:,i])=w[i]*R[:,i]
    D : (M, M) ndarray
        A diagonal matrix containing the eigenvalues, each repeated
        according to its multiplicity
    L : (M, M) ndarray
        The normalized (with respect to `R`) left eigenvectors, such that the
        row ``L[i, :]`` is the left eigenvector corresponding to the eigenvalue
        ``w[i]``, ``dot(L[i, :], T)``=``w[i]*L[i, :]``

    Notes
    -----
    If reversible=True the the eigenvalues and eigenvectors of the
    similar symmetric matrix `\sqrt(\mu_i / \mu_j) p_{ij}` will be
    used to compute the eigenvalues and eigenvectors of T.

    The precomputed stationary distribution will only be used if
    reversible=True.

    """
    if sparse.issparse(T) and k is None:
        raise ValueError("Number of eigenvectors required for decomposition of sparse matrix")
    # auto-set norm
    if norm == 'auto':
        if is_reversible(T):
            norm = 'reversible'
        else:
            norm = 'standard'

    if reversible:
        R, D, L = rdl_decomposition_rev(T, norm=norm, mu=mu, k=k, ncv=ncv)
    else:
        R, D, L = rdl_decomposition_nrev(T, norm=norm, k=k, ncv=ncv)

    if reversible or norm == 'reversible':
        D = D.real

    return R[:, :k], D[:k, :k], L[:k, :]


def rdl_decomposition_nrev(T, norm='standard', k=None, ncv=None):
    r"""Decomposition into left and right eigenvectors.

    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix
    norm: {'standard', 'reversible'}
        standard: (L'R) = Id, L[:,0] is a probability distribution,
            the stationary distribution mu of T. Right eigenvectors
            R have a 2-norm of 1
        reversible: R and L are related via L=L[:,0]*R

    Returns
    -------
    R : (M, M) ndarray
        The normalized (with respect to L) right eigenvectors, such that the
        column R[:,i] is the right eigenvector corresponding to the eigenvalue
        w[i], dot(T,R[:,i])=w[i]*R[:,i]
    D : (M, M) ndarray
        A diagonal matrix containing the eigenvalues, each repeated
        according to its multiplicity
    L : (M, M) ndarray
        The normalized (with respect to `R`) left eigenvectors, such that the
        row ``L[i, :]`` is the left eigenvector corresponding to the eigenvalue
        ``w[i]``, ``dot(L[i, :], T)``=``w[i]*L[i, :]``


    """
    is_sparse = sparse.issparse(T)
    d = T.shape[0]
    if is_sparse:
        v, R = sla.eigs(T, k=k, which='LM', ncv=ncv)
    else:
        v, R = la.eig(T)

    """Sort by decreasing magnitude of eigenvalue"""
    ind = np.argsort(np.abs(v))[::-1]
    v = v[ind]
    R = R[:, ind]

    """Diagonal matrix containing eigenvalues"""
    D = np.diag(v)

    # Standard norm: Euclidean norm is 1 for r and LR = I.
    if norm == 'standard':
        if is_sparse:
            r, L = sla.eigs(T.transpose(), k=k, which='LM', ncv=ncv)
            ind = np.argsort(np.abs(r))[::-1]
            L = L[:, ind]
        else:
            L = la.solve(np.transpose(R), np.eye(d))

        """l1- normalization of L[:, 0]"""
        R[:, 0] = R[:, 0] * np.sum(L[:, 0])
        L[:, 0] = L[:, 0] / np.sum(L[:, 0])

        if is_sparse:  # in dense case we already got this
            """Standard normalization L'R=Id"""
            ov = np.diag(np.dot(np.transpose(L), R))
            R = R / ov[np.newaxis, :]

        return R, D, np.transpose(L)

    # Reversible norm:
    elif norm == 'reversible':
        if is_sparse:
            mu = stationary_distribution(T)
        else:
            b = np.zeros(d)
            b[0] = 1.0

            A = np.transpose(R)
            nu = la.solve(A, b)
            mu = nu / np.sum(nu)

        """Ensure that R[:,0] is positive"""
        R[:, 0] = R[:, 0] / np.sign(R[0, 0])

        """Use mu to connect L and R"""
        L = mu[:, np.newaxis] * R

        """Compute overlap"""
        s = np.diag(np.dot(np.transpose(L), R))

        """Renormalize left-and right eigenvectors to ensure L'R=Id"""
        R = R / np.sqrt(s[np.newaxis, :])
        L = L / np.sqrt(s[np.newaxis, :])

        return R, D, np.transpose(L)

    else:
        raise ValueError("Keyword 'norm' has to be either 'standard' or 'reversible'")


def rdl_decomposition_rev(T, norm='reversible', mu=None, ncv=None, k=None):
    r"""Decomposition into left and right eigenvectors for reversible
    transition matrices.

    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix
    norm: {'standard', 'reversible'}
        standard: (L'R) = Id, L[:,0] is a probability distribution,
            the stationary distribution mu of T. Right eigenvectors
            R have a 2-norm of 1.
        reversible: R and L are related via L=L[:,0]*R.
    mu : (M,) ndarray, optional
        Stationary distribution of T

    Returns
    -------
    R : (M, M) ndarray
        The normalized (with respect to L) right eigenvectors, such that the
        column R[:,i] is the right eigenvector corresponding to the eigenvalue
        w[i], dot(T,R[:,i])=w[i]*R[:,i]
    D : (M, M) ndarray
        A diagonal matrix containing the eigenvalues, each repeated
        according to its multiplicity
    L : (M, M) ndarray
        The normalized (with respect to `R`) left eigenvectors, such that the
        row ``L[i, :]`` is the left eigenvector corresponding to the eigenvalue
        ``w[i]``, ``dot(L[i, :], T)``=``w[i]*L[i, :]``

    Notes
    -----
    The eigenvalues and eigenvectors of the similar symmetric matrix
    `\sqrt(\mu_i / \mu_j) p_{ij}` will be used to compute the
    eigenvalues and eigenvectors of T.

    The stationay distribution will be computed if no precomputed stationary
    distribution is given.

    """
    """ symmetrize T """
    is_sparse = sparse.issparse(T)
    smu, S = _symmetrize(T, mu)
    if is_sparse:
        val, eigvec = sla.eigsh(S, k=k, ncv=ncv, which='LM', return_eigenvectors=True)
    else:
        val, eigvec = la.eigh(S)

    """Sort eigenvalues and eigenvectors"""
    perm = np.argsort(np.abs(val))[::-1]
    val = val[perm]
    eigvec = eigvec[:, perm]

    """Diagonal matrix of eigenvalues"""
    D = np.diag(val)

    """Right and left eigenvectors"""
    R = eigvec / smu[:, np.newaxis]
    L = eigvec * smu[:, np.newaxis]

    """Ensure that R[:,0] is positive and unity"""
    tmp = R[0, 0]
    R[:, 0] = R[:, 0] / tmp

    """Ensure that L[:, 0] is probability vector"""
    L[:, 0] = L[:, 0] * tmp

    if norm == 'reversible':
        return R, D, L.T
    elif norm == 'standard':
        """Standard l2-norm of right eigenvectors"""
        w = np.diag(np.dot(R.T, R))
        sw = np.sqrt(w)
        """Don't change normalization of eigenvectors for dominant eigenvalue"""
        sw[0] = 1.0

        R = R / sw[np.newaxis, :]
        L = L * sw[np.newaxis, :]
        return R, D, L.T
    else:
        raise ValueError("Keyword 'norm' has to be either 'standard' or 'reversible'")


def timescales(T, tau=1, k=None, ncv=None, reversible=False, mu=None):
    r"""Compute implied time scales of given transition matrix

    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix
    tau : int, optional
        lag time
    k : int, optional
        Number of time scales
    reversible : bool, optional
        Indicate that transition matirx is reversible
    mu : (M,) ndarray, optional
        Stationary distribution of T

    Returns
    -------
    ts : (N,) ndarray
        Implied time scales of the transition matrix.
        If k=None then N=M else N=k

    Notes
    -----
    If reversible=True the the eigenvalues of the similar symmetric
    matrix `\sqrt(\mu_i / \mu_j) p_{ij}` will be computed.

    The precomputed stationary distribution will only be used if
    reversible=True.

    """
    if sparse.issparse(T) and k is None:
        raise ValueError("Number of time scales required for decomposition of sparse matrix")
    values = eigenvalues(T, reversible=reversible, mu=mu, k=k, ncv=ncv)
    return timescales_from_eigenvalues(values, tau)


def timescales_from_eigenvalues(evals, tau=1):
    r"""Compute implied time scales from given eigenvalues

    Parameters
    ----------
    evals : eigenvalues
    tau : lag time

    Returns
    -------
    ts : ndarray
        The implied time scales to the given eigenvalues, in the same order.

    """

    """Check for dominant eigenvalues with large imaginary part"""

    if not np.allclose(evals.imag, 0.0):
        warnings.warn('Using eigenvalues with non-zero imaginary part', ImaginaryEigenValueWarning)

    """Check for multiple eigenvalues of magnitude one"""
    ind_abs_one = np.isclose(np.abs(evals), 1.0, rtol=0.0, atol=1e-14)
    if sum(ind_abs_one) > 1:
        warnings.warn('Multiple eigenvalues with magnitude one.', SpectralWarning)

    """Compute implied time scales"""
    ts = np.zeros(len(evals))

    """Eigenvalues of magnitude one imply infinite timescale"""
    ts[ind_abs_one] = np.inf

    """All other eigenvalues give rise to finite timescales"""
    ts[np.logical_not(ind_abs_one)] = \
        -1.0 * tau / np.log(np.abs(evals[np.logical_not(ind_abs_one)]))
    return ts
