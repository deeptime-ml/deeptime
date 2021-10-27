import numpy as _np
import scipy as _sp
import scipy.sparse.linalg

__author__ = 'noe, clonker'


class ZeroRankError(_np.linalg.LinAlgError):
    """Input matrix has rank zero."""
    pass


def sort_eigs(evals, evecs, order='magnitude'):
    r""" Sorts the eigenvalues and eigenvectors by descending norm of the eigenvalues or lexicographically.

    Parameters
    ----------
    evals: (n, ) ndarray
        eigenvalues
    evecs: (k, n) ndarray
        eigenvectors in a column matrix
    order : str, default='magnitude'
        The order. Sorts by magnitude by default, can also be 'lexicographic' in which case it sorts lexicographically.

    Returns
    -------
    (evals, evecs) : (n, ) ndarray, (k, n) ndarray
        the sorted eigenvalues and eigenvectors
    """
    assert order in sort_eigs.supported_orders, f"Invalid order {order}, must be one of {sort_eigs.supported_orders}"
    sort_key = evals if order != 'magnitude' else _np.abs(evals)  # magnitude or plain depending on order
    order = _np.argsort(sort_key)[::-1]  # inverted argsort so that order is descending
    return evals[order], evecs[:, order]


sort_eigs.supported_orders = 'magnitude', 'lexicographic'  #: The order in which eigenvalues can be sorted.


def spd_truncated_svd(mat, dim=None, eps=0.):
    r""" Rank-reduced singular value decomposition of symmetric positive (semi-)definite matrix.
    The method yields for a matrix :math:`A\in\mathbb{R}^{n\times n}` singular values :math:`s\in\mathbb{R}^d`
    and singular vectors :math:`U\in\mathbb{R}^{n\times d}` so that `A \approx U\mathrm{diag}(x)U^\top`.

    All the negligible components are removed from the spectrum. In case of `dim` being specified it keeps
    at most the `dim` dominant components but may remove even more, depending on the input matrix.

    Eps influences the tolerance under which components are deemed negligible. In particular, if the product of
    `eps` and the largest singular value is larger than a component of the spectrum, it is removed.

    Parameters
    ----------
    mat : (n, n) ndarray
        Input matrix.
    dim : int, optional, default=None
        The dimension.
    eps : float, optional, default = 0
        Tolerance.

    Returns
    -------
    s : (k, ) ndarray
        Leading singular values.
    U : (n, k) ndarray
        Leading singular vectors.
    """
    if dim is None:
        dim = _np.inf
    dim = min(dim, mat.shape[0])
    S, U = scipy.linalg.schur(mat)
    s = _np.diag(S)
    max_sv = _np.abs(s).max()
    tol = mat.shape[0] * _np.spacing(max_sv)
    mi = _np.min(s)
    if mi < 0 and -mi > tol:
        tol = -mi
    tol = _np.maximum(tol, max_sv * eps)
    dim = min(dim, _np.count_nonzero(s > tol))
    idx = s.argsort()[::-1][:dim]
    return s[idx], U[:, idx]


def spd_eig(W, epsilon=1e-10, method='QR', canonical_signs=False, check_sym: bool = False):
    """ Rank-reduced eigenvalue decomposition of symmetric positive definite matrix.

    Removes all negligible eigenvalues

    Parameters
    ----------
    W : ndarray((n, n), dtype=float)
        Symmetric positive-definite (spd) matrix.
    epsilon : float
        Truncation parameter. Eigenvalues with norms smaller than this cutoff will
        be removed.
    method : str
        Method to perform the decomposition of :math:`W` before inverting. Options are:

        * 'QR': QR-based robust eigenvalue decomposition of W
        * 'schur': Schur decomposition of W
    canonical_signs : bool, default = False
        Fix signs in V, such that the largest element in every column of V is positive.
    check_sym : bool, default = False
        Check whether the input matrix is (almost) symmetric.

    Returns
    -------
    s : ndarray(k)
        k non-negligible eigenvalues, sorted by descending norms

    V : ndarray(n, k)
        k leading eigenvectors
    """
    # check input
    if check_sym and not _np.allclose(W.T, W):
        raise ValueError('W is not a symmetric matrix')

    if method == 'QR':
        from .eig_qr import eig_qr
        s, V = eig_qr(W)
    # compute the Eigenvalues of C0 using Schur factorization
    elif method == 'schur':
        from scipy.linalg import schur
        S, V = schur(W)
        s = _np.diag(S)
    else:
        raise ValueError(f'method {method} not implemented, available are {spd_eig.methods}')

    s, V = sort_eigs(s, V)  # sort them

    # determine the cutoff. We know that C0 is an spd matrix,
    # so we select the truncation threshold such that everything that is negative vanishes
    evmin = _np.min(s)
    if evmin < 0:
        epsilon = max(epsilon, -evmin + 1e-16)

    # determine effective rank m and perform low-rank approximations.
    evnorms = _np.abs(s)
    n = _np.shape(evnorms)[0]
    m = n - _np.searchsorted(evnorms[::-1], epsilon)
    if m == 0:
        raise ZeroRankError(
            'All eigenvalues are smaller than %g, rank reduction would discard all dimensions.' % epsilon)
    Vm = V[:, 0:m]
    sm = s[0:m]

    if canonical_signs:
        # enforce canonical eigenvector signs
        for j in range(m):
            jj = _np.argmax(_np.abs(Vm[:, j]))
            Vm[:, j] *= _np.sign(Vm[jj, j])

    return sm, Vm


spd_eig.methods = ('QR', 'schur')


def spd_inv(W, epsilon=1e-10, method='QR'):
    """
    Compute matrix inverse of symmetric positive-definite matrix :math:`W`.

    by first reducing W to a low-rank approximation that is truly spd
    (Moore-Penrose inverse).

    Parameters
    ----------
    W : ndarray((m,m), dtype=float)
        Symmetric positive-definite (spd) matrix.
    epsilon : float
        Truncation parameter. Eigenvalues with norms smaller than this cutoff will
        be removed.
    method : str
        Method to perform the decomposition of :math:`W` before inverting. Options are:

        * 'QR': QR-based robust eigenvalue decomposition of W
        * 'schur': Schur decomposition of W

    Returns
    -------
    L : ndarray((n, r))
        the Moore-Penrose inverse of the symmetric positive-definite matrix :math:`W`

    """
    if _np.shape(W)[0] == 1:
        if W[0, 0] < epsilon:
            raise ZeroRankError(
                'All eigenvalues are smaller than %g, rank reduction would discard all dimensions.' % epsilon)
        Winv = 1. / W[0, 0]
    else:
        sm, Vm = spd_eig(W, epsilon=epsilon, method=method)
        Winv = _np.dot(Vm, _np.diag(1.0 / sm)).dot(Vm.T)

    # return split
    return Winv


def spd_inv_sqrt(W, epsilon=1e-10, method='QR', return_rank=False):
    """
    Computes :math:`W^{-1/2}` of symmetric positive-definite matrix :math:`W`.

    by first reducing W to a low-rank approximation that is truly spd.

    Parameters
    ----------
    W : ndarray((m,m), dtype=float)
        Symmetric positive-definite (spd) matrix.
    epsilon : float, optional, default=1e-10
        Truncation parameter. Eigenvalues with norms smaller than this cutoff will be removed.
    method : str, optional, default='QR'
        Method to perform the decomposition of :math:`W` before inverting. Options are:

        * 'QR': QR-based robust eigenvalue decomposition of W
        * 'schur': Schur decomposition of W

    return_rank : bool, optional, default=False
        Whether to return the rank of W.

    Returns
    -------
    L : ndarray((n, r))
        :math:`W^{-1/2}` after reduction of W to a low-rank spd matrix

    """
    if _np.shape(W)[0] == 1:
        if W[0, 0] < epsilon:
            raise ZeroRankError(
                'All eigenvalues are smaller than %g, rank reduction would discard all dimensions.' % epsilon)
        Winv = 1. / _np.sqrt(W[0, 0])
        sm = _np.ones(1)
    else:
        sm, Vm = spd_eig(W, epsilon=epsilon, method=method)
        Winv = _np.dot(Vm, _np.diag(1.0 / _np.sqrt(sm))).dot(Vm.T)

    # return split
    if return_rank:
        return Winv, sm.shape[0]
    else:
        return Winv


def spd_inv_split(W, epsilon=1e-10, method='QR', canonical_signs=False):
    """
    Compute :math:`W^{-1} = L L^T` of the symmetric positive-definite matrix :math:`W`.

    by first reducing W to a low-rank approximation that is truly spd.

    Parameters
    ----------
    W : ndarray((m,m), dtype=float)
        Symmetric positive-definite (spd) matrix.
    epsilon : float
        Truncation parameter. Eigenvalues with norms smaller than this cutoff will
        be removed.
    method : str
        Method to perform the decomposition of :math:`W` before inverting. Options are:

        * 'QR': QR-based robust eigenvalue decomposition of W
        * 'schur': Schur decomposition of W

    canonical_signs : boolean, default = False
        Fix signs in L, s. t. the largest element of in every row of L is positive.


    Returns
    -------
    L : ndarray((n, r))
        Matrix :math:`L` from the decomposition :math:`W^{-1} = L L^T`.

    """
    if _np.shape(W)[0] == 1:
        if W[0, 0] < epsilon:
            raise ZeroRankError(
                'All eigenvalues are smaller than %g, rank reduction would discard all dimensions.' % epsilon)
        L = 1. / _np.sqrt(W[0, 0])
    else:
        if epsilon == 0.:
            sm, Vm = _np.linalg.eigh(.5 * (W + W.T))
            sm, Vm = sort_eigs(sm, Vm)
            # threshold eigenvalues to be >= 0 and sqrt of the eigenvalues to be >= 1e-16 so that no
            # division by zero can occur
            L = _np.dot(Vm, _np.diag(1.0 / _np.maximum(_np.sqrt(_np.maximum(sm, 1e-12)), 1e-12)))
        else:
            sm, Vm = spd_eig(W, epsilon=epsilon, method=method, canonical_signs=canonical_signs)
            L = _np.dot(Vm, _np.diag(1.0 / _np.sqrt(sm)))

    # return split
    return L


def eigs(matrix: _np.ndarray, n_eigs=None, which='LM'):
    r"""Computes the eigenvalues and eigenvectors of `matrix`. Optionally the number of eigenvalue-eigenvector pairs
    can be provided, in which case they are computed using the Lanczos algorithm.

    Parameters
    ----------
    matrix : (n, n) ndarray
        The matrix to compute the eigenvalues and eigenvectors of.
    n_eigs : int, optional, default=None
        The number of eigenpairs. Can be None, in which case all eigenvalues are computed.
    which : str, default='LM'
        If `n_eigs` is provided, determines which eigenvalues are returned. Default is 'Largest Magnitude.
        See the `scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html>`__
        documentation for all options and effects.

    Returns
    -------
    (eigenvalues, eigenvectors) : ((n, ) ndarray, (k, n) ndarray)
        The computed eigenvalues and eigenvectors.
    """
    n = matrix.shape[0]
    if n_eigs is not None and n_eigs < n:
        return _sp.sparse.linalg.eigs(matrix, n_eigs, which=which)
    else:
        return _sp.linalg.eig(matrix)


def eig_corr(C0, Ct, epsilon=1e-10, method='QR', canonical_signs=False, return_rank=False):
    r""" Solve generalized eigenvalue problem with correlation matrices C0 and Ct

    Numerically robust solution of a generalized Hermitian (symmetric) eigenvalue
    problem of the form

    .. math::
        \mathbf{C}_t \mathbf{r}_i = \mathbf{C}_0 \mathbf{r}_i l_i

    Computes :math:`m` dominant eigenvalues :math:`l_i` and eigenvectors
    :math:`\mathbf{r}_i`, where :math:`m` is the numerical rank of the problem.
    This is done by first conducting a Schur decomposition of the symmetric
    positive matrix :math:`\mathbf{C}_0`, then truncating its spectrum to
    retain only eigenvalues that are numerically greater than zero, then using
    this decomposition to define an ordinary eigenvalue Problem for
    :math:`\mathbf{C}_t` of size :math:`m`, and then solving this eigenvalue
    problem.

    Parameters
    ----------
    C0 : ndarray (n,n)
        time-instantaneous correlation matrix. Must be symmetric positive definite
    Ct : ndarray (n,n)
        time-lagged correlation matrix. Must be symmetric
    epsilon : float
        eigenvalue norm cutoff. Eigenvalues of C0 with norms <= epsilon will be
        cut off. The remaining number of Eigenvalues define the size of
        the output.
    method : str
        Method to perform the decomposition of :math:`W` before inverting. Options are:

        * 'QR': QR-based robust eigenvalue decomposition of W
        * 'schur': Schur decomposition of W
    canonical_signs : bool
        If True, re-scale each eigenvector such that its entry with maximal absolute value
        is positive.
    return_rank : bool, default=False
        If True, return the rank of generalized eigenvalue problem.

    Returns
    -------
    l : ndarray (m)
        The first m generalized eigenvalues, sorted by descending norm
    R : ndarray (n,m)
        The first m generalized eigenvectors, as a column matrix.
    rank: int
        Rank of :math:`C0^{-0.5}`, if return_rank is True.
    """
    L = spd_inv_split(C0, epsilon=epsilon, method=method, canonical_signs=True)
    Ct_trans = _np.dot(_np.dot(L.T, Ct), L)

    # solve the symmetric eigenvalue problem in the new basis
    if _np.allclose(Ct.T, Ct):
        from scipy.linalg import eigh
        l, R_trans = eigh(Ct_trans)
    else:
        from scipy.linalg import eig
        l, R_trans = eig(Ct_trans)

    # sort eigenpairs
    l, R_trans = sort_eigs(l, R_trans)

    # transform the eigenvectors back to the old basis
    R = _np.dot(L, R_trans)

    # Change signs of eigenvectors:
    if canonical_signs:
        for j in range(R.shape[1]):
            imax = _np.argmax(_np.abs(R[:, j]))
            R[:, j] *= _np.sign(R[imax, j])

    if return_rank:
        return l, R, L.shape[1] if L.ndim == 2 else 1

    # return result
    return l, R
