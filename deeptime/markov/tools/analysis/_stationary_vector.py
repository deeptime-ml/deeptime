r"""This module provides functions for the computation of stationary
vectors of stochastic matrices

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>
.. moduleauthor:: M. Hoffmann
"""
import numpy as np
import scipy.sparse as sparse
import scipy.linalg as la
import scipy.sparse.linalg as sla


class _LUSolve:

    def __init__(self, mat):
        self._sparse = sparse.issparse(mat)
        if sparse.issparse(mat):
            self._solve = sla.factorized(mat)
        else:
            self._lu = la.lu_factor(mat)

    def __call__(self, vec):
        if self._sparse:
            return self._solve(vec)
        else:
            return la.lu_solve(self._lu, vec)


def backward_iteration(A, mu, x0, tol=1e-14, maxiter=100):
    r"""Find eigenvector to approximate eigenvalue via backward iteration.

    Parameters
    ----------
    A : (N, N) ndarray or sparse matrix
        Matrix for which eigenvector is desired
    mu : float
        Approximate eigenvalue for desired eigenvector
    x0 : (N, ) ndarray
        Initial guess for eigenvector
    tol : float
        Tolerance parameter for termination of iteration
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    x : (N, ) ndarray
        Eigenvector to approximate eigenvalue mu

    """
    if sparse.issparse(A):
        T = A - mu * sparse.eye(A.shape[0], A.shape[0])
        T = T.tocsc()
    else:
        T = A - mu * np.eye(A.shape[0])
    lu = _LUSolve(T)  # LU-factor of T

    # Starting iterate with ||y_0||=1
    r0 = 1.0 / np.linalg.norm(x0)
    y0 = x0 * r0
    # Local variables for inverse iteration
    y = 1.0 * y0
    r = 1.0 * r0
    for i in range(maxiter):
        x = lu(y)
        r = 1.0 / np.linalg.norm(x)
        y = x * r
        if r <= tol:
            return y
    msg = "Failed to converge after %d iterations, residuum is %e" % (maxiter, r)
    raise RuntimeError(msg)


def stationary_distribution_from_backward_iteration(P, eps=1e-15):
    r"""Fast computation of the stationary vector using backward
    iteration.

    Parameters
    ----------
    P : (M, M) ndarray or scipy.sparse matrix
        Transition matrix
    eps : float (optional)
        Perturbation parameter for the true eigenvalue.

    Returns
    -------
    pi : (M,) ndarray
        Stationary vector

    """
    A = P.transpose()
    mu = 1.0 - eps
    x0 = np.ones(P.shape[0])
    y = backward_iteration(A, mu, x0)
    pi = y / y.sum()
    return pi


def stationary_distribution_from_eigenvector(T, ncv=None):
    r"""Compute stationary distribution of stochastic matrix T.

    The stationary distribution is the left eigenvector corresponding to the
    non-degenerate eigenvalue :math: `\lambda=1`.

    Input:
    ------
    T : numpy array, shape(d,d)
        Transition matrix (stochastic matrix).
    ncv : int (optional)
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k

    Returns
    -------
    mu : numpy array, shape(d,)
        Vector of stationary probabilities.

    """
    if sparse.issparse(T):
        vals, vecs = sla.eigs(T.transpose(), k=1, which='LR', ncv=ncv)
    else:
        vals, vecs = la.eig(T, left=True, right=False)

    # Sorted eigenvalues and left and right eigenvectors.
    perm = np.argsort(vals)[::-1]
    vecs = vecs[:, perm]

    # Make sure that stationary distribution is non-negative and l1-normalized
    nu = np.abs(vecs[:, 0])
    mu = nu / np.sum(nu)
    return mu


def stationary_distribution(T, ncv=None):
    r"""Compute stationary distribution of stochastic matrix T.

    Chooses the fastest applicable algorithm automatically

    Input:
    ------
    T : numpy array, shape(d,d)
        Transition matrix (stochastic matrix).
    ncv : int (optional)
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k. Only relevant for sparse matrices and if backward iteration is unsuccessful.

    Returns
    -------
    mu : numpy array, shape(d,)
        Vector of stationary probabilities.
    """
    mu = None
    try:
        mu = stationary_distribution_from_backward_iteration(T)
    except RuntimeError:
        pass

    if mu is None or np.any(mu < 0):  # numerical problem, fall back to more robust algorithm.
        mu = stationary_distribution_from_eigenvector(T, ncv=ncv)
        if np.any(mu < 0):  # still? Then set to 0 and renormalize
            mu = np.maximum(mu, 0.0)
            mu /= mu.sum()

    return mu
