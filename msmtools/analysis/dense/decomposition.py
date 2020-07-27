
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

r"""This module provides matrix-decomposition based functions for the
analysis of stochastic matrices

Below are the dense implementations for functions specified in api
Dense matrices are represented by numpy.ndarrays throughout this module.

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import numpy as np
import numbers
import warnings

from scipy.linalg import eig, eigh, eigvals, eigvalsh, solve

from ...util.exceptions import SpectralWarning, ImaginaryEigenValueWarning

from .stationary_vector import stationary_distribution
from .assessment import is_reversible


def eigenvalues(T, k=None, reversible=False, mu=None):
    r"""Compute eigenvalues of given transition matrix.

    Parameters
    ----------
    T : (d, d) ndarray
        Transition matrix (stochastic matrix)
    k : int or tuple of ints, optional
        Compute the first k eigenvalues of T
    reversible : bool, optional
        Indicate that transition matrix is reversible
    mu : (d,) ndarray, optional
        Stationary distribution of T

    Returns
    -------
    eig : (n,) ndarray,
        The eigenvalues of T ordered with decreasing absolute value.
        If k is None then n=d, if k is int then n=k otherwise
        n is the length of the given tuple of eigenvalue indices.

    Notes
    -----
    Eigenvalues are computed using the numpy.linalg interface
    for the corresponding LAPACK routines.

    If reversible=True the the eigenvalues of the similar symmetric
    matrix `\sqrt(\mu_i / \mu_j) p_{ij}` will be computed.

    The precomputed stationary distribution will only be used if
    reversible=True.

    """
    if reversible:
        try:
            evals = eigenvalues_rev(T, k=k, mu=mu)
        except ValueError:
            evals = eigvals(T).real  # use fallback code but cast to real
    else:
        evals = eigvals(T)  # nonreversible

    """Sort by decreasing absolute value"""
    ind = np.argsort(np.abs(evals))[::-1]
    evals = evals[ind]

    if isinstance(k, (list, set, tuple)):
        try:
            return [evals[n] for n in k]
        except IndexError:
            raise ValueError("given indices do not exist: ", k)
    elif k is not None:
        return evals[: k]
    else:
        return evals


def eigenvalues_rev(T, k=None, mu=None):
    r"""Compute eigenvalues of reversible transition matrix.

    Parameters
    ----------
    T : (d, d) ndarray
        Transition matrix (stochastic matrix)
    k : int or tuple of ints, optional
        Compute the first k eigenvalues of T
    mu : (d,) ndarray, optional
        Stationary distribution of T

    Returns
    -------
    eig : (n,) ndarray,
        The eigenvalues of T ordered with decreasing absolute value.
        If k is None then n=d, if k is int then n=k otherwise
        n is the length of the given tuple of eigenvalue indices.

    Raises
    ------
    ValueError
        If stationary distribution is nonpositive.

    """

    """compute stationary distribution if not given"""
    if mu is None:
        mu = stationary_distribution(T)
    if np.any(mu <= 0):
        raise ValueError('Cannot symmetrize transition matrix')
    """ symmetrize T """
    smu = np.sqrt(mu)
    S = smu[:,None] * T / smu
    """ symmetric eigenvalue problem """
    evals = eigvalsh(S)
    return evals


def eigenvectors(T, k=None, right=True, reversible=False, mu=None):
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
    if reversible:
        eigvec = eigenvectors_rev(T, right=right, mu=mu)
    else:
        eigvec = eigenvectors_nrev(T, right=right)

    """ Return eigenvectors """
    if k is None:
        return eigvec
    elif isinstance(k, numbers.Integral):
        return eigvec[:, 0:k]
    else:
        ind = np.asarray(k)
        return eigvec[:, ind]


def eigenvectors_nrev(T, right=True):
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

    Returns
    -------
    eigvec : (d, d) ndarray
        The eigenvectors of T ordered with decreasing absolute value
        of the corresponding eigenvalue

    """
    if right:
        val, R = eig(T, left=False, right=True)
        """ Sorted eigenvalues and left and right eigenvectors. """
        perm = np.argsort(np.abs(val))[::-1]
        # eigval=val[perm]
        eigvec = R[:, perm]

    else:
        val, L = eig(T, left=True, right=False)

        """ Sorted eigenvalues and left and right eigenvectors. """
        perm = np.argsort(np.abs(val))[::-1]
        # eigval=val[perm]
        eigvec = L[:, perm]
    return eigvec


def eigenvectors_rev(T, right=True, mu=None):
    r"""Compute eigenvectors of reversible transition matrix.

    Parameters
    ----------
    T : (d, d) ndarray
        Transition matrix (stochastic matrix)
    right : bool, optional
        If right=True compute right eigenvectors, left eigenvectors
        otherwise
    mu : (d,) ndarray, optional
        Stationary distribution of T

    Returns
    -------
    eigvec : (d, d) ndarray
        The eigenvectors of T ordered with decreasing absolute value
        of the corresponding eigenvalue

    """
    if mu is None:
        mu = stationary_distribution(T)
    """ symmetrize T """
    smu = np.sqrt(mu)
    S = smu[:,None] * T / smu
    val, eigvec = eigh(S)
    """Sort eigenvectors"""
    perm = np.argsort(np.abs(val))[::-1]
    eigvec = eigvec[:, perm]
    if right:
        return eigvec / smu[:, np.newaxis]
    else:
        return eigvec * smu[:, np.newaxis]


def rdl_decomposition(T, k=None, reversible=False, norm='standard', mu=None):
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
    # auto-set norm
    if norm == 'auto':
        if is_reversible(T):
            norm = 'reversible'
        else:
            norm = 'standard'

    if reversible:
        R, D, L = rdl_decomposition_rev(T, norm=norm, mu=mu)
    else:
        R, D, L = rdl_decomposition_nrev(T, norm=norm)

    if reversible or norm == 'reversible':
        D = D.real

    if k is None:
        return R, D, L
    else:
        return R[:, 0:k], D[0:k, 0:k], L[0:k, :]


def rdl_decomposition_nrev(T, norm='standard'):
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
    d = T.shape[0]
    w, R = eig(T)

    """Sort by decreasing magnitude of eigenvalue"""
    ind = np.argsort(np.abs(w))[::-1]
    w = w[ind]
    R = R[:, ind]

    """Diagonal matrix containing eigenvalues"""
    D = np.diag(w)

    # Standard norm: Euclidean norm is 1 for r and LR = I.
    if norm == 'standard':
        L = solve(np.transpose(R), np.eye(d))

        """l1- normalization of L[:, 0]"""
        R[:, 0] = R[:, 0] * np.sum(L[:, 0])
        L[:, 0] = L[:, 0] / np.sum(L[:, 0])

        return R, D, np.transpose(L)

    # Reversible norm:
    elif norm == 'reversible':
        b = np.zeros(d)
        b[0] = 1.0

        A = np.transpose(R)
        nu = solve(A, b)
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


def rdl_decomposition_rev(T, norm='reversible', mu=None):
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
    if mu is None:
        mu = stationary_distribution(T)
    """ symmetrize T """
    smu = np.sqrt(mu)
    S = smu[:,None] * T / smu
    val, eigvec = eigh(S)
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
    L[:, 0] = L[:, 0] *  tmp

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


def timescales(T, tau=1, k=None, reversible=False, mu=None):
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
    values = eigenvalues(T, reversible=reversible, mu=mu)

    """Sort by absolute value"""
    ind = np.argsort(np.abs(values))[::-1]
    values = values[ind]

    if k is None:
        values = values
    else:
        values = values[0:k]

    """Compute implied time scales"""
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
