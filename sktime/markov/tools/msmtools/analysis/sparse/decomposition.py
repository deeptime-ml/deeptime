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

Below are the sparse implementations for functions specified in msm.api.
Matrices are represented by scipy.sparse matrices throughout this module.

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import numpy as np
import scipy.sparse.linalg
import warnings

from scipy.sparse import diags

from ...util.exceptions import ImaginaryEigenValueWarning, SpectralWarning
from .stationary_vector import stationary_distribution
from .assessment import is_reversible


def eigenvalues(T, k=None, ncv=None, reversible=False, mu=None):
    r"""Compute the eigenvalues of a sparse transition matrix.

    Parameters
    ----------
    T : (M, M) scipy.sparse matrix
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
    if k is None:
        raise ValueError("Number of eigenvalues required for decomposition of sparse matrix")
    else:
        if reversible:
            try:
                v = eigenvalues_rev(T, k, ncv=ncv, mu=mu)
            except ValueError:  # use fallback code, but cast to real
                v = scipy.sparse.linalg.eigs(T, k=k, which='LM', return_eigenvectors=False, ncv=ncv).real
        else:
            v = scipy.sparse.linalg.eigs(T, k=k, which='LM', return_eigenvectors=False, ncv=ncv)

    ind = np.argsort(np.abs(v))[::-1]
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

    """compute stationary distribution if not given"""
    if mu is None:
        mu = stationary_distribution(T)
    if np.any(mu <= 0):
        raise ValueError('Cannot symmetrize transition matrix')
    """ symmetrize T """
    smu = np.sqrt(mu)
    D = diags(smu, 0)
    Dinv = diags(1.0 / smu, 0)
    S = (D.dot(T)).dot(Dinv)
    """Compute eigenvalues using a solver for symmetric/hermititan eigenproblems"""
    evals = scipy.sparse.linalg.eigsh(S, k=k, ncv=ncv, which='LM', return_eigenvectors=False)
    return evals


def eigenvectors(T, k=None, right=True, ncv=None, reversible=False, mu=None):
    r"""Compute eigenvectors of given transition matrix.

    Parameters
    ----------
    T : scipy.sparse matrix
        Transition matrix (stochastic matrix).
    k : int (optional) or array-like
        For integer k compute the first k eigenvalues of T
        else return those eigenvector sepcified by integer indices in k.
    right : bool, optional
        If True compute right eigenvectors, left eigenvectors otherwise
    ncv : int (optional)
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k
    reversible : bool, optional
        Indicate that transition matrix is reversible
    mu : (M,) ndarray, optional
        Stationary distribution of T


    Returns
    -------
    eigvec : numpy.ndarray, shape=(d, n)
        The eigenvectors of T ordered with decreasing absolute value of
        the corresponding eigenvalue. If k is None then n=d, if k is
        int then n=k otherwise n is the length of the given indices array.

    Notes
    -----
    Eigenvectors are computed using the scipy interface
    to the corresponding ARPACK routines.

    """
    if k is None:
        raise ValueError("Number of eigenvectors required for decomposition of sparse matrix")
    else:
        if reversible:
            eigvec = eigenvectors_rev(T, k, right=right, ncv=ncv, mu=mu)
            return eigvec
        else:
            eigvec = eigenvectors_nrev(T, k, right=right, ncv=ncv)
            return eigvec


def eigenvectors_nrev(T, k, right=True, ncv=None):
    r"""Compute eigenvectors of transition matrix.

    Parameters
    ----------
    T : (M, M) scipy.sparse matrix
        Transition matrix (stochastic matrix)
    k : int
        Number of eigenvalues to compute
    right : bool, optional
        If True compute right eigenvectors, left eigenvectors otherwise
    ncv : int, optional
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k

    Returns
    -------
    eigvec : (M, k) ndarray
        k-eigenvectors of T

    """
    if right:
        val, vecs = scipy.sparse.linalg.eigs(T, k=k, which='LM', ncv=ncv)
        ind = np.argsort(np.abs(val))[::-1]
        return vecs[:, ind]
    else:
        val, vecs = scipy.sparse.linalg.eigs(T.transpose(), k=k, which='LM', ncv=ncv)
        ind = np.argsort(np.abs(val))[::-1]
        return vecs[:, ind]


def eigenvectors_rev(T, k, right=True, ncv=None, mu=None):
    r"""Compute eigenvectors of reversible transition matrix.

    Parameters
    ----------
    T : (M, M) scipy.sparse matrix
        Transition matrix (stochastic matrix)
    k : int
        Number of eigenvalues to compute
    right : bool, optional
        If True compute right eigenvectors, left eigenvectors otherwise
    ncv : int, optional
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k
    mu : (M,) ndarray, optional
        Stationary distribution of T

    Returns
    -------
    eigvec : (M, k) ndarray
        k-eigenvectors of T

    """
    if mu is None:
        mu = stationary_distribution(T)
    """ symmetrize T """
    smu = np.sqrt(mu)
    D = diags(smu, 0)
    Dinv = diags(1.0 / smu, 0)
    S = (D.dot(T)).dot(Dinv)
    """Compute eigenvalues, eigenvecs using a solver for
    symmetric/hermititan eigenproblems"""
    val, eigvec = scipy.sparse.linalg.eigsh(S, k=k, ncv=ncv, which='LM',
                                            return_eigenvectors=True)
    """Sort eigenvectors"""
    ind = np.argsort(np.abs(val))[::-1]
    eigvec = eigvec[:, ind]
    if right:
        return eigvec / smu[:, np.newaxis]
    else:
        return eigvec * smu[:, np.newaxis]


def rdl_decomposition(T, k=None, norm='auto', ncv=None, reversible=False, mu=None):
    r"""Compute the decomposition into left and right eigenvectors.

    Parameters
    ----------
    T : sparse matrix
        Transition matrix
    k : int (optional)
        Number of eigenvector/eigenvalue pairs
    norm: {'standard', 'reversible', 'auto'}
        standard: (L'R) = Id, L[:,0] is a probability distribution,
            the stationary distribution mu of T. Right eigenvectors
            R have a 2-norm of 1.
        reversible: R and L are related via L=L[:,0]*R.
        auto: will be reversible if T is reversible, otherwise standard.
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

    """
    if k is None:
        raise ValueError("Number of eigenvectors required for decomposition of sparse matrix")
    # auto-set norm
    if norm == 'auto':
        if is_reversible(T):
            norm = 'reversible'
        else:
            norm = 'standard'
    if reversible:
        R, D, L = rdl_decomposition_rev(T, k, norm=norm, ncv=ncv, mu=mu)
    else:
        R, D, L = rdl_decomposition_nrev(T, k, norm=norm, ncv=ncv)

    if reversible or norm == 'reversible':
        D = D.real
    return R, D, L


def rdl_decomposition_nrev(T, k, norm='standard', ncv=None):
    r"""Compute the decomposition into left and right eigenvectors.

    Parameters
    ----------
    T : sparse matrix
        Transition matrix
    k : int
        Number of eigenvector/eigenvalue pairs
    norm: {'standard', 'reversible'}
        standard: (L'R) = Id, L[:,0] is a probability distribution,
            the stationary distribution mu of T. Right eigenvectors
            R have a 2-norm of 1.
        reversible: R and L are related via L=L[:,0]*R.
    ncv : int (optional)
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k

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

    """
    # Standard norm: Euclidean norm is 1 for r and LR = I.
    if norm == 'standard':
        v, R = scipy.sparse.linalg.eigs(T, k=k, which='LM', ncv=ncv)
        r, L = scipy.sparse.linalg.eigs(T.transpose(), k=k, which='LM', ncv=ncv)

        """Sort right eigenvectors"""
        ind = np.argsort(np.abs(v))[::-1]
        v = v[ind]
        R = R[:, ind]

        """Sort left eigenvectors"""
        ind = np.argsort(np.abs(r))[::-1]
        r = r[ind]
        L = L[:, ind]

        """l1-normalization of L[:, 0]"""
        L[:, 0] = L[:, 0] / np.sum(L[:, 0])

        """Standard normalization L'R=Id"""
        ov = np.diag(np.dot(np.transpose(L), R))
        R = R / ov[np.newaxis, :]

        """Diagonal matrix with eigenvalues"""
        D = np.diag(v)

        return R, D, np.transpose(L)

    # Reversible norm:
    elif norm == 'reversible':
        v, R = scipy.sparse.linalg.eigs(T, k=k, which='LM', ncv=ncv)
        mu = stationary_distribution(T)

        """Sort right eigenvectors"""
        ind = np.argsort(np.abs(v))[::-1]
        v = v[ind]
        R = R[:, ind]

        """Ensure that R[:,0] is positive"""
        R[:, 0] = R[:, 0] / np.sign(R[0, 0])

        """Diagonal matrix with eigenvalues"""
        D = np.diag(v)

        """Compute left eigenvectors from right ones"""
        L = mu[:, np.newaxis] * R

        """Compute overlap"""
        s = np.diag(np.dot(np.transpose(L), R))

        """Renormalize left-and right eigenvectors to ensure L'R=Id"""
        R = R / np.sqrt(s[np.newaxis, :])
        L = L / np.sqrt(s[np.newaxis, :])

        return R, D, np.transpose(L)
    else:
        raise ValueError("Keyword 'norm' has to be either 'standard' or 'reversible'")


def rdl_decomposition_rev(T, k, norm='reversible', ncv=None, mu=None):
    r"""Compute the decomposition into left and right eigenvectors.

    Parameters
    ----------
    T : sparse matrix
        Transition matrix
    k : int
        Number of eigenvector/eigenvalue pairs
    norm: {'standard', 'reversible'}
        standard: (L'R) = Id, L[:,0] is a probability distribution,
            the stationary distribution mu of T. Right eigenvectors
            R have a 2-norm of 1.
        reversible: R and L are related via L=L[:,0]*R.
    ncv : int (optional)
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k
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

    """
    if mu is None:
        mu = stationary_distribution(T)
    """ symmetrize T """
    smu = np.sqrt(mu)
    Dpi = diags(smu, 0)
    Dinv = diags(1.0 / smu, 0)
    S = (Dpi.dot(T)).dot(Dinv)
    """Compute eigenvalues, eigenvecs using a solver for
    symmetric/hermititan eigenproblems"""
    val, eigvec = scipy.sparse.linalg.eigsh(S, k=k, ncv=ncv, which='LM',
                                            return_eigenvectors=True)
    """Sort eigenvalues and eigenvectors"""
    ind = np.argsort(np.abs(val))[::-1]
    val = val[ind]
    eigvec = eigvec[:, ind]

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
    T : transition matrix
    tau : lag time
    k : int (optional)
        Compute the first k implied time scales.
    ncv : int (optional)
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k
    reversible : bool, optional
        Indicate that transition matrix is reversible
    mu : (M,) ndarray, optional
        Stationary distribution of T

    Returns
    -------
    ts : ndarray
        The implied time scales of the transition matrix.
    """
    if k is None:
        raise ValueError("Number of time scales required for decomposition of sparse matrix")
    values = eigenvalues(T, k=k, ncv=ncv, reversible=reversible)

    """Check for dominant eigenvalues with large imaginary part"""
    if not np.allclose(values.imag, 0.0):
        warnings.warn('Using eigenvalues with non-zero imaginary part '
                      'for implied time scale computation', ImaginaryEigenValueWarning)

    """Check for multiple eigenvalues of magnitude one"""
    ind_abs_one = np.isclose(np.abs(values), 1.0)
    if sum(ind_abs_one) > 1:
        warnings.warn('Multiple eigenvalues with magnitude one.', SpectralWarning)

    """Compute implied time scales"""
    ts = np.zeros(len(values))

    """Eigenvalues of magnitude one imply infinite rate"""
    ts[ind_abs_one] = np.inf

    """All other eigenvalues give rise to finite rates"""
    ts[np.logical_not(ind_abs_one)] = \
        -1.0 * tau / np.log(np.abs(values[np.logical_not(ind_abs_one)]))
    return ts


def timescales_from_eigenvalues(ev, tau=1):
    r"""Compute implied time scales from given eigenvalues

    Parameters
    ----------
    eval : eigenvalues
    tau : lag time

    Returns
    -------
    ts : ndarray
        The implied time scales to the given eigenvalues, in the same order.

    """

    """Check for dominant eigenvalues with large imaginary part"""
    if not np.allclose(ev.imag, 0.0):
        warnings.warn('Using eigenvalues with non-zero imaginary part '
                      'for implied time scale computation', ImaginaryEigenValueWarning)

    """Check for multiple eigenvalues of magnitude one"""
    ind_abs_one = np.isclose(np.abs(ev), 1.0, rtol=0.0, atol=1e-14)
    if sum(ind_abs_one) > 1:
        warnings.warn('Multiple eigenvalues with magnitude one.', SpectralWarning)

    """Compute implied time scales"""
    ts = np.zeros(len(ev))

    """Eigenvalues of magnitude one imply infinite timescale"""
    ts[ind_abs_one] = np.inf

    """All other eigenvalues give rise to finite timescales"""
    ts[np.logical_not(ind_abs_one)] = \
        -1.0 * tau / np.log(np.abs(ev[np.logical_not(ind_abs_one)]))
    return ts
