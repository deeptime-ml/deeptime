
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

'''
Created on 22.11.2013

@author: Jan-Hendrik Prinz
'''

import numpy
from .stationary_vector import stationary_distribution


# TODO:make faster. So far not effectively programmed
# Martin: done, but untested, since there is no testcase...
def forward_committor_sensitivity(T, A, B, index):
    """
    calculate the sensitivity matrix for index of the forward committor from A to B given transition matrix T.
    Parameters
    ----------
    T : numpy.ndarray shape = (n, n)
        Transition matrix
    A : array like
        List of integer state labels for set A
    B : array like
        List of integer state labels for set B
    index : entry of the committor for which the sensitivity is to be computed

    Returns
    -------
    x : ndarray, shape=(n, n)
        Sensitivity matrix for entry index around transition matrix T. Reversibility is not assumed.
    """

    n = len(T)
    set_X = numpy.arange(n)  # set(range(n))
    set_A = numpy.unique(A)  # set(A)
    set_B = numpy.unique(B)  # set(B)
    set_AB = numpy.union1d(set_A, set_B)  # set_A | set_B
    notAB = numpy.setdiff1d(set_X, set_AB, True)  # list(set_X - set_AB)
    m = len(notAB)

    K = T - numpy.diag(numpy.ones(n))

    U = K[numpy.ix_(notAB.tolist(), notAB.tolist())]

    v = numpy.zeros(m)

    # for i in xrange(0, m):
    #   for k in xrange(0, len(set_B)):
    #       v[i] = v[i] - K[notAB[i], B[k]]
    v[:] = v[:] - K[notAB[:], B[:]]

    qI = numpy.linalg.solve(U, v)

    q_forward = numpy.zeros(n)
    #q_forward[set_A] = 0 # double assignment.
    q_forward[set_B] = 1
    #for i in range(len(notAB)):
    q_forward[notAB[:]] = qI[:]

    target = numpy.eye(1, n, index)
    target = target[0, notAB]

    UinvVec = numpy.linalg.solve(U.T, target)
    Siab = numpy.zeros((n, n))

    for i in range(m):
        Siab[notAB[i]] = - UinvVec[i] * q_forward

    return Siab


def backward_committor_sensitivity(T, A, B, index):
    """
    calculate the sensitivity matrix for index of the backward committor from A to B given transition matrix T.
    Parameters
    ----------
    T : numpy.ndarray shape = (n, n)
        Transition matrix
    A : array like
        List of integer state labels for set A
    B : array like
        List of integer state labels for set B
    index : entry of the committor for which the sensitivity is to be computed

    Returns
    -------
    x : ndarray, shape=(n, n)
        Sensitivity matrix for entry index around transition matrix T. Reversibility is not assumed.
    """

    # This is really ugly to compute. The problem is, that changes in T induce changes in
    # the stationary distribution and so we need to add this influence, too
    # I implemented something which is correct, but don't ask me about the derivation

    n = len(T)

    trT = numpy.transpose(T)

    one = numpy.ones(n)
    eq = stationary_distribution(T)

    mEQ = numpy.diag(eq)
    mIEQ = numpy.diag(1.0 / eq)
    mSEQ = numpy.diag(1.0 / eq / eq)

    backT = numpy.dot(mIEQ, numpy.dot(trT, mEQ))

    qMat = forward_committor_sensitivity(backT, A, B, index)

    matA = trT - numpy.identity(n)
    matA = numpy.concatenate((matA, [one]))

    phiM = numpy.linalg.pinv(matA)

    phiM = phiM[:, 0:n]

    trQMat = numpy.transpose(qMat)

    d1 = numpy.dot(mSEQ, numpy.diagonal(numpy.dot(numpy.dot(trT, mEQ), trQMat), 0))
    d2 = numpy.diagonal(numpy.dot(numpy.dot(trQMat, mIEQ), trT), 0)

    psi1 = numpy.dot(d1, phiM)
    psi2 = numpy.dot(-d2, phiM)

    v1 = psi1 - one * numpy.dot(psi1, eq)
    v3 = psi2 - one * numpy.dot(psi2, eq)

    part1 = numpy.outer(eq, v1)
    part2 = numpy.dot(numpy.dot(mEQ, trQMat), mIEQ)
    part3 = numpy.outer(eq, v3)

    sensitivity = part1 + part2 + part3

    return sensitivity


def eigenvalue_sensitivity(T, k):
    """
    calculate the sensitivity matrix for eigenvalue k given transition matrix T.
    Parameters
    ----------
    T : numpy.ndarray shape = (n, n)
        Transition matrix
    k : int
        eigenvalue index for eigenvalues order descending

    Returns
    -------
    x : ndarray, shape=(n, n)
        Sensitivity matrix for entry index around transition matrix T. Reversibility is not assumed.
    """

    eValues, rightEigenvectors = numpy.linalg.eig(T)
    leftEigenvectors = numpy.linalg.inv(rightEigenvectors)

    perm = numpy.argsort(eValues)[::-1]

    rightEigenvectors = rightEigenvectors[:, perm]
    leftEigenvectors = leftEigenvectors[perm]

    sensitivity = numpy.outer(leftEigenvectors[k], rightEigenvectors[:, k])

    return sensitivity


def timescale_sensitivity(T, k):
    """
    calculate the sensitivity matrix for timescale k given transition matrix T.
    Parameters
    ----------
    T : numpy.ndarray shape = (n, n)
        Transition matrix
    k : int
        timescale index for timescales of descending order (k = 0 for the infinite one)

    Returns
    -------
    x : ndarray, shape=(n, n)
        Sensitivity matrix for entry index around transition matrix T. Reversibility is not assumed.
    """

    eValues, rightEigenvectors = numpy.linalg.eig(T)
    leftEigenvectors = numpy.linalg.inv(rightEigenvectors)

    perm = numpy.argsort(eValues)[::-1]

    eValues = eValues[perm]
    rightEigenvectors = rightEigenvectors[:, perm]
    leftEigenvectors = leftEigenvectors[perm]

    eVal = eValues[k]

    sensitivity = numpy.outer(leftEigenvectors[k], rightEigenvectors[:, k])

    if eVal < 1.0:
        factor = 1.0 / (numpy.log(eVal) ** 2) / eVal
    else:
        factor = 0.0

    sensitivity *= factor

    return sensitivity


# TODO: The eigenvector sensitivity depends on the normalization, e.g. l^T r = 1 or norm(r) = 1
# Should we fix that or add another option. Also the sensitivity depends on the initial eigenvectors
# Now everything is set to use norm(v) = 1 for left and right
# In the case of the stationary distribution we want sum(pi) = 1, so this function
# does NOT return the same as stationary_distribution_sensitivity if we choose k = 0 and right = False!

# TODO: If we choose k = 0 and right = False we might throw a warning!?!

def eigenvector_sensitivity(T, k, j, right=True):
    """
    calculate the sensitivity matrix for entry j of left or right eigenvector k given transition matrix T.
    Parameters
    ----------
    T : numpy.ndarray shape = (n, n)
        Transition matrix
    k : int
        eigenvector index ordered with descending eigenvalues
    j : int
        entry of eigenvector k for which the sensitivity is to be computed
    right : boolean (default: True)
        If set to True (default) the right eigenvectors are considered, otherwise the left ones

    Returns
    -------
    x : ndarray, shape=(n, n)
        Sensitivity matrix for entry index around transition matrix T. Reversibility is not assumed.

    Remarks
    -------
    Eigenvectors can naturally be scaled and so will their sensitivity depend on their size.
    For that reason we need to agree on a normalization for which the sensitivity is computed.
    Here we use the natural norm(vector) = 1 condition which is different from the results, e.g.
    the rdl_decomposition returns.
    This is especially important for the stationary distribution, which is the first left eigenvector.
    For this reason this function return a different sensitivity for the first left eigenvector
    than the function stationary_distribution_sensitivity and this function should not be used in this
    case!
    """

    n = len(T)

    if not right:
        T = numpy.transpose(T)

    eValues, rightEigenvectors = numpy.linalg.eig(T)
    leftEigenvectors = numpy.linalg.inv(rightEigenvectors)
    perm = numpy.argsort(eValues)[::-1]

    eValues = eValues[perm]
    rightEigenvectors = rightEigenvectors[:, perm]
    leftEigenvectors = leftEigenvectors[perm]

    rEV = rightEigenvectors[:, k]
    lEV = leftEigenvectors[k]
    eVal = eValues[k]

    vecA = numpy.zeros(n)
    vecA[j] = 1.0

    matA = T - eVal * numpy.identity(n)
    # Use here rEV as additional condition, means that we assume the vector to be
    # orthogonal to rEV
    matA = numpy.concatenate((matA, [rEV]))

    phi = numpy.linalg.lstsq(numpy.transpose(matA), vecA, rcond=-1)

    phi = numpy.delete(phi[0], -1)

    sensitivity = -numpy.outer(phi, rEV) + numpy.dot(phi, rEV) * numpy.outer(lEV, rEV)

    if not right:
        sensitivity = numpy.transpose(sensitivity)

    return sensitivity


def stationary_distribution_sensitivity(T, j):
    r"""Calculate the sensitivity matrix for entry j the stationary
    distribution vector given transition matrix T.

    Parameters
    ----------
    T : numpy.ndarray shape = (n, n)
        Transition matrix
    j : int
        entry of stationary distribution for which the sensitivity is to be computed

    Returns
    -------
    x : ndarray, shape=(n, n)
        Sensitivity matrix for entry index around transition matrix T. Reversibility is not assumed.

    Remark
    ------
    Note, that this function uses a different normalization convention for the sensitivity compared to
    eigenvector_sensitivity. See there for further information.
    """

    n = len(T)

    lEV = numpy.ones(n)
    rEV = stationary_distribution(T)
    eVal = 1.0

    T = numpy.transpose(T)

    vecA = numpy.zeros(n)
    vecA[j] = 1.0

    matA = T - eVal * numpy.identity(n)
    # normalize s.t. sum is one using rEV which is constant
    matA = numpy.concatenate((matA, [lEV]))

    phi = numpy.linalg.lstsq(numpy.transpose(matA), vecA, rcond=-1)
    phi = numpy.delete(phi[0], -1)

    sensitivity = -numpy.outer(rEV, phi) + numpy.dot(phi, rEV) * numpy.outer(rEV, lEV)

    return sensitivity


def mfpt_sensitivity(T, target, j):
    """
    calculate the sensitivity matrix for entry j of the mean first passage time (MFPT) given transition matrix T.
    Parameters
    ----------
    T : numpy.ndarray shape = (n, n)
        Transition matrix
    target : int
        target state to which the MFPT is computed
    j : int
        entry of the mfpt vector for which the sensitivity is to be computed

    Returns
    -------
    x : ndarray, shape=(n, n)
        Sensitivity matrix for entry index around transition matrix T. Reversibility is not assumed.
    """

    n = len(T)

    matA = T - numpy.diag(numpy.ones((n)))
    matA[target] *= 0
    matA[target, target] = 1.0

    tVec = -1. * numpy.ones(n)
    tVec[target] = 0

    mfpt = numpy.linalg.solve(matA, tVec)
    aVec = numpy.zeros(n)
    aVec[j] = 1.0

    phiVec = numpy.linalg.solve(numpy.transpose(matA), aVec)

    # TODO: Check sign of sensitivity!

    sensitivity = -1.0 * numpy.outer(phiVec, mfpt)
    sensitivity[target] *= 0

    return sensitivity


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
    M = T.shape[0]
    S = numpy.zeros((M, M))
    for i in range(M):
        S += a[i] * stationary_distribution_sensitivity(T, i)
    return S
