
# This file is part of MSMTools.
#
# Copyright (c) 2016, 2015, 2014 Computational Molecular Biology Group,
# Freie Universitaet Berlin (GER)
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
.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import numpy as np
from scipy.linalg import solve, lu_factor, lu_solve, cho_factor, cho_solve, eigvalsh
from scipy.sparse import issparse, diags, csr_matrix, bmat
from scipy.sparse.linalg import splu, SuperLU, minres


def mydot(A, B):
    r"""Dot-product that can handle dense and sparse arrays

    Parameters
    ----------
    A : numpy ndarray or scipy sparse matrix
        The first factor
    B : numpy ndarray or scipy sparse matrix
        The second factor

    Returns
    C : numpy ndarray or scipy sparse matrix
        The dot-product of A and B

    """
    if issparse(A) :
        return A.dot(B)
    elif issparse(B):
        return (B.T.dot(A.T)).T
    else:
        return np.dot(A, B)

def myfactor(A):
    if issparse(A):
        return splu(A.tocsc())
    else:
        return lu_factor(A)

def mysolve(LU, b):
    if isinstance(LU, SuperLU):
        return LU.solve(b)
    else:
        return lu_solve(LU, b)

###############################################################################
# Solve via full system
###############################################################################

def factor_full(z, DPhival, G, A):
    return DPhival

def solve_full(z, Fval, DPhival, G, A):
    M, N=G.shape
    P, N=A.shape

    """Total number of inequality constraints"""
    m=M

    """Primal variable"""
    x=z[0:N]

    """Multiplier for equality constraints"""
    nu=z[N:N+P]

    """Multiplier for inequality constraints"""
    l=z[N+P:N+P+M]

    """Slacks"""
    s=z[N+P+M:]

    """Dual infeasibility"""
    rd = Fval[0:N]

    """Primal infeasibility"""
    rp1 = Fval[N:N+P]
    rp2 = Fval[N+P:N+P+M]

    """Centrality"""
    rc = Fval[N+P+M:]

    """Sigma matrix"""
    SIG = np.diag(l/s)

    """Condensed system"""
    if issparse(DPhival):
        if not issparse(A):
            A = csr_matrix(A)
        H = DPhival + mydot(G.T, mydot(SIG, G))
        J = bmat([[H, A.T], [A, None]])
    else:
        if issparse(A):
            A = A.toarray()
        J = np.zeros((N+P, N+P))
        J[0:N, 0:N] = DPhival + mydot(G.T, mydot(SIG, G))
        J[0:N, N:] = A.T
        J[N:, 0:N] = A

    b1 = -rd - mydot(G.T, mydot(SIG, rp2)) + mydot(G.T, rc/s)
    b2 = -rp1
    b = np.hstack((b1, b2))

    """Prepare iterative solve via MINRES"""
    sign = np.zeros(N+P)
    sign[0:N/2] = 1.0
    sign[N/2:] = -1.0
    S = diags(sign, 0)
    J_new = mydot(S, csr_matrix(J))
    b_new = mydot(S, b)

    dJ_new = np.abs(J_new.diagonal())
    dPc = np.ones(J_new.shape[0])
    ind = (dJ_new > 0.0)
    dPc[ind] = 1.0/dJ_new[ind]
    Pc = diags(dPc, 0)
    dxnu, info = minres(J_new, b_new, tol=1e-8, M=Pc)

    # dxnu = solve(J, b)
    dx = dxnu[0:N]
    dnu = dxnu[N:]

    """Obtain search directions for l and s"""
    ds = -rp2 - mydot(G, dx)
    dl = -mydot(SIG, ds) - rc/s

    dz = np.hstack((dx, dnu, dl, ds))
    return dz

###############################################################################
# Solve via augmented system
###############################################################################

def factor_aug(z, DPhival, G, A):
    M, N = G.shape
    P, N = A.shape
    """Multiplier for inequality constraints"""
    l = z[N+P:N+P+M]

    """Slacks"""
    s = z[N+P+M:]

    """Sigma matrix"""
    SIG = diags(l/s, 0)

    """Condensed system"""
    if issparse(DPhival):
        if not issparse(A):
            A = csr_matrix(A)
        H = DPhival + mydot(G.T, mydot(SIG, G))
        J = bmat([[H, A.T], [A, None]])
    else:
        if issparse(A):
            A = A.toarray()
        J = np.zeros((N+P, N+P))
        J[0:N, 0:N] = DPhival + mydot(G.T, mydot(SIG, G))
        J[0:N, N:] = A.T
        J[N:, 0:N] = A

    LU = myfactor(J)
    return LU

def solve_factorized_aug(z, Fval, LU, G, A):
    M, N=G.shape
    P, N=A.shape

    """Total number of inequality constraints"""
    m = M

    """Primal variable"""
    x = z[0:N]

    """Multiplier for equality constraints"""
    nu = z[N:N+P]

    """Multiplier for inequality constraints"""
    l = z[N+P:N+P+M]

    """Slacks"""
    s = z[N+P+M:]

    """Dual infeasibility"""
    rd = Fval[0:N]

    """Primal infeasibility"""
    rp1 = Fval[N:N+P]
    rp2 = Fval[N+P:N+P+M]

    """Centrality"""
    rc = Fval[N+P+M:]

    """Sigma matrix"""
    SIG = diags(l/s, 0)

    """RHS for condensed system"""
    b1 = -rd - mydot(G.T, mydot(SIG, rp2)) + mydot(G.T, rc/s)
    b2 = -rp1
    b = np.hstack((b1, b2))
    dxnu = mysolve(LU, b)
    dx = dxnu[0:N]
    dnu = dxnu[N:]

    """Obtain search directions for l and s"""
    ds = -rp2 - mydot(G, dx)
    dl = -mydot(SIG, ds) - rc/s

    dz = np.hstack((dx, dnu, dl, ds))
    return dz

###############################################################################
# Solve via normal equations (Schur complement)
###############################################################################

def factor_schur(z, DPhival, G, A):
    M, N = G.shape
    P, N = A.shape
    """Multiplier for inequality constraints"""
    l = z[N+P:N+P+M]

    """Slacks"""
    s = z[N+P+M:]

    """Sigma matrix"""
    SIG = diags(l/s, 0)

    """Augmented Jacobian"""
    H = DPhival + mydot(G.T, mydot(SIG, G))

    """Factor H"""
    LU_H = myfactor(H)

    """Compute H^{-1}A^{T}"""
    HinvAt = mysolve(LU_H, A.T)

    """Compute Schur complement AH^{-1}A^{T}"""
    S = mydot(A, HinvAt)

    """Factor Schur complement"""
    LU_S = myfactor(S)

    LU = (LU_S, LU_H)
    return LU

def solve_factorized_schur(z, Fval, LU, G, A):
    M, N=G.shape
    P, N=A.shape

    """Total number of inequality constraints"""
    m = M

    """Primal variable"""
    x = z[0:N]

    """Multiplier for equality constraints"""
    nu = z[N:N+P]

    """Multiplier for inequality constraints"""
    l = z[N+P:N+P+M]

    """Slacks"""
    s = z[N+P+M:]

    """Dual infeasibility"""
    rd = Fval[0:N]

    """Primal infeasibility"""
    rp1 = Fval[N:N+P]
    rp2 = Fval[N+P:N+P+M]

    """Centrality"""
    rc = Fval[N+P+M:]

    """Sigma matrix"""
    SIG = diags(l/s, 0)

    """Assemble right hand side of augmented system"""
    r1 = rd + mydot(G.T, mydot(SIG, rp2)) - mydot(G.T, rc/s)
    r2 = rp1

    """Unpack LU-factors"""
    LU_S, LU_H = LU

    """Assemble right hand side for normal equation"""
    b = r2 - mydot(A, mysolve(LU_H, r1))

    """Solve for dnu"""
    dnu = mysolve(LU_S, b)

    """Solve for dx"""
    dx = mysolve(LU_H, -(r1 + mydot(A.T, dnu)))

    """Obtain search directions for l and s"""
    ds = -rp2 - mydot(G, dx)
    dl = -mydot(SIG, ds) - rc/s

    dz = np.hstack((dx, dnu, dl, ds))
    return dz

