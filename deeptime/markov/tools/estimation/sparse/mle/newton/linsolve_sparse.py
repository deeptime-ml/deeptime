r"""
.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import numpy as np
from scipy.sparse import issparse, csr_matrix, diags
from scipy.sparse.linalg import minres, LinearOperator


class AugmentedSystem(LinearOperator):

    def __init__(self, J, G, SIG, A):
        self.N1 = J.shape[0]
        self.N2 = A.shape[0]
        N = self.N1 + self.N2
        self.shape = (N, N)
        self.dtype = J.dtype

        self.E = ((G.T.dot(SIG)).dot(G)).tocsr()
        self.J = J
        self.A = A
        self.AT = A.T.tocsr()
        self.diag = np.hstack((J.diagonal() + self.E.diagonal(),
                               np.zeros(self.N2)))

    def _matvec(self, x):
        N1 = self.N1
        N2 = self.N2
        y = np.zeros(N1+N2)
        y[0:N1] = self.J.dot(x[0:N1]) + self.E.dot(x[0:N1]) + self.AT.dot(x[N1:])
        y[N1:] = self.A.dot(x[0:N1])
        return y

    def diagonal(self):
        return self.diag

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

def factor_aug(z, DPhival, G, A):
    r"""Set up augmented system and return.

    Parameters
    ----------
    z : (N+P+M+M,) ndarray
        Current iterate, z = (x, nu, l, s)
    DPhival : LinearOperator
        Jacobian of the variational inequality mapping
    G : (M, N) ndarray or sparse matrix
        Inequality constraints
    A : (P, N) ndarray or sparse matrix
        Equality constraints

    Returns
    -------
    J : LinearOperator
        Augmented system

    """
    M, N = G.shape
    P, N = A.shape
    """Multiplier for inequality constraints"""
    l = z[N+P:N+P+M]

    """Slacks"""
    s = z[N+P+M:]

    """Sigma matrix"""
    SIG = diags(l/s, 0)
    # SIG = diags(l*s, 0)

    """Convert A"""
    if not issparse(A):
        A = csr_matrix(A)

    """Convert G"""
    if not issparse(G):
        G = csr_matrix(G)

    """Since we expect symmetric DPhival, we need to change A"""
    sign = np.zeros(N)
    sign[0:N//2] = 1.0
    sign[N//2:] = -1.0
    T = diags(sign, 0)

    A_new = A.dot(T)

    W = AugmentedSystem(DPhival, G, SIG, A_new)
    return W

def solve_factorized_aug(z, Fval, LU, G, A):
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
    SIG = diags(l/s, 0)

    """LU is actually the augmented system W"""
    W = LU

    b1 = -rd - mydot(G.T, mydot(SIG, rp2)) + mydot(G.T, rc/s)
    b2 = -rp1
    b = np.hstack((b1, b2))

    """Prepare iterative solve via MINRES"""
    sign = np.zeros(N+P)
    sign[0:N//2] = 1.0
    sign[N//2:] = -1.0
    T = diags(sign, 0)

    """Change rhs"""
    b_new = mydot(T, b)

    dW = np.abs(W.diagonal())
    dPc = np.ones(W.shape[0])
    ind = (dW > 0.0)
    dPc[ind] = 1.0/dW[ind]
    Pc = diags(dPc, 0)
    dxnu, info = minres(W, b_new, tol=1e-10, M=Pc)

    # dxnu = solve(J, b)
    dx = dxnu[0:N]
    dnu = dxnu[N:]

    """Obtain search directions for l and s"""
    ds = -rp2 - mydot(G, dx)
    # ds = s*ds
    # SIG = np.diag(l/s)
    dl = -mydot(SIG, ds) - rc/s

    dz = np.hstack((dx, dnu, dl, ds))
    return dz
