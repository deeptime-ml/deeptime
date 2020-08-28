import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import LinearOperator

from . import objective_sparse_ops as ops


def convert_solution(z: np.ndarray, Cs):
    if not isinstance(Cs, csr_matrix):
        """Convert to csr_matrix"""
        Cs = csr_matrix(Cs)

    M = Cs.shape[0]
    x = z[0:M]
    y = z[M:]

    data = Cs.data
    indptr = Cs.indptr
    indices = Cs.indices

    data_P = np.zeros_like(data)
    diag_P = np.zeros(M, dtype=data.dtype)
    nu = np.zeros(M, dtype=data.dtype)

    ops.convertImpl(M, x, y, data, nu, data_P, diag_P, indices, indptr)

    P = csr_matrix((data_P, indices, indptr), shape=(M, M)) + diags(diag_P, 0)
    return nu / nu.sum(), P


def F(z: np.ndarray, Cs, c: np.ndarray):
    r"""Monotone mapping for the reversible MLE problem.

    Parameters
    ----------
    z : (2*M,) ndarray
        Point at which to evaluate mapping, z=(x, y)
    C : (M, M) scipy.sparse matrix
        Count matrix of reversible chain

    Returns
    -------
    Fval : (2*M,) ndarray
        Value of the mapping at z

    """
    if not isinstance(Cs, csr_matrix):
        """Convert to csr_matrix"""
        Cs = csr_matrix(Cs)

    M = Cs.shape[0]

    x = z[0:M]
    y = z[M:]

    data = Cs.data
    indptr = Cs.indptr
    indices = Cs.indices

    Fval = np.zeros(2 * M, dtype=data.dtype)

    ops.FImpl(M, x, y, c, data, Fval, indices, indptr)

    return Fval


def _DF(z: np.ndarray, Cs, c: np.ndarray):
    r"""Jacobian of the monotone mapping.

    Parameters
    ----------
    z : (2*M,) ndarray
        Point at which to evaluate mapping, z=(x, y)
    C : (M, M) scipy.sparse matrix
        Count matrix of reversible chain

    Returns
    -------
    DFval : (2*M, 2*M) scipy.sparse matrix
        Value of the Jacobian at z

    """
    if not isinstance(Cs, csr_matrix):
        """Convert to csr_matrix"""
        Cs = csr_matrix(Cs)

    M = Cs.shape[0]

    data = Cs.data
    indptr = Cs.indptr
    indices = Cs.indices

    """All subblocks DF_ij can be written as follows, DF_ij = H_ij +
    D_ij. H_ij has the same sparsity structure as C+C.T and D_ij is a
    diagonal matrix, i, j \in {x, y}
    """

    data_Hxx = np.zeros_like(data)
    data_Hyx = np.zeros_like(data)
    data_Hyy = np.zeros_like(data)

    diag_Dxx = np.zeros(M, dtype=data.dtype)
    diag_Dyx = np.zeros(M, dtype=data.dtype)
    diag_Dyy = np.zeros(M, dtype=data.dtype)

    x = z[0:M]
    y = z[M:]

    ops.DFImpl(M, x, y, data, data_Hxx, data_Hyy, data_Hyx, diag_Dxx, diag_Dyy, diag_Dyx, indices, indptr)

    Hxx = csr_matrix((data_Hxx, indices, indptr), shape=(M, M))
    Dxx = diags(diag_Dxx, 0)
    DFxx = Hxx + Dxx

    Hyy = csr_matrix((data_Hyy, indices, indptr), shape=(M, M))
    Dyy = diags(diag_Dyy, 0)
    DFyy = Hyy + Dyy

    Hyx = csr_matrix((data_Hyx, indices, indptr), shape=(M, M))
    Dyx = diags(diag_Dyx, 0)
    DFyx = Hyx + Dyx

    return DFxx, DFyx, DFyy


def DFsym(z: np.ndarray, Cs, c: np.ndarray):
    r"""Jacobian of the monotone mapping. Symmetric version.

    Parameters
    ----------
    z : (2*M,) ndarray
        Point at which to evaluate mapping, z=(x, y)
    C : (M, M) scipy.sparse matrix
        Count matrix of reversible chain

    Returns
    -------
    DFval : (2*M, 2*M) scipy.sparse matrix
        Value of the Jacobian at z

    """
    DFxx, DFyx, DFyy = _DF(z, Cs, c)
    return JacobianOperatorSymmetric(DFxx, DFyx, DFyy)


class JacobianOperatorSymmetric(LinearOperator):
    r"""Realise the following symmetric block sparse matrix.

           A   B.T
    M = (          )
           B   C

    with symmetric sub blocks A=A.T, C=C.T

    """

    def __init__(self, A, B, C):
        self.N1 = A.shape[0]
        self.N2 = C.shape[0]
        N = self.N1 + self.N2
        super().__init__(A.dtype, (N, N))

        self.A = A
        self.B = B
        self.BT = B.T.tocsr()
        self.C = C
        self.diag = np.hstack((A.diagonal(), C.diagonal()))

    def _matvec(self, x):
        N1 = self.N1
        N2 = self.N2
        y = np.zeros(N1 + N2, dtype=self.dtype)
        y[0:N1] = self.A.dot(x[0:N1]) + self.BT.dot(x[N1:])
        y[N1:] = self.B.dot(x[0:N1]) + self.C.dot(x[N1:])
        return y

    def diagonal(self):
        return self.diag
