
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
from scipy.sparse import csr_matrix, diags, bmat
from scipy.sparse.construct import _compressed_sparse_stack
from scipy.sparse.linalg import LinearOperator

cimport cython
cimport numpy as np

from libc.math cimport exp

ctypedef np.int32_t DTYPE_INT_t
ctypedef np.float_t DTYPE_FLOAT_t

def convert_solution(np.ndarray[DTYPE_FLOAT_t, ndim=1] z, Cs):
    cdef size_t M, k, l, j
    cdef double cs_kj, ekj
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] x, y
    cdef np.ndarray[DTYPE_INT_t, ndim=1] indices, indptr
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] data, data_P, diag_P, nu

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
    diag_P = np.zeros(M)
    nu = np.zeros(M)

    """Loop over rows of Cs"""
    for k in range(M):
        nu[k] = exp(y[k])
        """Loop over nonzero entries in row of Cs"""
        for l in range(indptr[k], indptr[k+1]):
            """Column index of current element"""
            j = indices[l]
            if k != j:
                """Current element of Cs at (k, j)"""
                cs_kj = data[l]
                """Exponential of difference"""
                ekj = exp(y[k]-y[j])
                """Compute off diagonal element"""
                data_P[l] = cs_kj/(x[k] + x[j]*ekj)
                """Update diagonal element"""
                diag_P[k] -= data_P[l]
        diag_P[k] += 1.0

    P = csr_matrix((data_P, indices, indptr), shape=(M, M)) + diags(diag_P, 0)
    return nu/nu.sum(), P    

def F(np.ndarray[DTYPE_FLOAT_t, ndim=1] z, Cs, np.ndarray[DTYPE_FLOAT_t, ndim=1] c):
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
    cdef size_t M, k, l, j
    cdef double cs_kj, ekj
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] x, y
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] data, Fval
    cdef np.ndarray[DTYPE_INT_t, ndim=1] indices, indptr

    if not isinstance(Cs, csr_matrix):
        """Convert to csr_matrix"""
        Cs = csr_matrix(Cs)    

    M = Cs.shape[0]

    x = z[0:M]
    y = z[M:]

    data = Cs.data
    indptr = Cs.indptr
    indices = Cs.indices

    Fval = np.zeros(2*M,)

    """Loop over rows of Cs"""
    for k in range(M):
        Fval[k] += 1.0
        Fval[k+M] -= c[k]

        """Loop over nonzero entries in row of Cs"""
        for l in range(indptr[k], indptr[k+1]):
            """Column index of current element"""
            j = indices[l]
            """Current element of Cs at (k, j)"""
            cs_kj = data[l]
            """Exponential of difference"""
            ekj = exp(y[k]-y[j])
            """Update Fx"""
            Fval[k] += -cs_kj/(x[k]+x[j]*ekj)
            """Update Fy"""
            Fval[k+M] -= -cs_kj*x[j]/(x[k]/ekj + x[j])               
    return Fval

def _DF(np.ndarray[DTYPE_FLOAT_t, ndim=1] z, Cs, np.ndarray[DTYPE_FLOAT_t, ndim=1] c):
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
    cdef size_t M, k, l, j
    cdef double cs_kj, ekj, tmp1, tmp2
    cdef np.ndarray[np.float_t, ndim=1] x, y
    cdef np.ndarray[np.float_t, ndim=1] data, data_Hxx, data_Hyy, data_Hyx
    cdef np.ndarray[np.float_t, ndim=1] diag_Dxx, diag_Dyy, diag_Dyx
    cdef np.ndarray[np.int32_t, ndim=1] indices, indptr

    if not isinstance(Cs, csr_matrix):
        """Convert to csr_matrix"""
        Cs = csr_matrix(Cs)    

    M = Cs.shape[0]

    x = z[0:M]
    y = z[M:]

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

    diag_Dxx = np.zeros(M)
    diag_Dyx = np.zeros(M)
    diag_Dyy = np.zeros(M)

    """Loop over rows of Cs"""
    for k in range(M):
        """Loop over nonzero entries in row of Cs"""
        for l in range(indptr[k], indptr[k+1]):
            """Column index of current element"""
            j = indices[l]
            """Current element of Cs at (k, j)"""
            cs_kj = data[l]

            ekj = exp(y[k]-y[j])

            tmp1 = cs_kj/((x[k]+x[j]*ekj)*(x[k]/ekj+x[j]))
            tmp2 = cs_kj/(x[k] + x[j]*ekj)**2

            data_Hxx[l] = tmp1
            diag_Dxx[k] += tmp2

            data_Hyy[l] = tmp1*x[k]*x[j]
            diag_Dyy[k] -= tmp1*x[k]*x[j]

            data_Hyx[l] = -tmp1*x[k]
            diag_Dyx[k] += tmp1*x[j]

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

def DF(np.ndarray[DTYPE_FLOAT_t, ndim=1] z, Cs, np.ndarray[DTYPE_FLOAT_t, ndim=1] c):
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
    DFxx, DFyx, DFyy = _DF(z, Cs, c)
    """The call to bmat is really expensive, but I don't know how to avoid it
    if a sparse matrix is desired"""
    DFval = bmat([[DFxx, DFyx.T], [-1.0*DFyx, -1.0*DFyy]]).tocsr()    
    return DFval

def DFsym(np.ndarray[DTYPE_FLOAT_t, ndim=1] z, Cs, np.ndarray[DTYPE_FLOAT_t, ndim=1] c):
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

class JacobianOperator(LinearOperator):
    r"""Realise the following block sparse matrix.
           A   B.T
    M = (          )
          -B   -C     
    """
    
    def __init__(self, A, B, C):
        self.N1 = A.shape[0]
        self.N2 = C.shape[0]
        N = self.N1+self.N2
        self.shape = (N, N)
        self.dtype = A.dtype
        
        self.A = A
        self.B = B
        self.BT = B.T.tocsr()
        self.C = C
        self.diag = np.hstack((A.diagonal(), -1.0*C.diagonal()))

    def _matvec(self, x):
        N1 = self.N1
        N2 = self.N2
        y = np.zeros(N1+N2)
        y[0:N1] = self.A.dot(x[0:N1])  + self.BT.dot(x[N1:])
        y[N1:] = -self.B.dot(x[0:N1]) - self.C.dot(x[N1:])
        return y

    def diagonal(self):
        return self.diag

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
        N = self.N1+self.N2
        self.shape = (N, N)
        self.dtype = A.dtype
        
        self.A = A
        self.B = B
        self.BT = B.T.tocsr()
        self.C = C
        self.diag = np.hstack((A.diagonal(), C.diagonal()))

    def _matvec(self, x):
        N1 = self.N1
        N2 = self.N2
        y = np.zeros(N1+N2)
        y[0:N1] = self.A.dot(x[0:N1])  + self.BT.dot(x[N1:])
        y[N1:] = self.B.dot(x[0:N1]) + self.C.dot(x[N1:])
        return y

    def diagonal(self):
        return self.diag

