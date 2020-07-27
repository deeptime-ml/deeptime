
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
from scipy.sparse import issparse


def convert_solution(z, Cs):
    if issparse(Cs):
        Cs = Cs.toarray()
    M = Cs.shape[0]
    x = z[0:M]
    y = z[M:]

    w=np.exp(y)
    pi=w/w.sum()

    X=pi[:,np.newaxis]*x[np.newaxis,:]
    Y=X+np.transpose(X)
    denom=Y
    enum=Cs*np.transpose(pi)
    P=enum/denom
    ind=np.diag_indices(Cs.shape[0])
    P[ind]=0.0
    rowsums=P.sum(axis=1)
    P[ind]=1.0-rowsums
    return pi, P

###############################################################################
# Objective, Gradient, and Hessian
###############################################################################

def f(z, C):
    N=z.shape[0]
    x=z[0:N/2]
    y=z[N/2:]
    q=np.exp(y)
    W=x[:,np.newaxis]*q[np.newaxis,:]
    Z=W+W.transpose()
    return -1.0*np.sum(C*np.log(Z))+np.sum(x)+np.sum(C*y[np.newaxis,:])

def F(z, Cs, c):
    if not isinstance(Cs, np.ndarray):
        """Convert to dense array"""
        Cs.toarray()

    N=z.shape[0]
    x=z[0:N/2]
    y=z[N/2:]
    q=np.exp(y)
    # Cs=C+C.transpose()
    W=x[:,np.newaxis]*q[np.newaxis,:]
    Z=W+W.transpose()
    Fx=-1.0*np.sum(Cs*q[np.newaxis, :]/Z, axis=1)+1.0
    Fy= -1.0*np.sum(Cs*W.transpose()/Z, axis=1)+c
    return np.hstack((Fx, -1.0*Fy))

def DF(z, Cs, c):
    if not isinstance(Cs, np.ndarray):
        """Convert to dense array"""
        Cs.toarray()

    N=z.shape[0]
    x=z[0:N/2]
    y=z[N/2:]

    q=np.exp(y)

    # Cs=C+C.transpose()
    W=x[:,np.newaxis]*q[np.newaxis,:]
    Wt=W.transpose()
    Z=W+Wt

    Z2=Z**2
    Q=q[:,np.newaxis]*q[np.newaxis,:]

    dxx=np.sum(Cs*(q**2)[np.newaxis,:]/Z2, axis=1)
    DxDxf= np.diag(dxx)+Cs*Q/Z2

    dxy=np.sum(Cs*(x*q)[:,np.newaxis]*q[np.newaxis,:]/Z2, axis=0)
    DyDxf=-1.0*Cs*q[np.newaxis,:]/Z + Cs*(W*q[np.newaxis,:])/Z2+np.diag(dxy)

    DxDyf=DyDxf.transpose()

    Dyy1=-1.0*Cs*W/Z
    Dyy2=Cs*W**2/Z2
    dyy=np.sum(Dyy1, axis=0)+np.sum(Dyy2, axis=0)

    DyDyf=np.diag(dyy)+Cs*W*Wt/Z2

    J=np.zeros((N, N))
    J[0:N/2, 0:N/2]=DxDxf
    J[0:N/2, N/2:]=DyDxf
    J[N/2:, 0:N/2]=-1.0*DxDyf
    J[N/2:, N/2:]=-1.0*DyDyf

    return J

# def F(z, C):
#     N=z.shape[0]
#     x=z[0:N/2]
#     y=z[N/2:]
#     q=np.exp(y)
#     C_sym=C+C.transpose()
#     W=x[:,np.newaxis]*q[np.newaxis,:]
#     Z=W+W.transpose()
#     Fx=-1.0*np.sum(C_sym*q[np.newaxis, :]/Z, axis=1)+1.0
#     Fy= -1.0*np.sum(C_sym*W.transpose()/Z, axis=1)+np.sum(C, axis=0)
#     return np.hstack((Fx, -1.0*Fy))

# def DF(z, C):
#     N=z.shape[0]
#     x=z[0:N/2]
#     y=z[N/2:]

#     q=np.exp(y)

#     C_sym=C+C.transpose()
#     W=x[:,np.newaxis]*q[np.newaxis,:]
#     Wt=W.transpose()
#     Z=W+Wt

#     Z2=Z**2
#     Q=q[:,np.newaxis]*q[np.newaxis,:]

#     dxx=np.sum(C_sym*(q**2)[np.newaxis,:]/Z2, axis=1)
#     DxDxf= np.diag(dxx)+C_sym*Q/Z2

#     dxy=np.sum(C_sym*(x*q)[:,np.newaxis]*q[np.newaxis,:]/Z2, axis=0)
#     DyDxf=-1.0*C_sym*q[np.newaxis,:]/Z + C_sym*(W*q[np.newaxis,:])/Z2+np.diag(dxy)

#     DxDyf=DyDxf.transpose()

#     Dyy1=-1.0*C_sym*W/Z
#     Dyy2=C_sym*W**2/Z2
#     dyy=np.sum(Dyy1, axis=0)+np.sum(Dyy2, axis=0)

#     DyDyf=np.diag(dyy)+C_sym*W*Wt/Z2

#     J=np.zeros((N, N))
#     J[0:N/2, 0:N/2]=DxDxf
#     J[0:N/2, N/2:]=DyDxf
#     J[N/2:, 0:N/2]=-1.0*DxDyf
#     J[N/2:, N/2:]=-1.0*DyDyf

#     return J

