
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

r"""Decomposition of a netflux network into its dominant reaction pathways

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import warnings
import numpy as np
import scipy.sparse.csgraph as csgraph
from scipy.sparse import coo_matrix


class PathwayError(Exception):
    """Exception for failed attempt to find pathway in a given flux
    network"""
    pass


def find_bottleneck(F, A, B):
    r"""Find dynamic bottleneck of flux network.

    Parameters
    ----------
    F : scipy.sparse matrix
        The flux network
    A : array_like
        The set of starting states
    B : array_like
        The set of end states

    Returns
    -------
    e : tuple of int
    The edge corresponding to the dynamic bottleneck

    """
    if F.nnz == 0:
        raise PathwayError('no more pathways left: Flux matrix does not contain any positive entries')
    F = F.tocoo()
    n = F.shape[0]

    """Get exdges and corresponding flux values"""
    val = F.data
    row = F.row
    col = F.col

    """Sort edges according to flux"""
    ind = np.argsort(val)
    val = val[ind]
    row = row[ind]
    col = col[ind]

    """Check if edge with largest conductivity connects A and B"""
    b = np.array(row[-1], col[-1])
    if has_path(b, A, B):
        return b
    else:
        """Bisection of flux-value array"""
        r = val.size
        l = 0
        N = 0
        while r - l > 1:
            m = np.int(np.floor(0.5 * (r + l)))
            valtmp = val[m:]
            rowtmp = row[m:]
            coltmp = col[m:]
            C = coo_matrix((valtmp, (rowtmp, coltmp)), shape=(n, n))
            """Check if there is a path connecting A and B by
            iterating over all starting nodes in A"""
            if has_connection(C, A, B):
                l = 1 * m
            else:
                r = 1 * m

        E_AB = coo_matrix((val[l + 1:], (row[l + 1:], col[l + 1:])), shape=(n, n))
        b1 = row[l]
        b2 = col[l]
        return b1, b2, E_AB


def has_connection(graph, A, B):
    r"""Check if the given graph contains a path connecting A and B.

    Parameters
    ----------
    graph : scipy.sparse matrix
        Adjacency matrix of the graph
    A : array_like
        The set of starting states
    B : array_like
        The set of end states

    Returns
    -------
    hc : bool
       True if the graph contains a path connecting A and B, otherwise
       False.

    """
    for istart in A:
        nodes = csgraph.breadth_first_order(graph, istart, directed=True, return_predecessors=False)
        if has_path(nodes, A, B):
            return True
    return False


def has_path(nodes, A, B):
    r"""Test if nodes from a breadth_first_order search lead from A to
    B.

    Parameters
    ----------
    nodes : array_like
        Nodes from breadth_first_oder_seatch
    A : array_like
        The set of educt states
    B : array_like
        The set of product states

    Returns
    -------
    has_path : boolean
        True if there exists a path, else False

    """
    x1 = np.intersect1d(nodes, A).size > 0
    x2 = np.intersect1d(nodes, B).size > 0
    return x1 and x2


def pathway(F, A, B):
    r"""Compute the dominant reaction-pathway.

    Parameters
    ----------
    F : (M, M) scipy.sparse matrix
        The flux network (matrix of netflux values)
    A : array_like
        The set of starting states
    B : array_like
        The set of end states

    Returns
    -------
    w : list
        The dominant reaction-pathway

    """
    if F.nnz == 0:
        raise PathwayError('no more pathways left: Flux matrix does not contain any positive entries')
    b1, b2, F = find_bottleneck(F, A, B)
    if np.any(A == b1):
        wL = [b1, ]
    elif np.any(B == b1):
        raise PathwayError(("Roles of vertices b1 and b2 are switched."
                            "This should never happen for a correct flux network"
                            "obtained from a reversible transition meatrix."))
    else:
        wL = pathway(F, A, [b1, ])
    if np.any(B == b2):
        wR = [b2, ]
    elif np.any(A == b2):
        raise PathwayError(("Roles of vertices b1 and b2 are switched."
                            "This should never happen for a correct flux network"
                            "obtained from a reversible transition meatrix."))
    else:
        wR = pathway(F, [b2, ], B)
    return wL + wR


def capacity(F, path):
    r"""Compute capacity (min. current) of path.

    Paramters
    ---------
    F : (M, M) scipy.sparse matrix
        The flux network (matrix of netflux values)
    path : list
        Reaction path

    Returns
    -------
    c : float
        Capacity (min. current of path)

    """
    F = F.todok()
    L = len(path)
    currents = np.zeros(L - 1)
    for l in range(L - 1):
        i = path[l]
        j = path[l + 1]
        currents[l] = F[i, j]

    return currents.min()


def remove_path(F, path):
    r"""Remove capacity along a path from flux network.

    Parameters
    ----------
    F : (M, M) scipy.sparse matrix
        The flux network (matrix of netflux values)
    path : list
        Reaction path

    Returns
    -------
    F : (M, M) scipy.sparse matrix
        The updated flux network

    """
    c = capacity(F, path)
    F = F.todok()
    L = len(path)
    for l in range(L - 1):
        i = path[l]
        j = path[l + 1]
        F[i, j] -= c
    return F


def pathways(F, A, B, fraction=1.0, maxiter=1000, tol=1e-14):
    r"""Decompose flux network into dominant reaction paths.

    Parameters
    ----------
    F : (M, M) scipy.sparse matrix
        The flux network (matrix of netflux values)
    A : array_like
        The set of starting states
    B : array_like
        The set of end states
    fraction : float, optional
        Fraction of total flux to assemble in pathway decomposition
    maxiter : int, optional
        Maximum number of pathways for decomposition
    tol : float, optional
        Floating point tolerance. The iteration is terminated once the
        relative capacity of all discovered path matches the desired
        fraction within floating point tolerance

    Returns
    -------
    paths : list
        List of dominant reaction pathways
    capacities: list
        List of capacities corresponding to each reactions pathway in paths

    References
    ----------
    .. [1] P. Metzner, C. Schuette and E. Vanden-Eijnden.
        Transition Path Theory for Markov Jump Processes.
        Multiscale Model Simul 7: 1192-1219 (2009)

    """
    F, a, b = add_endstates(F, A, B)
    A = [a, ]
    B = [b, ]

    """Total flux"""
    TF = F.tocsr()[A, :].sum()

    """Total capacity fo all previously found reaction paths"""
    CF = 0.0
    niter = 0

    """List of dominant reaction pathways"""
    paths = []
    """List of corresponding capacities"""
    capacities = []

    while True:
        """Find dominant pathway of flux-network"""
        try:
            path = pathway(F, A, B)
        except PathwayError:
            break
        """Compute capacity of current pathway"""
        c = capacity(F, path)
        """Remove artifical end-states"""
        path = path[1:-1]
        """Append to lists"""
        paths.append(np.array(path))
        capacities.append(c)
        """Update capacity of all previously found paths"""
        CF += c
        """Remove capacity along given path from flux-network"""
        F = remove_path(F, path)
        niter += 1
        """Current flux numerically equals fraction * total flux or is
        greater equal than fraction * total flux"""
        if (abs(CF/TF - fraction) <= tol) or (CF/TF >= fraction):
            break
        if niter > maxiter:
            warnings.warn("Maximum number of iterations reached", RuntimeWarning)
            break
    return paths, capacities


def add_endstates(F, A, B):
    r"""Adds artifical end states replacing source and sink sets.

    Parameters
    ----------
    F : (M, M) scipy.sparse matrix
        The flux network (matrix of netflux values)
    A : array_like
        The set of starting states
    B : array_like
        The set of end states

    Returns
    -------
    F_new : (M+2, M+2) scipy.sparse matrix
        The artifical flux network with extra end states
    a_new : int
        The new single source a_new = M
    b_new : int
        The new single sink b_new = M+1

    """

    """Outgoing currents from A"""
    F = F.tocsr()
    outA = (F[A, :].sum(axis=1)).getA()[:, 0]

    """Incoming currents into B"""
    F = F.tocsc()
    inB = (F[:, B].sum(axis=0)).getA()[0, :]

    F = F.tocoo()
    M = F.shape[0]

    data_old = F.data
    row_old = F.row
    col_old = F.col

    """Add currents from new A=[n,] to all states in A"""
    row1 = np.zeros(outA.shape[0], dtype=np.int)
    row1[:] = M
    col1 = np.array(A)
    data1 = outA

    """Add currents from old B to new B=[n+1,]"""
    row2 = np.array(B)
    col2 = np.zeros(inB.shape[0], dtype=np.int)
    col2[:] = M + 1
    data2 = inB

    """Stack data, row and col arrays"""
    data = np.hstack((data_old, data1, data2))
    row = np.hstack((row_old, row1, row2))
    col = np.hstack((col_old, col1, col2))

    """New netflux matrix"""
    F_new = coo_matrix((data, (row, col)), shape=(M + 2, M + 2))

    return F_new, M, M + 1
