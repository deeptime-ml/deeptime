
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

r"""This module implements the countmatrix estimation functionality

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import numpy as np
import scipy.sparse

################################################################################
# count_matrix
################################################################################

def count_matrix_coo2_mult(dtrajs, lag, sliding=True, sparse=True, nstates=None):
    r"""Generate a count matrix from a given list discrete trajectories.

    The generated count matrix is a sparse matrix in compressed
    sparse row (CSR) or numpy ndarray format.

    Parameters
    ----------
    dtraj : list of ndarrays
        discrete trajectories
    lag : int
        Lagtime in trajectory steps
    sliding : bool, optional
        If true the sliding window approach
        is used for transition counting
    sparse : bool (optional)
        Whether to return a dense or a sparse matrix
    nstates : int, optional
        Enforce a count-matrix with shape=(nstates, nstates). If there are
        more states in the data, this will lead to an exception.

    Returns
    -------
    C : scipy.sparse.csr_matrix or numpy.ndarray
        The countmatrix at given lag in scipy compressed sparse row
        or numpy ndarray format.

    """
    # Determine number of states
    if nstates is None:
        from ...dtraj import number_of_states
        nstates = number_of_states(dtrajs)
    rows = []
    cols = []
    # collect transition index pairs
    for dtraj in dtrajs:
        if dtraj.size > lag:
            if (sliding):
                rows.append(dtraj[0:-lag])
                cols.append(dtraj[lag:])
            else:
                rows.append(dtraj[0:-lag:lag])
                cols.append(dtraj[lag::lag])
    # is there anything?
    if len(rows) == 0:
        raise ValueError('No counts found - lag ' + str(lag) + ' may exceed all trajectory lengths.')
    # feed into one COO matrix
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    data = np.ones(row.size)
    C = scipy.sparse.coo_matrix((data, (row, col)), shape=(nstates, nstates))
    # export to output format
    if sparse:
        return C.tocsr()
    else:
        return C.toarray()
