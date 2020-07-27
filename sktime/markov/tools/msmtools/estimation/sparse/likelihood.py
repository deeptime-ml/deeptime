
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

"""This module implements the transition matrix functionality"""

import numpy as np
import scipy


def log_likelihood(C, T):
    """
        implementation of likelihood of C given T
    """
    C = C.tocsr()
    T = T.tocsr()
    ind = scipy.nonzero(C)
    relT = np.array(T[ind])[0, :]
    relT = np.log(relT)
    relC = np.array(C[ind])[0, :]

    return relT.dot(relC)
