# This file is part of scikit-time
#
# Copyright (c) 2020 AI4Science Group, Freie Universitaet Berlin (GER)
#
# scikit-time is free software: you can redistribute it and/or modify
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
.. currentmodule: sktime.numeric

===============================================================================
General numerical tools
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    is_diagonal_matrix
    is_square_matrix

===============================================================================
Numerical tools for eigenvalue problems
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    eig_corr
    sort_by_norm
    spd_eig
    spd_inv
    spd_inv_split
    spd_inv_sqrt
    ZeroRankError
"""
from .utils import drop_nan_rows, is_diagonal_matrix, is_square_matrix
from .eigen import eig_corr, sort_by_norm, spd_eig, spd_inv, spd_inv_split, spd_inv_sqrt, ZeroRankError
