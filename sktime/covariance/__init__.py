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
.. currentmodule: sktime.covariance

===============================================================================
Estimation
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    Covariance
    CovarianceModel

===============================================================================
Koopman reweighting
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    KoopmanWeightingEstimator
    KoopmanWeightingModel

===============================================================================
Implementations
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    covar
    covars

    moments_XX
    moments_XXXY
    moments_block
"""

from .util.moments import moments_XX, moments_XXXY, moments_block, covar, covars
from .covariance import Covariance, CovarianceModel
from .covariance import KoopmanWeightingEstimator, KoopmanWeightingModel
