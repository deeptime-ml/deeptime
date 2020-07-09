
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
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
.. currentmodule: sktime.clustering

===============================================================================
K-Means
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    KmeansClustering
    MiniBatchKmeansClustering
    KMeansClusteringModel

===============================================================================
Regular space clustering
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    RegularSpaceClustering
    ClusterModel

===============================================================================
Adding a new metric
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    _clustering_bindings.Metric
    metrics
    MetricRegistry
"""

from .metric import metrics, MetricRegistry
from ._clustering_bindings import Metric
from .kmeans import KmeansClustering, MiniBatchKmeansClustering, KMeansClusteringModel
from .regspace import RegularSpaceClustering
from .cluster_model import ClusterModel
