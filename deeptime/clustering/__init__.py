r"""
.. currentmodule: deeptime.clustering

===============================================================================
Estimators
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    KmeansClustering
    MiniBatchKmeansClustering
    RegularSpaceClustering


===============================================================================
Models
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    ClusterModel
    KMeansClusteringModel


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
