r"""
.. currentmodule: deeptime.clustering

===============================================================================
Estimators
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    Kmeans
    MiniBatchKmeans
    RegularSpace


===============================================================================
Models
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    ClusterModel
    KMeansModel


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
from .kmeans import Kmeans, MiniBatchKmeans, KMeansModel
from .regspace import RegularSpace
from .cluster_model import ClusterModel
