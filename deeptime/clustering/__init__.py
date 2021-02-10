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

from ._metric import metrics, MetricRegistry
from ._clustering_bindings import Metric
from ._kmeans import Kmeans, MiniBatchKmeans, KMeansModel
from ._regspace import RegularSpace
from ._cluster_model import ClusterModel
