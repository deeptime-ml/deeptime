r"""
.. currentmodule: deeptime.clustering

===============================================================================
Estimators
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    KMeans
    MiniBatchKMeans
    RegularSpace
    BoxDiscretization


===============================================================================
Models
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    ClusterModel
    KMeansModel
    BoxDiscretizationModel


===============================================================================
Functions
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    kmeans_plusplus


===============================================================================
Adding a new metric
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    Metric
    metrics
    MetricRegistry
"""

from ._metric import metrics, MetricRegistry
from ._clustering_bindings import Metric
from ._kmeans import KMeans, MiniBatchKMeans, KMeansModel
from ._regspace import RegularSpace
from ._box import BoxDiscretization, BoxDiscretizationModel
from ._cluster_model import ClusterModel

from ._kmeans import kmeans_plusplus
