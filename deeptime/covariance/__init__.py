r"""
.. currentmodule: deeptime.covariance

===============================================================================
Estimators
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    Covariance
    KoopmanWeightingEstimator


===============================================================================
Models
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    CovarianceModel
    KoopmanWeightingModel


===============================================================================
Utilities
===============================================================================

.. autosummary::
    :toctree: generated/impl/
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
