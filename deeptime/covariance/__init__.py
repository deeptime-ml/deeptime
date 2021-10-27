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

    WhiteningTransform

    covar
    covars

    moments_XX
    moments_XXXY
    moments_block
"""

from .util import moments_XX, moments_XXXY, moments_block, covar, covars
from ._covariance import Covariance, CovarianceModel, WhiteningTransform
from ._covariance import KoopmanWeightingEstimator, KoopmanWeightingModel
