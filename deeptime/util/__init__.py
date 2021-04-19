r"""
.. currentmodule: deeptime.util

===============================================================================
Data utilities
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    data.timeshifted_split
    data.TimeLaggedDataset
    data.ConcatDataset

===============================================================================
Statistics utilities
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    QuantityStatistics
    confidence_interval
    LaggedModelValidator
"""

from .stats import QuantityStatistics, confidence_interval
from ._validation import LaggedModelValidator

from . import data
