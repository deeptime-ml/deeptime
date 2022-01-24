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
    data.TimeLaggedConcatDataset
    data.TrajectoryDataset
    data.TrajectoriesDataset
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

===============================================================================
Type utilities
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    types.to_dataset
    types.is_timelagged_dataset
    types.atleast_nd


===============================================================================
Other utilities
===============================================================================
.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    parallel.handle_n_jobs
    decorators.cached_property
    decorators.plotting_function

    callbacks.supports_progress_interface
    callbacks.Callback

    platform.module_available
    platform.handle_progress_bar
"""

from .stats import QuantityStatistics, confidence_interval
from ._validation import LaggedModelValidator

from . import data
from . import types
from . import callbacks
from . import platform
