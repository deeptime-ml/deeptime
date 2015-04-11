r"""
===============================================================================
io - IO Utilities (:mod:`pyemma.coordinates.data`)
===============================================================================

.. currentmodule: pyemma.coordinates.data

Order parameters
================

.. autosummary::
    :toctree: generated/

    MDFeaturizer - selects and computes features from MD trajectories
    CustomFeature -

Reader
======

.. autosummary::
    :toctree: generated/

    FeatureReader - reads features via featurizer
    DataInMemory - used if data is already available in mem

"""
from feature_reader import FeatureReader
from featurizer import MDFeaturizer, CustomFeature
from data_in_memory import DataInMemory