
r"""
===============================================================================
data - Data and input/output utilities (:mod:`pyemma.coordinates.data`)
===============================================================================

.. currentmodule: pyemma.coordinates.data

Order parameters
================

.. autosummary::
    :toctree: generated/

    MDFeaturizer - selects and computes features from MD trajectories
    CustomFeature - define arbitrary function to extract features

Reader
======

.. autosummary::
    :toctree: generated/

    FeatureReader - reads features via featurizer
    NumPyFileReader - reads numpy files
    PyCSVReader - reads tabulated ascii files
    DataInMemory - used if data is already available in mem

"""
from .feature_reader import FeatureReader
from .featurizer import MDFeaturizer, CustomFeature
from .data_in_memory import DataInMemory
from .numpy_filereader import NumPyFileReader
from .py_csv_reader import PyCSVReader

# util func
from .util.reader_utils import create_file_reader
