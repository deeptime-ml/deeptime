
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


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
from .featurization.featurizer import MDFeaturizer, CustomFeature
from .data_in_memory import DataInMemory
from .numpy_filereader import NumPyFileReader
from .py_csv_reader import PyCSVReader

# util func
from .util.reader_utils import create_file_reader
