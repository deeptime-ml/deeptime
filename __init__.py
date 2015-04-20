# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
