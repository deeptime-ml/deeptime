
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

from pyemma.coordinates.data.numpy_filereader import NumPyFileReader as _NumPyFileReader
from pyemma.coordinates.data.py_csv_reader import PyCSVReader as _CSVReader
from pyemma.coordinates.data import FeatureReader as _FeatureReader
import mdtraj as md
import os


def create_file_reader(input_files, topology, featurizer):
    r"""
    Creates a (possibly featured) file reader by a number of input files and either a topology file or a featurizer.
    Parameters
    ----------
    :param input_files:
        A single input file or a list of input files.
    :param topology:
        A topology file. If given, the featurizer argument can be None.
    :param featurizer:
        A featurizer. If given, the topology file can be None.
    :return: Returns the reader.
    """
    if isinstance(input_files, basestring) \
            or (isinstance(input_files, (list, tuple))
                and (any(isinstance(item, basestring) for item in input_files) or len(input_files) is 0)):
        reader = None
        # check: if single string create a one-element list
        if isinstance(input_files, basestring):
            input_list = [input_files]
        elif len(input_files) > 0 and all(isinstance(item, basestring) for item in input_files):
            input_list = input_files
        else:
            if len(input_files) is 0:
                raise ValueError("The passed input list should not be empty.")
            else:
                raise ValueError("The passed list did not exclusively contain strings.")

        _, suffix = os.path.splitext(input_list[0])

        # check: do all files have the same file type? If not: raise ValueError.
        if all(item.endswith(suffix) for item in input_list):

            # do all the files exist? If not: Raise value error
            all_exist = True
            err_msg = ""
            for item in input_list:
                if not os.path.isfile(item):
                    err_msg += "\n" if len(err_msg) > 0 else ""
                    err_msg += "File %s did not exist or was no file" % item
                    all_exist = False
            if not all_exist:
                raise ValueError("Some of the given input files were directories"
                                 " or did not exist:\n%s" % err_msg)

            if all_exist:
                from mdtraj.formats.registry import _FormatRegistry

                # CASE 1.1: file types are MD files
                if suffix in _FormatRegistry.loaders.keys():
                    # check: do we either have a featurizer or a topology file name? If not: raise ValueError.
                    # create a MD reader with file names and topology
                    if not featurizer and not topology:
                        raise ValueError("The input files were MD files which makes it mandatory to have either a "
                                         "featurizer or a topology file.")

                    reader = _FeatureReader(input_list, featurizer=featurizer, topologyfile=topology)
                else:
                    if suffix in ['.npy', '.npz']:
                        reader = _NumPyFileReader(input_list)
                    # otherwise we assume that given files are ascii tabulated data
                    else:
                        reader = _CSVReader(input_list)
        else:
            raise ValueError("Not all elements in the input list were of the type %s!" % suffix)
    else:
        raise ValueError("Input \"%s\" was no string or list of strings." % input)
    return reader


def single_traj_from_n_files(file_list, top):
    """ Creates a single trajectory object from a list of files

    """
    traj = None
    for ff in file_list:
        if traj is None:
            traj = md.load(ff, top=top)
        else:
            traj = traj.join(md.load(ff, top=top))

    return traj