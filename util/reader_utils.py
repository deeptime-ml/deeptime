
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
from numpy import vstack
import mdtraj as md
import numpy as np
import os


def create_file_reader(input_files, topology, featurizer, chunk_size=100):
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
    :param chunk_size:
        The chunk size with which the corresponding reader gets initialized.
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

                    reader = _FeatureReader(input_list, featurizer=featurizer, topologyfile=topology,
                                            chunksize=chunk_size)
                else:
                    if suffix in ['.npy', '.npz']:
                        reader = _NumPyFileReader(input_list, chunksize=chunk_size)
                    # otherwise we assume that given files are ascii tabulated data
                    else:
                        reader = _CSVReader(input_list, chunksize=chunk_size)
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

def save_traj_w_md_load_frame(reader, sets):
    # Creates a single trajectory object from a "sets" array via md.load_frames
    traj = None
    for file_idx, frame_idx in vstack(sets):
        if traj is None:
            traj = md.load_frame(reader.trajfiles[file_idx], frame_idx, reader.topfile)
        else:
            traj = traj.join(md.load_frame(reader.trajfiles[file_idx], frame_idx, reader.topfile))
    return traj


def compare_coords_md_trajectory_objects(traj1, traj2, atom = None, eps = 1e-6, mess = False ):
    # Compares the coordinates of "atom" for all frames in traj1 and traj2
    # Returns a boolean found_diff and an errmsg informing where
    assert isinstance(traj1, md.Trajectory)
    assert isinstance(traj2, md.Trajectory)
    assert traj1.n_frames == traj2.n_frames
    assert traj2.n_atoms == traj2.n_atoms

    R = np.zeros((2, traj1.n_frames, 3))
    if atom is None:
        atom_index = np.random.randint(0, high = traj1.n_atoms)
    else:
        atom_index = atom

    # Artificially mess the the coordinates
    if mess:
        traj1.xyz [0, atom_index, 2] += 10*eps

    for ii, traj in enumerate([traj1, traj2]):
        R[ii, :] = traj.xyz[:, atom_index]

    # Compare the R-trajectories among themselves
    found_diff = False
    first_diff = None
    errmsg = ''

    for ii, iR in enumerate(R):
    # Norm of the difference vector
        norm_diff = np.sqrt(((iR - R) ** 2).sum(2))

        # Any differences?
        if (norm_diff > eps).any():
            first_diff = np.argwhere(norm_diff > eps)[0]
            found_diff = True
            errmsg = "Delta R_%u at frame %u: [%2.1e, %2.1e]" % (atom_index, first_diff[1],
                                                                 norm_diff[0, first_diff[1]],
                                                                 norm_diff[1, first_diff[1]])
            errmsg2 = "\nThe position of atom %u differs by > %2.1e for the same frame between trajectories" % (
                atom_index, eps)
            errmsg += errmsg2
            break

    return found_diff, errmsg
