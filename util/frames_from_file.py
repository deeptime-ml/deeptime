
# This file is part of PyEMMA.
#
# Copyright (c) 2016, 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
from __future__ import absolute_import

import itertools
from logging import getLogger

import mdtraj as md
import numpy as np

from pyemma.coordinates import source
from pyemma.coordinates.data import FeatureReader
from pyemma.coordinates.data.fragmented_trajectory_reader import FragmentedTrajectoryReader
from pyemma.coordinates.data.util.reader_utils import (copy_traj_attributes as _copy_traj_attributes,
                                                       preallocate_empty_trajectory as _preallocate_empty_trajectory,
                                                       enforce_top as _enforce_top)
from pyemma.util.annotators import deprecated

__all__ = ['frames_from_files']

log = getLogger(__name__)


def frames_from_files(files, top, frames, chunksize=1000, stride=1, verbose=False, copy_not_join=None, reader=None):
    """
    Constructs a Trajectory object out of given frames collected from files (or given reader).

    :param files: source files
    :param top: topology file
    :param frames: indices
    :param chunksize:
    :param stride:
    :param verbose:
    :param copy_not_join: not used
    :param reader: if a reader is given, ignore files and top param!
    :return: mdtra.Trajectory consisting out of frames indices.
    """
    # Enforce topology to be a md.Topology object
    if reader is None:
        top = _enforce_top(top)
    else:
        if not reader.number_of_trajectories():
            raise ValueError("need at least one trajectory file in reader.")
        if isinstance(reader, FragmentedTrajectoryReader):
            top = reader._readers[0][0].featurizer.topology
        elif isinstance(reader, FeatureReader):
            top = reader.featurizer.topology
        else:
            raise ValueError("unsupported reader (only md readers).")

    stride = int(stride)
    frames = np.array(frames)

    # only one file, so we expect frames to be a one dimensional array
    if isinstance(files, str):
        files = [files]
        if frames.ndim == 1:
            # insert a constant column for file index
            frames = np.insert(np.atleast_2d(frames), 0, np.zeros_like(frames), axis=0).T

    if stride != 1:
        frames[:, 1] *= int(stride)
        if verbose:
            log.info('A stride value of = %u was parsed, '
                     'interpreting "indexes" accordingly.' % stride)

    # sort by file and frame index
    sort_inds = np.lexsort((frames[:, 1], frames[:, 0]))
    sorted_inds = frames[sort_inds]
    assert len(sorted_inds) == len(frames)

    file_inds_unique = np.unique(sorted_inds[:, 0])
    # construct reader
    if reader is None:
        # filter out files, we would never read, because no indices are pointing to them
        reader = source(np.array(files)[file_inds_unique].tolist(), top=top)
        # re-map indices to reflect filtered files:
        for itraj, c in zip(file_inds_unique, itertools.count(0)):
            mask = sorted_inds[:, 0] == itraj
            sorted_inds[mask, 0] = c

        inds_to_check = np.arange(len(file_inds_unique))
    else:
        inds_to_check = file_inds_unique

    # sanity check of indices
    for itraj in inds_to_check:
        inds_by_traj = sorted_inds[sorted_inds[:, 0] == itraj]
        largest_ind_in_traj = np.max(inds_by_traj)
        length = reader.trajectory_length(itraj)
        if length < largest_ind_in_traj:
            raise ValueError("largest specified index (%i * stride=%i * %i=%i) "
                             "is larger than trajectory length '%s' = %i" %
                             (largest_ind_in_traj/stride, largest_ind_in_traj/stride,
                              stride, largest_ind_in_traj, reader.filenames[itraj],
                              length))

    # we want the FeatureReader to return mdtraj.Trajectory objects
    if isinstance(reader, FeatureReader):
        reader._return_traj_obj = True
    elif isinstance(reader, FragmentedTrajectoryReader):
        for file in reader.filenames_flat:
            r = reader.reader_by_filename(file)
            if isinstance(r, FeatureReader):
                r = [r]
            for _r in r:
                _r._return_traj_obj = True

    it = reader.iterator(chunk=chunksize, stride=sorted_inds, return_trajindex=False)
    collected_frames = []
    with it:
        for x in it:
            collected_frames.append(x)

    dest = _preallocate_empty_trajectory(top, len(frames))
    i = 0
    for chunk in collected_frames:
        _copy_traj_attributes(dest, chunk, i)
        i += len(chunk)
    dest = dest.slice(sort_inds.argsort(), copy=False)
    return dest


@deprecated("use_frames_from_files")
def frames_from_file(file_name, top, frames, chunksize=100,
                     stride=1, verbose=False, copy_not_join=False):
    r"""Reads one "file_name" molecular trajectory and returns an mdtraj trajectory object 
        containing only the specified "frames" in the specified order.

    Extracts the specified sequence of time/trajectory indexes from the input loader
    and saves it in a molecular dynamics trajectory. The output format will be determined
    by the outfile name.

    Parameters
    ----------
    file_name: str.
        Absolute path to the molecular trajectory file, ex. trajout.xtc 

    top : str, mdtraj.Trajectory, or mdtraj.Topology
        Topology information to load the molecular trajectroy file in :py:obj:`file_name`

    frames : ndarray of shape (n_frames, ) and integer type
        Contains the frame indices to be retrieved from "file_name". There is no restriction as to what 
        this array has to look like other than:
             - positive integers
             - <= the total number of frames in "file_name".
        "frames" need not be monotonous or unique, i.e, arrays like
        [3, 1, 4, 1, 5, 9, 9, 9, 9, 3000, 0, 0, 1] are welcome 

    verbose: boolean.
        Level of verbosity while looking for "frames". Useful when using "chunksize" with large trajectories.
        It provides the no. of frames accumulated for every chunk.

    stride  : integer, default is 1
        This parameter informs :py:func:`save_traj` about the stride used in :py:obj:`indexes`. Typically, :py:obj:`indexes`
        contains frame-indexes that match exactly the frames of the files contained in :py:obj:`traj_inp.trajfiles`.
        However, in certain situations, that might not be the case. Examples are cases in which a stride value != 1
        was used when reading/featurizing/transforming/discretizing the files contained in :py:obj:`traj_inp.trajfiles`.

    copy_not_join : boolean, default is False
        This parameter decides how geometry objects are appended onto one another. If left to False, mdtraj's own
        :py:obj:`join` method will be used, which is the recommended method. However, for some combinations of
        py:obj:`chunksizes` and :py:obj:`frames` this might be not very effective. If one sets :py:obj:`copy_not_join`
        to True, the returned :py:obj:`traj` is preallocated and the important attributes (currently traj.xyz, traj.time,
         traj.unit_lengths, traj.unit_angles) are broadcasted onto it.


    Returns
    -------
    traj : an md trajectory object containing the frames specified in "frames",
           in the order specified in "frames".
    """

    assert isinstance(frames, np.ndarray), "input frames frames must be a numpy ndarray, got %s instead "%type(frames)
    assert np.ndim(frames) == 1, "input frames frames must have ndim = 1, got np.ndim = %u instead "%np.ndim(frames)
    assert isinstance(file_name, str), "input file_name must be a string, got %s instead"%type(file_name)
    assert isinstance(top, (str, md.Trajectory, md.Topology)), "input topology must of one of type: " \
                                                                    "str, mdtraj.Trajectory, or mdtraj.Topology. " \
                                                                    "Got %s instead" % type(top)

    return frames_from_files([file_name], top, frames, chunksize, stride, verbose)
