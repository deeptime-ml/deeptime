
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

import numpy as np

from pyemma.coordinates import source
from pyemma.coordinates.data import FeatureReader
from pyemma.coordinates.data.fragmented_trajectory_reader import FragmentedTrajectoryReader
from pyemma.coordinates.data.util.reader_utils import (copy_traj_attributes as _copy_traj_attributes,
                                                       preallocate_empty_trajectory as _preallocate_empty_trajectory,
                                                       enforce_top as _enforce_top)

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
        reader_given = False
    else:
        if not reader.number_of_trajectories():
            raise ValueError("need at least one trajectory file in reader.")
        if isinstance(reader, FragmentedTrajectoryReader):
            top = reader._readers[0][0].featurizer.topology
        elif isinstance(reader, FeatureReader):
            top = reader.featurizer.topology
        else:
            raise ValueError("unsupported reader (only md readers).")
        reader_given = True

    stride = int(stride)
    frames = np.array(frames)

    # only one file, so we expect frames to be a one dimensional array
    if isinstance(files, str):
        files = [files]
        if frames.ndim == 1:
            # insert a constant column for file index
            frames = np.insert(np.atleast_2d(frames), 0, np.zeros_like(frames), axis=0).T

    if stride != 1:
        frames[:, 1] *= stride
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
        inds_by_traj = sorted_inds[sorted_inds[:, 0] == itraj][:, 1]
        assert inds_by_traj.ndim == 1
        largest_ind_in_traj = np.max(inds_by_traj)
        length = reader.trajectory_length(itraj)
        if largest_ind_in_traj >= length:
            raise ValueError("largest specified index ({largest_without_stride} * stride="
                             "{largest_without_stride} * {stride}={largest}) "
                             "is larger than trajectory length '{filename}' = {length}".format(
                                largest_without_stride=largest_ind_in_traj / stride,
                                stride=stride,
                                largest=largest_ind_in_traj,
                                filename=reader.filenames[itraj],
                                length=length))

    def set_reader_return_traj_objects(reader, flag):
        if isinstance(reader, FeatureReader):
            reader._return_traj_obj = flag
        elif isinstance(reader, FragmentedTrajectoryReader):
            for file in reader.filenames_flat:
                r = reader.reader_by_filename(file)
                if isinstance(r, FeatureReader):
                    r = [r]
                for _r in r:
                    _r._return_traj_obj = flag

    # we want the FeatureReader to return mdtraj.Trajectory objects
    set_reader_return_traj_objects(reader, True)

    try:
        it = reader.iterator(chunk=chunksize, stride=sorted_inds, return_trajindex=False)
        with it:
            collected_frames = [f for f in it]
        dest = _preallocate_empty_trajectory(top, len(frames))
        t = 0
        for chunk in collected_frames:
            _copy_traj_attributes(dest, chunk, t)
            t += len(chunk)
        # reshuffle the indices of the final trajectory object to obtain the desired order
        dest = dest.slice(sort_inds.argsort(), copy=False)
    finally:
        # in any case we want to reset the reader to its previous state (return features, instead of md.Trajectory)
        if reader_given:
            set_reader_return_traj_objects(reader, False)
    return dest
