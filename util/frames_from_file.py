
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
from __future__ import absolute_import

import mdtraj as md
import numpy as np
from logging import getLogger
from pyemma.coordinates.data.util.reader_utils import copy_traj_attributes as _copy_traj_attributes, \
    preallocate_empty_trajectory as _preallocate_empty_trajectory, enforce_top as _enforce_top
from mdtraj.core.trajectory import Trajectory
__all__ = ['frames_from_file']

log = getLogger(__name__)


def frames_from_files(files, top, frames, chunksize=1000, stride=1, verbose=False, copy_not_join=None):
    from pyemma.coordinates import source
    # Enforce topology to be a md.Topology object
    top = _enforce_top(top)
    reader = source(files, top=top)
    stride = int(stride)

    if stride != 1:
        frames[:, 1] *= int(stride)
        if verbose:
            log.info('A stride value of = %u was parsed, '
                     'interpreting "indexes" accordingly.' % stride)

    # sort by file and frame index
    sort_inds = np.lexsort((frames[:, 1], frames[:, 0]))
    sorted_inds = frames[sort_inds]
    assert len(sorted_inds) == len(frames)

    for u in np.unique(sorted_inds[:, 0]):
        largest_ind_in_traj = np.max(sorted_inds[sorted_inds == u])
        if reader.trajectory_length(u) < largest_ind_in_traj:
            raise ValueError("largest specified index (%i * stride=%i * %i=%i) "
                             "is larger than trajectory length '%s' = %i"
                             (largest_ind_in_traj/stride, largest_ind_in_traj/stride,
                              stride, largest_ind_in_traj, reader.filenames[u],
                              reader.trajectory_length(u)))

    collected_frames = []
    with reader.iterator(chunk=chunksize, stride=sorted_inds, return_trajindex=False) as it:
        for x in it:
            collected_frames.append(x)

    collected_frames = np.vstack(collected_frames)
    collected_frames = collected_frames[sort_inds.argsort()]
    collected_frames = collected_frames.reshape(-1, top.n_atoms, 3)

    return Trajectory(collected_frames, top)


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
    # Enforce topology to be a md.Topology object
    top = _enforce_top(top)

    # Prepare the trajectory object
    if copy_not_join:
        traj = _preallocate_empty_trajectory(top, frames.shape[0])
    else:
        traj = None

    # Prepare the running number of accumulated frames
    cum_frames = 0

    # Because the trajectory is streamed "chronologically", but "frames" can have any arbitrary order
    # we store that order in "orig_order" to reshuffle the traj at the end
    orig_order = frames.argsort().argsort()
    sorted_frames = np.sort(frames)

    for jj, traj_chunk in enumerate(md.iterload(file_name, top=top,
                                                chunk=chunksize, stride=stride)):

        # Create an indexing array for this trajchunk
        i_idx = jj*chunksize
        f_idx = i_idx+chunksize
        chunk_frames = np.arange(i_idx, f_idx)[:traj_chunk.n_frames]

        # Frames that appear more than one time will be kept
        good_frames = np.hstack([np.argwhere(ff == chunk_frames).squeeze() for ff in sorted_frames])

        # Keep the good frames of this chunk
        if np.size(good_frames) > 0:

            if copy_not_join:   # => traj has been already preallocated, see above
                traj = _copy_traj_attributes(traj, traj_chunk[good_frames], cum_frames)
            elif traj is None: # => copy_not_join is False AND 1st run
                traj = traj_chunk[good_frames]
            else: # => copy_not_join is False AND we're not on the 1st run
                traj = traj.join(traj_chunk[good_frames])

            cum_frames += np.size(good_frames)

        if verbose:
            log.info('chunk %u of traj has size %u, indices %6u...%6u. Accumulated frames %u'
                 % (jj, traj_chunk.n_frames, chunk_frames[0], chunk_frames[-1], cum_frames))

        # Check if we can already stop iterating
        if chunk_frames[-1] >= frames.max():
            break

    # Make sure that "frames" did not contain impossible frames
    if (frames > chunk_frames[-1]).any():
        raise Exception('Cannot provide frames %s for trajectory %s with n_frames = %u'
                        % (frames[frames > chunk_frames[-1]], file_name, chunk_frames[-1]))

    if stride != 1 and verbose:
        log.info('A stride value of = %u was parsed, interpreting "indexes" accordingly.'%stride)

    # Trajectory coordinates are is returned "reshuffled"
    return traj[orig_order]