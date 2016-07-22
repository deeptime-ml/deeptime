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

import itertools

import numpy as np

from pyemma.coordinates.data._base.datasource import DataSource, DataSourceIterator
from pyemma.coordinates.data.util.reader_utils import preallocate_empty_trajectory
from pyemma.util.annotators import fix_docs


class _FragmentedTrajectoryIterator(object):
    def __init__(self, fragmented_reader, readers, chunksize, stride, skip):
        # global time variable
        self._t = 0
        # time variable on a per-reader basis
        self._reader_t = 0
        self._readers = readers
        self._chunksize = chunksize
        self._stride = stride
        self._skip = skip
        self._frag_reader = fragmented_reader
        # lengths array per reader
        self._reader_lengths = [reader.trajectory_length(0, 1) for reader in self._readers]
        # composite trajectory length
        self._lengths = [sum(self._reader_lengths)]
        # mapping reader_index -> cumulative length
        self._cumulative_lengths = np.cumsum(self._reader_lengths)
        # current reader index
        self._reader_at = 0
        self._done = False
        self._ra_indices = None

    @property
    def ra_indices(self):
        return self._ra_indices

    @ra_indices.setter
    def ra_indices(self, value):
        self._ra_indices = value

    def __iter__(self):
        return self

    def _allocate_chunk(self, expected_length, ndim):
        if (hasattr(self._reader_it._data_source, '_return_traj_obj') and
                self._reader_it._data_source._return_traj_obj):
            X = preallocate_empty_trajectory(n_frames=expected_length,
                                             top=self._reader_it._data_source.featurizer.topology)
        else:
            X = np.empty((expected_length, ndim), dtype=self._frag_reader.output_type())

        return X

    def __next__(self):
        skip = self._skip if self._t == 0 else 0
        if self._chunksize == 0:
            X = self._read_full(skip)
            return X
        else:
            # first chunk
            if self._reader_at == 0 and self._t == 0:
                self._reader_overlap = skip
                if self.ra_indices is not None:
                    self._fragment_indices = self.__get_ra_index_indices()
                    self._reader_it = self._readers[self._reader_at].iterator(
                        self.__get_ifrag_ra_indices(self._fragment_indices, 0), return_trajindex=False
                    )
                else:
                    self._reader_it = self._readers[self._reader_at].iterator(self._stride, return_trajindex=False)
            if self.ra_indices is None:
                self._reader_it.skip = self._reader_overlap
            # set original chunksize
            self._reader_it.chunksize = self._chunksize
            # chunk is contained in current reader
            if self.__chunk_contained_in_current_reader():
                self._t += self._chunksize
                self._reader_t += self._chunksize
                X = next(self._reader_it)
                return X
            # chunk has to be collected from subsequent readers
            else:
                ndim = self._readers[0].ndim
                expected_length = self.__get_chunk_expected_length()
                X = self._allocate_chunk(expected_length, ndim)
                read = 0
                while read < expected_length or expected_length == 0:
                    # reader has data left:
                    reader_trajlen = self.__get_reader_trajlen()
                    if reader_trajlen - self._reader_t > 0:
                        chunk = next(self._reader_it)
                        L = len(chunk)
                        X[read:read + L, :] = chunk[:]
                        read += L
                        self._reader_t += L
                    # need new reader
                    if read < expected_length or expected_length == 0:
                        self._reader_at += 1
                        self._reader_t = 0
                        if len(self._readers) <= self._reader_at:
                            raise StopIteration()
                        self._reader_it.close()
                        if self.ra_indices is not None:
                            self._reader_it = self._readers[self._reader_at].iterator(
                                self.__get_ifrag_ra_indices(self._fragment_indices, self._reader_at),
                                return_trajindex=False
                            )
                        else:
                            self._reader_it = self._readers[self._reader_at].iterator(self._stride, return_trajindex=False)
                            self._reader_it.skip = skip
                            self._reader_overlap = self._calculate_new_overlap(self._stride,
                                                                               self._reader_lengths[self._reader_at - 1],
                                                                               self._reader_overlap)
                            self._reader_it.skip = self._reader_overlap
                        if expected_length - read > 0:
                            self._reader_it.chunksize = expected_length - read
                self._t += read
                return X

    def __get_reader_trajlen(self):
        if self.ra_indices is not None:
            reader_trajlen = self._readers[self._reader_at].trajectory_length(
                0,
                self.__get_ifrag_ra_indices(self._fragment_indices, self._reader_at),
                self._skip if self._reader_at == 0 else 0
            )
        else:
            reader_trajlen = self._readers[self._reader_at].trajectory_length(
                0, self._stride,
                self._skip if self._reader_at == 0 else 0
            )
        return reader_trajlen

    def __get_chunk_expected_length(self):
        if self.ra_indices is not None:
            expected_length = min(self._chunksize, len(self.ra_indices) - self._t)
        else:
            expected_length = min(self._chunksize, sum(self._traj_lengths(self._stride)) - self._t)
        return expected_length

    def __chunk_contained_in_current_reader(self):
        trajlen = self.__get_reader_trajlen()
        return trajlen - self._reader_t - self._chunksize > 0

    def next(self):
        return self.__next__()

    def _allocate_chunk(self, expected_length, ndim):
        from pyemma.coordinates.data.feature_reader import FeatureReader
        if all(isinstance(r, FeatureReader) and r._return_traj_obj for r in self._readers):
            X = preallocate_empty_trajectory(n_frames=expected_length,
                                             top=self._readers[0].featurizer.topology)
        else:
            X = np.empty((expected_length, ndim), dtype=self._frag_reader.output_type())

        return X

    def _read_full(self, skip):
        if self._ra_indices is not None:
            fragment_indices = self.__get_ra_index_indices()
            ndim = self._readers[0].ndim
            length = len(self.ra_indices)
            X = self._allocate_chunk(length, ndim)
            L = 0
            for ifrag, r in enumerate(self._readers):
                indices = self.__get_ifrag_ra_indices(fragment_indices, ifrag)
                if len(indices) > 0:
                    ifrag_data = None
                    for ifrag_data in r.iterator(chunk=0, stride=indices, return_trajindex=False):
                        pass
                    l = len(ifrag_data)
                    X[L:L+l, :] = ifrag_data[:]
                    L += l
            return X
        else:
            overlap = skip
            self._skip = overlap
            ndim = len(np.zeros(self._readers[0].dimension())[0::])
            length = sum(self._traj_lengths(self._stride))
            X = self._allocate_chunk(length, ndim)
            for idx, r in enumerate(self._readers):
                _skip = overlap
                # if stride doesn't divide length, one has to offset the next trajectory
                overlap = self._calculate_new_overlap(self._stride, self._reader_lengths[idx], overlap)
                chunksize = min(length, r.trajectory_length(0, self._stride, skip=_skip))
                it = r._create_iterator(stride=self._stride, skip=_skip, chunk=chunksize, return_trajindex=True)
                with it:
                    for itraj, data in it:
                        L = len(data)
                        if L > 0:
                            X[self._t:self._t + L, :] = data[:]
                        self._t += L
            return X

    def __get_ifrag_ra_indices(self, fragment_indices, ifrag):
        offset = self._cumulative_lengths[ifrag - 1] if ifrag > 0 else 0
        ra = self.ra_indices[fragment_indices[ifrag]] - offset
        indices = np.zeros((len(ra), 2), dtype=int)
        indices[:, 1] = ra.squeeze()
        return indices

    def __get_ra_index_indices(self):
        """
        Returns a list containing indices of the ra_index array, which correspond to the separate trajectory fragments,
        i.e., ra_indices[fragment_indices[itraj]] are the ra indices for itraj (plus some offset by
        cumulative length)
        """
        fragment_indices = []
        for idx, cumlen in enumerate(self._cumulative_lengths):
            cumlen_prev = self._cumulative_lengths[idx - 1] if idx > 0 else 0
            fragment_indices.append([np.argwhere(
                np.logical_and(self.ra_indices >= cumlen_prev, self.ra_indices < cumlen)
            )])
        return fragment_indices

    def _traj_lengths(self, stride):
        return np.array([(l - self._skip - 1) // stride + 1 for l in self._lengths], dtype=int)

    @staticmethod
    def _calculate_new_overlap(stride, traj_len, skip):
        """
        Given two trajectories T_1 and T_2, this function calculates for the first trajectory an overlap, i.e.,
        a skip parameter for T_2 such that the trajectory fragments T_1 and T_2 appear as one under the given stride.

        Idea for deriving the formula: It is

        K = ((traj_len - skip - 1) // stride + 1) = #(data points in trajectory of length (traj_len - skip)).

        Therefore, the first point's position that is not contained in T_1 anymore is given by

        pos = skip + s * K.

        Thus the needed skip of T_2 such that the same stride parameter makes T_1 and T_2 "look as one" is

        overlap = pos - traj_len.

        :param stride: the (global) stride parameter
        :param traj_len: length of T_1
        :param skip: skip of T_1
        :return: skip of T_2
        """
        overlap = stride * ((traj_len - skip - 1) // stride + 1) - traj_len + skip
        return overlap

    def close(self):
        if hasattr(self, '_reader_it'):
            self._reader_it.close()


class FragmentIterator(DataSourceIterator):
    """
    outer iterator, which encapsulates _FragmentedTrajectoryIterator
    """

    def __init__(self, data_source, skip=0, chunk=0, stride=1, return_trajindex=False, cols=None):
        super(FragmentIterator, self).__init__(data_source, skip=skip, chunk=chunk,
                                               stride=stride, return_trajindex=return_trajindex,
                                               cols=cols)
        self._it = None
        self._itraj = 0

    def _next_chunk(self):
        if self._it is None:
            if self._itraj < self.number_of_trajectories():
                self._it = _FragmentedTrajectoryIterator(self._data_source, self._data_source._readers[self._itraj],
                                                         self.chunksize, self.stride, self.skip)
                if not self.uniform_stride:
                    self._it.ra_indices = self.ra_indices_for_traj(self._itraj)
            else:
                raise StopIteration()

        X = next(self._it, None)
        if X is None:
            raise StopIteration()
        self._t += len(X)
        if self._t >= self._data_source.trajectory_length(self._itraj, stride=self.stride, skip=self.skip):
            self._itraj += 1
            self._it.close()
            self._it = None
            self._t = 0
        while (not self.uniform_stride) and (self._itraj not in self.traj_keys or self._t >= self.ra_trajectory_length(self._itraj)) \
                and self._itraj < self.number_of_trajectories():
            self._itraj += 1
        return X

    def close(self):
        if self._it is not None:
            self._it.close()


@fix_docs
class FragmentedTrajectoryReader(DataSource):
    """
    Parameters
    ----------
    trajectories: nested list or nested tuple, 1 level depth
    
    topologyfile, str, default None
    
    chunksize: int, default 1000
    
    featurizer: MDFeaturizer, default None
    
    """

    def __init__(self, trajectories, topologyfile=None, chunksize=1000, featurizer=None):
        # sanity checks
        assert isinstance(trajectories, (list, tuple)), "input trajectories should be of list or tuple type"
        # if it contains no further list: treat as single trajectory
        if not any([isinstance(traj, (list, tuple)) for traj in trajectories]):
            trajectories = [trajectories]
        # if not list of lists, treat as single-element-fragment-trajectory
        trajectories = [traj if isinstance(traj, (list, tuple)) else [traj] for traj in trajectories]
        # some trajectory should be provided
        assert len(trajectories) > 0, "no input trajectories provided"
        # call super
        super(FragmentedTrajectoryReader, self).__init__(chunksize=chunksize)
        self._is_reader = True
        # number of trajectories
        self._ntraj = len(trajectories)
        # store readers
        from pyemma.coordinates.api import source

        self._readers = [[source(input_item, features=featurizer, top=topologyfile, chunk_size=chunksize)
                          for input_item in trajectories[itraj]] for itraj in range(0, self._ntraj)]

        # check all readers have same dimension
        if not len(set(itraj_r.ndim for r in self._readers for itraj_r in r)) == 1:
            # lookup the evil reader:
            last_dim = -1
            for r in self._readers:
                for itraj_r in r:
                    if last_dim == -1:
                        last_dim = itraj_r.ndim
                    if itraj_r.ndim != last_dim:
                        raise ValueError("%s has different dimension (%i) than expected (%i)"
                                         % (itraj_r.describe(), itraj_r.ndim, last_dim))

        self._reader_by_filename = {}
        for r in self._readers:
            for itraj_r in r:
                for filename in itraj_r.filenames:
                    if filename in self._reader_by_filename:
                        self._reader_by_filename[filename].append(itraj_r)
                    else:
                        self._reader_by_filename[filename] = [itraj_r]

        # lengths array per reader
        self._reader_lengths = [[reader.trajectory_length(0, 1)
                                 for reader in self._readers[itraj]] for itraj in range(0, self._ntraj)]
        # composite trajectory length
        self._lengths = [sum(self._reader_lengths[itraj]) for itraj in range(0, self._ntraj)]
        # mapping reader_index -> cumulative length
        self._cumulative_lengths = [np.cumsum(self._reader_lengths[itraj]) for itraj in range(0, self._ntraj)]
        # store trajectory files
        self._trajectories = trajectories
        self._filenames = trajectories

        # random-accessible
        #self._is_random_accessible = all(r._is_random_accessible for r in self._readers[itraj]
        #                                 for itraj in range(0, self._ntraj))

    @property
    def filenames_flat(self):
        flat_readers = itertools.chain.from_iterable(self._readers)
        return [f for flat_reader in flat_readers for f in flat_reader.filenames]

    def reader_by_filename(self, filename):
        res = self._reader_by_filename[filename]
        if isinstance(res, list) and len(res) == 1:
            res = res[0]
        return res

    def _create_iterator(self, skip=0, chunk=0, stride=1, return_trajindex=True, cols=None):
        return FragmentIterator(self, skip, chunk, stride, return_trajindex, cols=cols)

    def describe(self):
        return "[FragmentedTrajectoryReader files=%s]" % self._trajectories

    def dimension(self):
        return self._readers[0][0].dimension()

    def _index_to_reader_index(self, index, itraj):
        """
        Accepts an index parameter in [0, sum(reader_lenghts)) and returns a tuple (reader_index, local_index),
        where the tuple (reader_index, local_index) corresponds to the global frame index of the fragmented trajectory.
        :param index: the global index
        :return: a tuple (reader_index, local_index)
        """
        prev_len = 0
        for readerIndex, length in enumerate(self._cumulative_lengths[itraj]):
            if prev_len <= index < length:
                return readerIndex, index - prev_len
            prev_len = length
        raise ValueError("Requested index %s was out of bounds [0,%s)" % (index, self._cumulative_lengths[itraj][-1]))

    def _get_traj_info(self, filename):
        # get info for a fragment from specific reader
        reader = self._reader_by_filename[filename]
        return reader._get_traj_info(filename)
