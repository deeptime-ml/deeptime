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

import functools
import numbers

import numpy as np

from pyemma.coordinates.data._base.datasource import DataSourceIterator, DataSource
from pyemma.coordinates.data._base.random_accessible import RandomAccessStrategy
from pyemma.util.annotators import fix_docs

__author__ = 'noe, marscher'


@fix_docs
class DataInMemory(DataSource):
    r"""
    multi-dimensional data fully stored in memory.

    Used to pass arbitrary coordinates to pipeline. Data is being flattened to
    two dimensions to ensure it is compatible.

    Parameters
    ----------
    data : ndarray (nframe, ndim) or list of ndarrays (nframe, ndim)
        Data has to be either one 2d array which stores amount of frames in first
        dimension and coordinates/features in second dimension or a list of this
        arrays.
    """
    IN_MEMORY_FILENAME = '<in_memory_file>'

    def _create_iterator(self, skip=0, chunk=0, stride=1, return_trajindex=False, cols=None):
        return DataInMemoryIterator(self, skip, chunk, stride, return_trajindex, cols)

    def __init__(self, data, chunksize=5000, **kw):
        super(DataInMemory, self).__init__(chunksize=chunksize)
        self._is_reader = True
        self._is_random_accessible = True

        self._ra_cuboid = DataInMemoryCuboidRandomAccessStrategy(self, 3)
        self._ra_jagged = DataInMemoryJaggedRandomAccessStrategy(self, 3)
        self._ra_linear_strategy = DataInMemoryLinearRandomAccessStrategy(self, 2)
        self._ra_linear_itraj_strategy = DataInMemoryLinearItrajRandomAccessStrategy(self, 3)

        if not isinstance(data, (list, tuple)):
            data = [data]

        # storage for arrays (used in _add_array_to_storage)
        self._data = []

        # everything is an array
        if all(isinstance(d, np.ndarray) for d in data):
            for d in data:
                self._add_array_to_storage(d)
        else:
            raise ValueError("Please supply numpy.ndarray, or list/tuple of ndarray."
                             " Your input was %s" % str(data))

        self._set_dimensions_and_lenghts()
        self._filenames = [DataInMemory.IN_MEMORY_FILENAME] * self._ntraj

    @property
    def data(self):
        """
        Property that returns the data that was hold in storage (data in memory mode).
        Returns
        -------
        list : The stored data.
        """
        return self._data

    def _add_array_to_storage(self, array):
        """
        checks shapes, eg convert them (2d), raise if not possible
        after checks passed, add array to self._data
        """
        if array.ndim == 1:
            array = np.atleast_2d(array).T
        elif array.ndim == 2:
            pass
        else:
            shape = array.shape
            # hold first dimension, multiply the rest
            shape_2d = (shape[0], functools.reduce(lambda x, y: x * y, shape[1:]))
            array = np.reshape(array, shape_2d)

        self.data.append(array)

    def _set_dimensions_and_lenghts(self):
        # number of trajectories/data sets
        self._ntraj = len(self.data)
        if self.ntraj == 0:
            raise ValueError("no valid data")

        # this works since everything is flattened to 2d
        self._lengths = [np.shape(d)[0] for d in self.data]

        # ensure all trajs have same dim
        ndims = [np.shape(x)[1] for x in self.data]
        if not np.unique(ndims).size == 1:
            raise ValueError("input data has different dimensions!"
                             "Dimensions are = %s" % ndims)

        self._ndim = ndims[0]

    @classmethod
    def load_from_files(cls, files):
        """ construct this by loading all files into memory

        Parameters
        ----------
        files: str or list of str
            filenames to read from
        """
        # import here to avoid cyclic import
        from pyemma.coordinates.data.numpy_filereader import NumPyFileReader

        reader = NumPyFileReader(files)
        data = reader.get_output()
        return cls(data)

    def describe(self):
        return "[DataInMemory array shapes: %s]" % [np.shape(x) for x in self.data]


class DataInMemoryCuboidRandomAccessStrategy(RandomAccessStrategy):
    def _handle_slice(self, idx):
        idx = np.index_exp[idx]
        itrajs, frames, dims = None, None, None
        if isinstance(idx, (list, tuple)):
            if len(idx) == 1:
                itrajs, frames, dims = idx[0], slice(None, None, None), slice(None, None, None)
            if len(idx) == 2:
                itrajs, frames, dims = idx[0], idx[1], slice(None, None, None)
            if len(idx) == 3:
                itrajs, frames, dims = idx[0], idx[1], idx[2]
            if len(idx) > 3 or len(idx) == 0:
                raise IndexError("invalid slice by %s" % idx)
        return self._get_itraj_random_accessible(itrajs, frames, dims)

    def _get_itraj_random_accessible(self, itrajs, frames, dims):
        dims = [dims] if isinstance(dims, numbers.Integral) else dims
        itrajs = self._get_indices(itrajs, self._source.ntraj)
        frames = self._get_indices(frames, min(self._source.trajectory_lengths(1, 0)[itrajs]))
        if isinstance(dims, (list, tuple)):
            return np.array(
                    [self._source.data[itraj][frames] for itraj in itrajs],
                    dtype=self._source.output_type()
            )[:, :, dims]
        return np.array([self._source.data[itraj][frames, dims] for itraj in itrajs], dtype=self._source.output_type())


class DataInMemoryJaggedRandomAccessStrategy(DataInMemoryCuboidRandomAccessStrategy):
    def _get_itraj_random_accessible(self, itrajs, frames, dims):
        itrajs = self._get_indices(itrajs, self._source.ntraj)
        return [self._source.data[itraj][frames, dims] for itraj in itrajs]


class DataInMemoryLinearRandomAccessStrategy(RandomAccessStrategy):
    def _handle_slice(self, idx):
        idx = np.index_exp[idx]
        frames, dims = None, None
        if isinstance(idx, (tuple, list)):
            if len(idx) == 1:
                frames, dims = idx[0], slice(None, None, None)
            if len(idx) == 2:
                frames, dims = idx[0], idx[1]
            if len(idx) > 2:
                raise IndexError("Slice was more than two-dimensional, not supported.")

        cumsum = np.cumsum(self._source.trajectory_lengths())
        if not isinstance(frames, (list, np.ndarray)):
            frames = self._get_indices(frames, cumsum[-1])
        dims = self._get_indices(dims, self._source.ndim)

        nframes = len(frames)
        ndims = len(dims)

        data = np.empty((nframes, ndims), dtype=self._source.output_type())

        from pyemma.coordinates.clustering import UniformTimeClustering
        for i, x in enumerate(frames):
            traj, idx = UniformTimeClustering._idx_to_traj_idx(x, cumsum)
            data[i, :] = self._source.data[traj][idx, dims]
        return data


class DataInMemoryLinearItrajRandomAccessStrategy(DataInMemoryCuboidRandomAccessStrategy):
    def _get_itraj_random_accessible(self, itrajs, frames, dims):
        itrajs = self._get_indices(itrajs, self._source.ntraj)
        frames = self._get_indices(frames, sum(self._source.trajectory_lengths()[itrajs]))
        dims = self._get_indices(dims, self._source.ndim)

        nframes = len(frames)
        ndims = len(dims)

        if max(dims) > self._source.ndim:
            raise IndexError("Data only has %s dimensions, wanted to slice by dimension %s."
                             % (self._source.ndim, max(dims)))

        cumsum = np.cumsum(self._source.trajectory_lengths()[itrajs])
        data = np.empty((nframes, ndims), dtype=self._source.output_type())

        from pyemma.coordinates.clustering import UniformTimeClustering
        for i, x in enumerate(frames):
            traj, idx = self._map_to_absolute_traj_idx(UniformTimeClustering._idx_to_traj_idx(x, cumsum), itrajs)
            data[i, :] = self._source.data[traj][idx, dims]

        return data

    @staticmethod
    def _map_to_absolute_traj_idx(cumsum_idx, itrajs):
        return itrajs[cumsum_idx[0]], cumsum_idx[1]


class DataInMemoryIterator(DataSourceIterator):
    def close(self):
        pass

    def __init__(self, data_source, skip=0, chunk=0, stride=1, return_trajindex=False, cols=None):
        super(DataInMemoryIterator, self).__init__(data_source, skip, chunk,
                                                   stride, return_trajindex, cols)

    def _next_chunk(self):
        if self._itraj >= self._data_source.ntraj:
            raise StopIteration()

        traj_len = self._data_source._lengths[self._itraj]
        traj = self._data_source.data[self._itraj]

        # only apply _skip at the beginning of each trajectory
        skip = self.skip if self._t == 0 else 0

        # complete trajectory mode
        if self.chunksize == 0:
            if not self.uniform_stride:
                chunk = self._data_source.data[self._itraj][self.ra_indices_for_traj(self._itraj)]
                self._itraj += 1
                # skip trajs which are not included in stride
                while self._itraj not in self.traj_keys and self._itraj < self.number_of_trajectories():
                    self._itraj += 1
                return chunk
            else:
                chunk = traj[skip::self.stride]
                self._itraj += 1
                return chunk
        # chunked mode
        else:
            if not self.uniform_stride:
                random_access_chunk = self._data_source.data[self._itraj][
                    self.ra_indices_for_traj(self._itraj)[self._t:min(
                            self._t + self.chunksize, self.ra_trajectory_length(self._itraj)
                    )]
                ]
                self._t += self.chunksize
                if self._t >= self.ra_trajectory_length(self._itraj):
                    self._itraj += 1
                    self._t = 0

                # skip trajs which are not included in stride
                while (self._itraj not in self.traj_keys or self._t >= self.ra_trajectory_length(self._itraj)) \
                        and self._itraj < self.number_of_trajectories():
                    self._itraj += 1
                    self._t = 0
                return random_access_chunk
            else:
                upper_bound = min(skip + self._t + self.chunksize * self.stride, traj_len)
                slice_x = slice(skip + self._t, upper_bound, self.stride)
                chunk = traj[slice_x]

                self._t = upper_bound

                if upper_bound >= traj_len:
                    self._itraj += 1
                    self._t = 0

                return chunk
