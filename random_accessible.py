from abc import ABCMeta, abstractmethod

import numpy as np
import numbers

import six

from pyemma.coordinates.data.datasource import DataSource


class RandomAccessibleDataSource(DataSource):
    def __init__(self, chunksize=100):
        super(RandomAccessibleDataSource, self).__init__(chunksize)
        self._needs_sorted_random_access_stride = False
        self._default_random_access_strategy = DefaultRandomAccessStrategy(self)
        self._ra_linear_strategy = LinearRandomAccessStrategy(self)
        self._ra_linear_itraj_strategy = LinearItrajRandomAccessStrategy(self)
        self._ra_jagged = JaggedRandomAccessStrategy(self)

    @property
    def ra_itraj_cuboid(self):
        """
        Implementation of random access with slicing that can be up to 3-dimensional, where the first dimension
        corresponds to the trajectory index, the second dimension corresponds to the frames and the third dimension
        corresponds to the dimensions of the frames.

        The with the frame slice selected frames will be loaded from each in the trajectory-slice selected trajectories
        and then sliced with the dimension slice. For example: The data consists out of three trajectories with length
        10, 20, 10, respectively. The slice `data[:, :15, :3]` returns a 3D array of shape (3, 10, 3), where the first
        component corresponds to the three trajectories, the second component corresponds to 10 frames (note that
        the last 5 frames are being truncated as the other two trajectories only have 10 frames) and the third component
        corresponds to the selected first three dimensions.

        :return: Returns an object that allows access by slices in the described manner.
        """
        return self._default_random_access_strategy

    @property
    def ra_itraj_jagged(self):
        """
        Behaves like ra_itraj_cuboid just that the trajectories are not truncated and returned as a list.

        :return: Returns an object that allows access by slices in the described manner.
        """
        return self._ra_jagged

    @property
    def ra_linear(self):
        """
        Implementation of random access that takes a (maximal) two-dimensional slice where the first component
        corresponds to the frames and the second component corresponds to the dimensions. Here it is assumed that
        the frame indexing is contiguous, i.e., the first frame of the second trajectory has the index of the last frame
        of the first trajectory plus one.

        :return: Returns an object that allows access by slices in the described manner.
        """
        return self._ra_linear_strategy

    @property
    def ra_itraj_linear(self):
        """
        Implementation of random access that takes arguments as the default random access (i.e., up to three dimensions
        with trajs, frames and dims, respectively), but which considers the frame indexing to be contiguous. Therefore,
        it returns a simple 2D array.

        :return: A 2D array of the sliced data containing [frames, dims].
        """
        return self._ra_linear_itraj_strategy


class RandomAccessStrategy(six.with_metaclass(ABCMeta)):
    def __init__(self, source):
        self._source = source

    @abstractmethod
    def _handle_slice(self, idx):
        pass

    def __getitem__(self, idx):
        return self._handle_slice(idx)

    def __getslice__(self, start, stop):
        """For slices of the form data[1:3]."""
        return self.__getitem__(slice(start, stop))

    def _get_indices(self, item, length):
        if isinstance(item, slice):
            item = range(*item.indices(length))
        else:
            item = np.arange(0, length)[item]
            if isinstance(item, numbers.Integral):
                item = [item]
        return item

    def _max(self, elems):
        if isinstance(elems, numbers.Integral):
            elems = [elems]
        return max(elems)


class DefaultRandomAccessStrategy(RandomAccessStrategy):
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
        itrajs = self._get_indices(itrajs, self._source.ntraj)
        frames = self._get_indices(frames, min(self._source.trajectory_lengths(1, 0)[itrajs]))
        dims = self._get_indices(dims, self._source.ndim)

        ntrajs = len(itrajs)
        nframes = len(frames)
        ndims = len(dims)

        if max(dims) > self._source.ndim:
            raise IndexError("Data only has %s dimensions, wanted to slice by dimension %s."
                             % (self._source.ndim, max(dims)))

        ra_indices = np.empty((ntrajs * nframes, 2), dtype=int)
        for idx, itraj in enumerate(itrajs):
            ra_indices[idx * nframes: (idx + 1) * nframes, 0] = itraj * np.ones(nframes, dtype=int)
            ra_indices[idx * nframes: (idx + 1) * nframes, 1] = frames

        data = np.empty((ntrajs, nframes, ndims))

        count = 0
        for X in self._source.iterator(stride=ra_indices, lag=0, chunk=0, return_trajindex=False):
            data[count, :, :] = X[:, dims]
            count += 1

        return data


class JaggedRandomAccessStrategy(DefaultRandomAccessStrategy):

    def _get_itraj_random_accessible(self, itrajs, frames, dims):
        itrajs = self._get_indices(itrajs, self._source.ntraj)
        return [self._source._default_random_access_strategy[itraj, frames, dims][0] for itraj in itrajs]


class LinearItrajRandomAccessStrategy(DefaultRandomAccessStrategy):
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
        from pyemma.coordinates.clustering import UniformTimeClustering
        ra = np.array([self._map_to_absolute_traj_idx(UniformTimeClustering._idx_to_traj_idx(x, cumsum), itrajs)
                       for x in frames])

        data = np.empty((nframes, ndims))

        count = 0
        for X in self._source.iterator(stride=ra, lag=0, chunk=0, return_trajindex=False):
            L = len(X)
            data[count:count + L, :] = X[:, dims]
            count += L

        return data

    def _map_to_absolute_traj_idx(self, cumsum_idx, itrajs):
        return itrajs[cumsum_idx[0]], cumsum_idx[1]


class LinearRandomAccessStrategy(RandomAccessStrategy):
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
        frames = self._get_indices(frames, cumsum[-1])
        dims = self._get_indices(dims, self._source.ndim)

        nframes = len(frames)
        ndims = len(dims)

        from pyemma.coordinates.clustering import UniformTimeClustering
        ra_stride = np.array([UniformTimeClustering._idx_to_traj_idx(x, cumsum) for x in frames])
        data = np.empty((nframes, ndims))

        offset = 0
        for X in self._source.iterator(stride=ra_stride, lag=0, chunk=0, return_trajindex=False):
            L = len(X)
            data[offset:offset + L, :] = X[:, dims]
            offset += L
        return data
