import functools
from abc import ABCMeta, abstractmethod

import six
import numpy as np
from math import ceil

from pyemma.coordinates.data.iterable import Iterable


class DataSource(Iterable):

    def __init__(self, chunksize=100):
        super(DataSource, self).__init__(chunksize)
        self._lengths = []
        # storage for arrays (used in _add_array_to_storage)
        self._data = []
        # following properties have to be set in subclass
        self._ntraj = 0
        self._lengths = []
        self._is_reader = False

    @property
    def ntraj(self):
        __doc__ = self.number_of_trajectories.__doc__
        return self._ntraj

    @property
    def is_random_accessible(self):
        from pyemma.coordinates.data.random_accessible import RandomAccessibleDataSource
        return isinstance(self, RandomAccessibleDataSource)

    @property
    def is_reader(self):
        return self._is_reader

    @property
    def data(self):
        return self._data

    @property
    def data_producer(self):
        return self

    def number_of_trajectories(self):
        """
        Returns the number of trajectories

        :return:
            number of trajectories
        """
        return self._ntraj

    def trajectory_length(self, itraj, stride=1, skip=None):
        """
        Returns the length of trajectory

        :param itraj:
            trajectory index
        :param stride:
            return value is the number of frames in trajectory when
            running through it with a step size of `stride`
        :param skip:
            skip parameter
        :return:
            length of trajectory
        """
        if itraj >= self._ntraj:
            raise IndexError
        if isinstance(stride, np.ndarray):
            selection = stride[stride[:, 0] == itraj][:, 0]
            return 0 if itraj not in selection else len(selection)
        else:
            return (self._lengths[itraj] - (0 if skip is None else skip) - 1) // int(stride) + 1

    def trajectory_lengths(self, stride=1, skip=0):
        """
        Returns the length of each trajectory

        :param stride:
            return value is the number of frames in trajectories when
            running through them with a step size of `stride`
        :param skip:
            return value is the number of frames in trajectories when
            skipping the first "skip" frames (plus stride)

        :return:
            numpy array containing length of each trajectory
        """
        if isinstance(stride, np.ndarray):
            return np.array(
                    [self.trajectory_length(itraj, stride) for itraj in range(0, self.number_of_trajectories())],
                    dtype=int)
        else:
            return np.array([(l - skip - 1) // stride + 1 for l in self._lengths], dtype=int)

    def n_frames_total(self, stride=1):
        """
        Returns the total number of frames, over all trajectories

        :param stride:
            return value is the number of frames in trajectories when
            running through them with a step size of `stride`

        :return:
            the total number of frames, over all trajectories
        """
        if isinstance(stride, np.ndarray):
            return stride.shape[0]
        if stride == 1:
            return np.sum(self._lengths)
        else:
            return sum(self.trajectory_lengths(stride))

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


class IteratorState(object):

    def __init__(self, stride=1, skip=0, chunk=0, return_trajindex=False, ntraj=0):
        self.skip = skip
        self.chunk = chunk
        self.return_trajindex = return_trajindex
        self.itraj = 0
        self.pos = 0
        self.__init_stride(stride)

        if not IteratorState.is_uniform_stride(stride):
            # skip trajs which are not included in stride
            while self.itraj not in self.traj_keys and self.itraj < ntraj:
                self.itraj += 1

    def __init_stride(self, stride):
        self.stride = stride
        if isinstance(stride, np.ndarray):
            keys = stride[:, 0]
            self.traj_keys, self.trajectory_lengths = np.unique(keys, return_counts=True)
        else:
            self.traj_keys = None
        self.uniform_stride = IteratorState.is_uniform_stride(stride)
        if not self.uniform_stride and not self.is_stride_sorted():
            raise ValueError("Currently only sorted arrays allowed for random access")

    def ra_indices_for_traj(self, traj):
        """
        Gives the indices for a trajectory file index (without changing the order within the trajectory itself).
        :param traj: a trajectory file index
        :return: a Nx1 - np.array of the indices corresponding to the trajectory index
        """
        assert not self.uniform_stride, "requested random access indices, but is in uniform stride mode"
        return self.stride[self.stride[:, 0] == traj][:, 1] if traj in self.traj_keys else np.array([])

    def ra_trajectory_length(self, traj):
        assert not self.uniform_stride, "requested random access trajectory length, but is in uniform stride mode"
        return int(self.trajectory_lengths[np.where(self.traj_keys == traj)]) if traj in self.traj_keys else 0

    @staticmethod
    def is_uniform_stride(stride):
        return not isinstance(stride, np.ndarray)

    def is_stride_sorted(self):
        if not self.uniform_stride:
            stride_traj_keys = self.stride[:, 0]
            if not all(np.diff(stride_traj_keys) >= 0):
                # traj keys were not sorted
                return False
            for idx in self.traj_keys:
                if not all(np.diff(self.stride[stride_traj_keys == idx][:, 1]) >= 0):
                    # traj indices were not sorted
                    return False
        return True


class DataSourceIterator(six.with_metaclass(ABCMeta)):

    def __init__(self, data_source, skip=0, chunk=0, stride=1, return_trajindex=False):
        self._data_source = data_source
        self.state = IteratorState(stride=stride, skip=skip, chunk=chunk,
                                   return_trajindex=return_trajindex, ntraj=self.number_of_trajectories())

    def ra_indices_for_traj(self, traj):
        """
        Gives the indices for a trajectory file index (without changing the order within the trajectory itself).
        :param traj: a trajectory file index
        :return: a Nx1 - np.array of the indices corresponding to the trajectory index
        """
        return self.state.ra_indices_for_traj(traj)

    def ra_trajectory_length(self, traj):
        return self.state.ra_trajectory_length(traj)

    def is_stride_sorted(self):
        return self.state.is_stride_sorted()

    def _n_chunks(self, stride=None):
        """ rough estimate of how many chunks will be processed """
        stride = stride if stride is not None else self.stride
        if self.chunksize != 0:
            if not DataSourceIterator.is_uniform_stride(stride):
                chunks = ceil(len(stride[:, 0]) / float(self.chunksize))
            else:
                chunks = sum((ceil(l / float(self.chunksize))
                              for l in self.trajectory_lengths()))
        else:
            chunks = 1
        return int(chunks)

    def number_of_trajectories(self):
        return self._data_source.number_of_trajectories()

    def trajectory_length(self):
        return self._data_source.trajectory_length(self.current_trajindex, self.stride, self.skip)

    def trajectory_lengths(self):
        return self._data_source.trajectory_lengths(self.stride, self.skip)

    def n_frames_total(self):
        return self._data_source.n_frames_total(self.stride)

    @abstractmethod
    def close(self):
        pass

    def reset(self):
        self._t = 0
        self._itraj = 0

    @property
    def pos(self):
        return self._t

    @property
    def current_trajindex(self):
        return self._itraj

    @property
    def skip(self):
        return self.state.skip

    @property
    def _t(self):
        return self.state.pos

    @_t.setter
    def _t(self, value):
        self.state.pos = value

    @property
    def _itraj(self):
        return self.state.itraj

    @_itraj.setter
    def _itraj(self, value):
        self.state.itraj = value

    @skip.setter
    def skip(self, value):
        self.state.skip = value

    @property
    def chunksize(self):
        return self.state.chunk

    @chunksize.setter
    def chunksize(self, value):
        if not value >= 0:
            raise ValueError("chunksize has to be positive")
        self.state.chunk = value

    @property
    def stride(self):
        return self.state.stride

    @stride.setter
    def stride(self, value):
        self.state.stride = value

    @property
    def return_traj_index(self):
        return self.state.return_trajindex

    @property
    def traj_keys(self):
        return self.state.traj_keys

    @property
    def uniform_stride(self):
        return self.state.uniform_stride

    @return_traj_index.setter
    def return_traj_index(self, value):
        self.state.return_trajindex = value

    @staticmethod
    def is_uniform_stride(stride):
        return IteratorState.is_uniform_stride(stride)

    def last_chunk(self, itraj):
        return itraj == self.number_of_trajectories() - 1 and self.last_chunk_in_traj(itraj)

    def last_chunk_in_traj(self, itraj):
        if self.chunksize > 0:
            return self._t + self.chunksize >= self._data_source.trajectory_length(itraj, self.stride, self.skip)
        else:
            return True

    @abstractmethod
    def next_chunk(self):
        pass

    def __next__(self):
        return self.next()

    def next(self):
        # we have to obtain the current index before invoking next_chunk (which increments itraj)
        itraj = self.current_trajindex
        X = self.next_chunk()
        if self.return_traj_index:
            return itraj, X
        return X

    def __iter__(self):
        return self
