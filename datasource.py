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
        n = self.number_of_trajectories()
        if isinstance(stride, np.ndarray):
            return np.fromiter((self.trajectory_length(itraj, stride)
                                for itraj in range(n)),
                               dtype=int, count=n)
        else:
            return np.fromiter(((l - skip - 1) // stride + 1 for l in self._lengths),
                               dtype=int, count=n)

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
        self._current_itraj = 0
        self._t = 0
        self._pos = 0
        self._pos_adv = 0
        self.stride = None
        self.uniform_stride = False
        self.traj_keys = None
        self.trajectory_lengths = None

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
        self.state = IteratorState(skip=skip, chunk=chunk,
                                   return_trajindex=return_trajindex, ntraj=self.number_of_trajectories())
        self.__init_stride(stride)
        self._pos = 0
        self._last_chunk_in_traj = False

    def __init_stride(self, stride):
        self.state.stride = stride
        if isinstance(stride, np.ndarray):
            keys = stride[:, 0]
            self.state.traj_keys, self.state.trajectory_lengths = np.unique(keys, return_counts=True)
        else:
            self.state.traj_keys = None
        self.state.uniform_stride = IteratorState.is_uniform_stride(stride)
        if not IteratorState.is_uniform_stride(stride):
            if self._data_source.needs_sorted_ra_stride and not self.state.is_stride_sorted():
                raise ValueError("For this data source, currently only sorted arrays allowed for random access")
            # skip trajs which are not included in stride
            while self.state.itraj not in self.state.traj_keys and self.state.itraj < self._data_source.ntraj:
                self.state.itraj += 1

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

    @property
    def _n_chunks(self):
        """ rough estimate of how many chunks will be processed """
        if self.chunksize != 0:
            if not DataSourceIterator.is_uniform_stride(self.stride):
                chunks = ceil(len(self.stride[:, 0]) / float(self.chunksize))
            else:
                chunks = sum((ceil(l / float(self.chunksize))
                              for l in self.trajectory_lengths()))
        else:
            chunks = self.number_of_trajectories()
        return int(chunks)

    def number_of_trajectories(self):
        return self._data_source.number_of_trajectories()

    def trajectory_length(self):
        return self._data_source.trajectory_length(self._itraj, self.stride, self.skip)

    def trajectory_lengths(self):
        return self._data_source.trajectory_lengths(self.stride, self.skip)

    def n_frames_total(self):
        return self._data_source.n_frames_total(self.stride)

    @abstractmethod
    def close(self):
        pass

    def reset(self):
        """
        Method allowing to reset the iterator so that it can iterare from beginning on again.
        """
        self._t = 0
        self._itraj = 0

    @property
    def pos(self):
        """
        Gives the current position in the current trajectory.
        Returns
        -------
        int
            The current iterator's position in the current trajectory.
        """
        return self.state._pos

    @property
    def current_trajindex(self):
        """
        Gives the current iterator's trajectory index.
        Returns
        -------
        int
            The current iterator's trajectory index.
        """
        return self.state._current_itraj

    @property
    def skip(self):
        """
        Returns the skip value, i.e., the number of frames that are being omitted at the beginning of each
        trajectory.
        Returns
        -------
        int
            The skip value.
        """
        return self.state.skip

    @property
    def _t(self):
        """
        Reader-internal property that tracks the upcoming iterator position. Should not be used within iterator loop.
        Returns
        -------
        int
            The upcoming iterator position.
        """
        return self.state._t

    @_t.setter
    def _t(self, value):
        """
        Reader-internal property that tracks the upcoming iterator position.
        Parameters
        ----------
        value : int
            The upcoming iterator position.
        """
        self.state._t = value

    @property
    def _itraj(self):
        """
        Reader-internal property that tracks the upcoming trajectory index. Should not be used within iterator loop.
        Returns
        -------
        int
            The upcoming trajectory index.
        """
        return self.state.itraj

    @_itraj.setter
    def _itraj(self, value):
        """
        Reader-internal property that tracks the upcoming trajectory index. Should not be used within iterator loop.
        Parameters
        ----------
        value : int
            The upcoming trajectory index.
        """
        self.state.itraj = value

    @skip.setter
    def skip(self, value):
        """
        Sets the skip parameter. This can be used to skip the first n frames of the next trajectory in the iterator.
        Parameters
        ----------
        value : int
            The new skip parameter.
        """
        self.state.skip = value

    @property
    def chunksize(self):
        """
        The current chunksize of the iterator. Can be changed dynamically during iteration.
        Returns
        -------
        int
            The current chunksize of the iterator.
        """
        return self.state.chunk

    @chunksize.setter
    def chunksize(self, value):
        """
        Sets the current chunksize of the iterator. Can be changed dynamically during iteration.
        Parameters
        ----------
        value : int
            The chunksize of the iterator. Required to be non-negative.
        """
        if not value >= 0:
            raise ValueError("chunksize has to be non-negative")
        self.state.chunk = value

    @property
    def stride(self):
        """
        Gives the current stride parameter.
        Returns
        -------
        int
            The current stride parameter.
        """
        return self.state.stride

    @stride.setter
    def stride(self, value):
        """
        Sets the stride parameter.
        Parameters
        ----------
        value : int
            The new stride parameter.
        """
        self.state.stride = value

    @property
    def return_traj_index(self):
        """
        Property that gives information whether the trajectory index gets returned during the iteration.
        Returns
        -------
        bool
            True if the trajectory index should be returned, otherwise False.
        """
        return self.state.return_trajindex

    @property
    def traj_keys(self):
        """
        Random access property returning the trajectory indices that were handed in.
        Returns
        -------
        list
            Trajectories that are used in random access.
        """
        return self.state.traj_keys

    @property
    def uniform_stride(self):
        """
        Boolean property that tells if the stride argument was integral (i.e., uniform stride) or a random access
        dictionary.
        Returns
        -------
        bool
            True if the stride argument was integral, otherwise False.
        """
        return self.state.uniform_stride

    @return_traj_index.setter
    def return_traj_index(self, value):
        """
        Setter for return_traj_index, determining if the trajectory index gets returned in the iteration loop.
        Parameters
        ----------
        value : bool
            True if it should be returned, otherwise False
        """
        self.state.return_trajindex = value

    @staticmethod
    def is_uniform_stride(stride):
        return IteratorState.is_uniform_stride(stride)

    @property
    def last_chunk(self):
        """
        Property returning if the current chunk is the last chunk before the iterator terminates.
        Returns
        -------
        bool
            True if the iterator terminates after the current chunk, otherwise False
        """
        return self.current_trajindex == self.number_of_trajectories() - 1 and self.last_chunk_in_traj

    @property
    def last_chunk_in_traj(self):
        """
        Property returning if the current chunk is the last chunk before the iterator terminates or the next trajectory.
        Returns
        -------
        bool
            True if the next chunk either belongs to a new trajectory or the iterator terminates.
        """
        if self.chunksize > 0:
            return self._last_chunk_in_traj
        else:
            return True

    @abstractmethod
    def _next_chunk(self):
        pass

    def __next__(self):
        return self.next()

    def next(self):
        # first chunk at all, skip prepending trajectories that are not considered in random access
        if self._t == 0 and self._itraj == 0 and not self.uniform_stride:
            while (self._itraj not in self.traj_keys or self._t >= self.ra_trajectory_length(self._itraj)) \
                    and self._itraj < self.number_of_trajectories():
                self._itraj += 1
        # we have to obtain the current index before invoking next_chunk (which increments itraj)
        self.state._current_itraj = self._itraj
        self.state._pos = self.state._pos_adv
        try:
            X = self._next_chunk()
        except StopIteration:
            self._last_chunk_in_traj = True
            raise
        if self.state._current_itraj != self._itraj:
            self.state._pos_adv = 0
            self._last_chunk_in_traj = True
        else:
            self.state._pos_adv += len(X)
            length = self._data_source.trajectory_length(itraj=self.state._current_itraj,
                                                         stride=self.stride, skip=self.skip)
            self._last_chunk_in_traj = self.state._pos_adv >= length
        if self.return_traj_index:
            return self.state._current_itraj, X
        return X

    def __iter__(self):
        return self
