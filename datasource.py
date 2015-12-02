import functools
from abc import ABCMeta, abstractmethod

import six
import numpy as np

from pyemma.coordinates.data.iterable import Iterable
from pyemma.util.annotators import deprecated


class DataSource(Iterable):
    def __init__(self, chunksize=100):
        super(DataSource, self).__init__(chunksize)
        self._lengths = []
        # storage for arrays (used in _add_array_to_storage)
        self._data = []
        # following properties have to be set in subclass
        self._ndim = 0
        self._ntraj = 0
        self._lengths = []

    @deprecated("legacy code")
    def dimension(self):
        return self.ndim

    @property
    def ndim(self):
        return self._ndim

    @property
    def ntraj(self):
        return self._ntraj

    @property
    def data(self):
        return self._data

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
        if isinstance(stride, np.ndarray):
            selection = stride[stride[:, 0] == itraj][:, 0]
            return 0 if itraj not in selection else len(selection)
        else:
            return (self._lengths[itraj] - (self._skip if skip is None else skip) - 1) // int(stride) + 1

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


class DataSourceIterator(six.with_metaclass(ABCMeta)):
    def __init__(self, data_source, skip=0, chunk=0, stride=1, return_trajindex=False):
        self._data_source = data_source
        self._skip = skip
        self._chunk = chunk
        self.__init_stride(stride)
        self._return_trajindex = return_trajindex
        self._itraj = 0
        self._t = 0

    def __init_stride(self, stride):
        self._stride = stride
        if isinstance(stride, np.ndarray):
            keys = stride[:, 0]
            self._trajectory_keys, self._trajectory_lengths = np.unique(keys, return_counts=True)
        else:
            self._trajectory_keys = None
        self._uniform_stride = DataSourceIterator.is_uniform_stride(stride)
        if not self.uniform_stride and not self.is_stride_sorted():
            raise ValueError("Currently only sorted arrays allowed for random access")

    def ra_indices_for_traj(self, traj):
        """
        Gives the indices for a trajectory file index (without changing the order within the trajectory itself).
        :param traj: a trajectory file index
        :return: a Nx1 - np.array of the indices corresponding to the trajectory index
        """
        assert not self.uniform_stride, "requested random access indices, but is in uniform stride mode"
        return self._stride[self._stride[:, 0] == traj][:, 1] if traj in self.traj_keys else np.array([])

    def ra_trajectory_length(self, traj):
        assert not self.uniform_stride, "requested random access trajectory length, but is in uniform stride mode"
        return int(self._trajectory_lengths[np.where(self.traj_keys == traj)]) if traj in self.traj_keys else 0

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

    @property
    def current_trajindex(self):
        return self._itraj

    @property
    def skip(self):
        return self._skip

    @skip.setter
    def skip(self, value):
        self._skip = value

    @skip.deleter
    def skip(self):
        del self._skip

    @property
    def chunksize(self):
        return self._chunk

    @chunksize.setter
    def chunksize(self, value):
        if not value >= 0:
            raise ValueError("chunksize has to be positive")
        self._chunk = value

    @chunksize.deleter
    def chunksize(self):
        del self._chunk

    @property
    def stride(self):
        return self._stride

    @stride.setter
    def stride(self, value):
        self._stride = value

    @stride.deleter
    def stride(self):
        del self._stride

    @property
    def return_traj_index(self):
        return self._return_trajindex

    @property
    def traj_keys(self):
        return self._trajectory_keys

    @property
    def uniform_stride(self):
        return self._uniform_stride

    @return_traj_index.setter
    def return_traj_index(self, value):
        self._return_trajindex = value

    @return_traj_index.deleter
    def return_traj_index(self):
        del self._return_trajindex

    @staticmethod
    def is_uniform_stride(stride):
        return not isinstance(stride, np.ndarray)

    @abstractmethod
    def next_chunk(self):
        pass

    def __next__(self):
        return self.next()

    def next(self):
        return self.next_chunk()

# class PipelineStage(DataSource):
#     def __init__(self, data_source, transformer):
#         super(PipelineStage, self).__init__()
#         self._data_source = data_source
#         self._transformer = transformer
