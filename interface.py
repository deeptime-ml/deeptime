'''
Created on 09.04.2015

@author: marscher
'''
from pyemma.coordinates.transform.transformer import Transformer
import numpy as np
import functools


class ReaderInterface(Transformer):
    """basic interface for readers
    """

    def __init__(self, chunksize=100):
        super(ReaderInterface, self).__init__(chunksize=chunksize)
        # TODO: think about if this should be none or self
        self.data_producer = self

        # internal counters
        self._t = 0
        self._itraj = 0

        # lengths and dims
        # NOTE: children have to make sure, they set these attributes in their ctor
        self._ntraj = -1
        self._ndim = -1
        self._lengths = []

        # storage for arrays (used in _add_array_to_storage)
        self._data = []

    def number_of_trajectories(self):
        """
        Returns the number of trajectories

        :return:
            number of trajectories
        """
        return self._ntraj

    def trajectory_length(self, itraj, stride=1):
        """
        Returns the length of trajectory

        :param itraj:
            trajectory index
        :param stride: 
            return value is the number of frames in trajectory when
            running through it with a step size of `stride`

        :return:
            length of trajectory
        """
        return (self._lengths[itraj] - 1) // int(stride) + 1

    def trajectory_lengths(self, stride=1):
        """
        Returns the length of each trajectory

        :param stride:
            return value is the number of frames in trajectories when
            running through them with a step size of `stride`

        :return:
            list containing length of each trajectory
        """
        return [(l - 1) // stride + 1 for l in self._lengths]

    def n_frames_total(self, stride=1):
        """
        Returns the total number of frames, over all trajectories

        :param stride:
            return value is the number of frames in trajectories when
            running through them with a step size of `stride`

        :return:
            the total number of frames, over all trajectories
        """
        if stride == 1:
            return np.sum(self._lengths)
        else:
            return sum(self.trajectory_lengths(stride))

    def dimension(self):
        """
        Returns the number of output dimensions

        :return:
        """
        return self._ndim

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
            shape_2d = (shape[0],
                        functools.reduce(lambda x, y: x * y, shape[1:]))
            array = np.reshape(array, shape_2d)

        self._data.append(array)
