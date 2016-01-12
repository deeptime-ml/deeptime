
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

'''
Created on 09.04.2015

@author: marscher
'''

from __future__ import absolute_import
from pyemma.coordinates.transform.transformer import Transformer
import numpy as np
import functools
from six.moves import range


class ReaderInterface(Transformer):
    """basic interface for readers
    """

    def __init__(self, chunksize=100):
        super(ReaderInterface, self).__init__()
        # needs to be set, to ensure some getters of Transformer will work.
        self._data_producer = self
        self.chunksize = chunksize

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

        # number of initially skipped frames, regardless of lag, no lag or chunk size
        self.__skip = 0

    @Transformer.data_producer.setter
    def data_producer(self, value):
        self._logger.warning("tried to set data_producer in reader, which makes"
                             " no sense.")
        import inspect
        res = inspect.getouterframes(inspect.currentframe())[1]
        self._logger.debug(str(res))

    @property
    def _skip(self):
        # TODO implement and test this for all readers
        return self.__skip

    @_skip.setter
    def _skip(self, value):
        # TODO implement and test this for all readers
        self.__skip = value

    @property
    def chunksize(self):
        """chunksize defines how much data is being processed at once."""
        return self._cs

    @chunksize.setter
    def chunksize(self, size):
        if not size >= 0:
            raise ValueError("chunksize has to be positive")

        self._cs = int(size)

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

        :return:
            length of trajectory
        """
        if isinstance(stride, np.ndarray):
            selection = stride[stride[:, 0] == itraj][:, 0]
            return 0 if itraj not in selection else len(selection)
        else:
            return (self._lengths[itraj] - (self._skip if skip is None else skip) - 1) // int(stride) + 1

    def trajectory_lengths(self, stride=1):
        """
        Returns the length of each trajectory

        :param stride:
            return value is the number of frames in trajectories when
            running through them with a step size of `stride`

        :return:
            numpy array containing length of each trajectory
        """
        if isinstance(stride, np.ndarray):
            return np.array(
                [self.trajectory_length(itraj, stride) for itraj in range(0, self.number_of_trajectories())],
                dtype=int)
        else:
            return np.array([(l - self._skip - 1) // stride + 1 for l in self._lengths], dtype=int)

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

    # handle abstract methods and special cases
    def transform(self, X):
        raise NotImplementedError("a reader can not map data, it is a data source")

    def _transform_array(self, X):
        raise NotImplementedError("a reader can not map data, it is a data source")

    def _param_add_data(self, *args, **kwargs):
        raise NotImplementedError("a reader is not meant to be parameterized by data")
