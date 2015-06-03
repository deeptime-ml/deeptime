# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
        # needs to be set, to ensure some getters of Transformer will work.
        self._data_producer = self

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

    @Transformer.data_producer.setter
    def data_producer(self, value):
        self._logger.warning("tried to set data_producer in reader, which makes"
                             " no sense.")
        import inspect
        res = inspect.getouterframes(inspect.currentframe())[1]
        self._logger.debug(str(res))

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
            numpy array containing length of each trajectory
        """
        return np.array([(l - 1) // stride + 1 for l in self._lengths], dtype=int)

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

    # handle abstract methods and special cases
    def map(self, X):
        raise NotImplementedError("a reader can not map data, it is a data source")

    def _map_array(self, X):
        raise NotImplementedError("a reader can not map data, it is a data source")

    def _param_add_data(self, *args, **kwargs):
        raise NotImplementedError("a reader is not meant to be parameterized by data")
