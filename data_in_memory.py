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
import numpy as np

from pyemma.coordinates.data.interface import ReaderInterface

__author__ = 'noe, marscher'


class DataInMemory(ReaderInterface):

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

    def __init__(self, data, chunksize=5000):
        super(DataInMemory, self).__init__(chunksize=chunksize)

        # storage
        self._data = []

        if not isinstance(data, (list, tuple)):
            data = [data]

        # everything is an array
        if all(isinstance(d, np.ndarray) for d in data):
            for d in data:
                self._add_array_to_storage(d)
        else:
            raise ValueError("Please supply numpy.ndarray, or list/tuple of ndarray."
                             " Your input was %s" % str(data))

        self.__set_dimensions_and_lenghts()
        self._parametrized = True

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
        return "[DataInMemory array shapes: %s]" % [np.shape(x) for x in self._data]

    def __set_dimensions_and_lenghts(self):
        # number of trajectories/data sets
        self._ntraj = len(self._data)
        if self._ntraj == 0:
            raise ValueError("no valid data")

        # this works since everything is flattened to 2d
        self._lengths = [np.shape(d)[0] for d in self._data]

        # ensure all trajs have same dim
        ndims = [np.shape(x)[1] for x in self._data]
        if not np.unique(ndims).size == 1:
            raise ValueError("input data has different dimensions!"
                             "Dimensions are = %s" % ndims)

        self._ndim = ndims[0]

    def _reset(self, stride=1):
        """Resets the data producer
        """
        self._itraj = 0
        self._t = 0

    def _next_chunk(self, lag=0, stride=1):
        # finished once with all trajectories? so _reset the pointer to allow
        # multi-pass
        if self._itraj >= self._ntraj:
            self._reset()

        traj_len = self._lengths[self._itraj]
        traj = self._data[self._itraj]

        # complete trajectory mode
        if self._chunksize == 0:
            X = traj[::stride]
            self._itraj += 1

            if lag == 0:
                return X
            else:
                Y = traj[lag * stride:traj_len:stride]
                return (X, Y)
        # chunked mode
        else:
            upper_bound = min(
                self._t + self._chunksize * stride, traj_len)
            slice_x = slice(self._t, upper_bound, stride)

            X = traj[slice_x]

            if lag!=0:
                 upper_bound_Y = min(
                     self._t + (lag + self._chunksize) * stride, traj_len)
                 slice_y = slice(self._t + lag*stride, upper_bound_Y, stride)
                 Y = traj[slice_y]

            self._t = upper_bound

            if upper_bound >= traj_len:
                 self._itraj += 1
                 self._t = 0
                 
            if lag==0:
                return X
            else: 
                return (X, Y)
