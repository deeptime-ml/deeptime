
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
        self._logger.info("hi from data in mem")

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

    def _reset(self, context=None):
        """Resets the data producer
        """
        self._itraj = 0
        self._t = 0

    def _next_chunk(self, ctx):
        # finished once with all trajectories? so _reset the pointer to allow
        # multi-pass
        if self._itraj >= self._ntraj:
            self._reset()

        traj_len = self._lengths[self._itraj]
        traj = self._data[self._itraj]

        # complete trajectory mode
        if self.chunksize == 0:
            if not ctx.uniform_stride:
                X = self._data[self._itraj][ctx.ra_indices_for_traj(self._itraj)]
                self._itraj += 1
                # skip trajs which are not included in stride
                while self._itraj not in ctx.traj_keys and self._itraj < self.number_of_trajectories():
                    self._itraj += 1
                if ctx.lag == 0:
                    return X
                else:
                    raise ValueError("Random access with lag not supported")
            else:
                X = traj[::ctx.stride]
                self._itraj += 1
                if ctx.lag == 0:
                    return X
                else:
                    Y = traj[ctx.lag::ctx.stride]
                    return X, Y
        # chunked mode
        else:
            if not ctx.uniform_stride:
                Y0 = self._data[self._itraj][
                    ctx.ra_indices_for_traj(self._itraj)[self._t:min(
                        self._t + self.chunksize, ctx.ra_trajectory_length(self._itraj)
                    )]
                ]
                if ctx.lag != 0:
                    raise ValueError("Random access with lag not supported")

                self._t += self.chunksize
                if self._t >= ctx.ra_trajectory_length(self._itraj):
                    self._itraj += 1
                    self._t = 0

                # skip trajs which are not included in stride
                while (self._itraj not in ctx.traj_keys or self._t >= ctx.ra_trajectory_length(self._itraj)) \
                        and self._itraj < self.number_of_trajectories():
                    self._itraj += 1
                    self._t = 0
                return Y0
            else:
                upper_bound = min(self._t + self.chunksize * ctx.stride, traj_len)
                slice_x = slice(self._t, upper_bound, ctx.stride)

                X = traj[slice_x]

                if ctx.lag != 0:
                    upper_bound_Y = min(
                         self._t + ctx.lag + self.chunksize * ctx.stride, traj_len)
                    slice_y = slice(self._t + ctx.lag, upper_bound_Y, ctx.stride)
                    Y = traj[slice_y]

                self._t = upper_bound

                if upper_bound >= traj_len:
                    self._itraj += 1
                    self._t = 0

                if ctx.lag == 0:
                    return X
                else:
                    return X, Y
