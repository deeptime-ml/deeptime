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
Created on 07.04.2015

@author: marscher
'''

from __future__ import absolute_import

import functools

import numpy as np

from pyemma.coordinates.data._base.datasource import DataSourceIterator, DataSource
from pyemma.coordinates.data.data_in_memory import DataInMemoryIterator
from pyemma.coordinates.data.util.traj_info_cache import TrajInfo
from pyemma.util.annotators import fix_docs


@fix_docs
class NumPyFileReader(DataSource):

    """reads NumPy files in chunks. Supports .npy files

    Parameters
    ----------
    filenames : str or list of strings

    chunksize : int
        how many rows are read at once

    mmap_mode : str (optional), default='r'
        binary NumPy arrays are being memory mapped using this flag.
    """

    def __init__(self, filenames, chunksize=1000, mmap_mode='r'):
        super(NumPyFileReader, self).__init__(chunksize=chunksize)
        self._is_reader = True

        if not isinstance(filenames, (list, tuple)):
            filenames = [filenames]

        for f in filenames:
            if not f.endswith('.npy'):
                raise ValueError('given file "%s" is not supported by this'
                                 ' reader, since it does not end with .npy' % f)

        self.mmap_mode = mmap_mode
        self.filenames = filenames

    def _create_iterator(self, skip=0, chunk=0, stride=1, return_trajindex=False, cols=None):
        return NPYIterator(self, skip=skip, chunk=chunk, stride=stride, 
                           return_trajindex=return_trajindex, cols=cols)

    def describe(self):
        return "[NumpyFileReader arrays with shape %s]" % [np.shape(x)
                                                           for x in self._data]

    def _reshape(self, array):
        """
        checks shapes, eg convert them (2d), raise if not possible
        after checks passed, set self._array and return it.
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
        return array

    def _load_file(self, itraj):
        filename = self._filenames[itraj]
        #self._logger.debug("opening file %s" % filename)

        if filename.endswith('.npy'):
            x = np.load(filename, mmap_mode=self.mmap_mode)
            arr = self._reshape(x)
        else:
            raise ValueError("given file '%s' is not a NumPy array. Make sure"
                             " it has a .npy extension" % filename)
        return arr

    def _get_traj_info(self, filename):
        idx = self.filenames.index(filename)
        array = self._load_file(idx)
        length, ndim = np.shape(array)

        return TrajInfo(ndim, length)


class NPYIterator(DataInMemoryIterator):

    def __init__(self, data_source, skip=0, chunk=0, stride=1, return_trajindex=False, cols=False):
        super(NPYIterator, self).__init__(data_source=data_source, skip=skip,
                                          chunk=chunk, stride=stride,
                                          return_trajindex=return_trajindex,
                                          cols=cols)

    def close(self):
        if not hasattr(self, 'data') or self.data is None:
            return
        # delete the memmap to close it.
        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.memmap.html
        del self.data
        self.data = None

    def _select_file(self, itraj):
        if self._selected_itraj != itraj:
            self._first_file_opened = True
            self.close()
            self._t = 0
            self._itraj = itraj
            self._selected_itraj = self._itraj
            if itraj < self.number_of_trajectories():
                self.data = self._data_source._load_file(itraj)

    def _next_chunk(self):
        return self._next_chunk_impl(self.data)
