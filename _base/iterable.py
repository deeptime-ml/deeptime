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

from __future__ import print_function
from abc import ABCMeta, abstractmethod
import numpy as np
import six

from pyemma._base.loggable import Loggable
from pyemma.coordinates.data._base._in_memory_mixin import InMemoryMixin
from pyemma.util.contexts import attribute
from pyemma.util.types import is_int


class Iterable(six.with_metaclass(ABCMeta, InMemoryMixin, Loggable)):

    def __init__(self, chunksize=None):
        super(Iterable, self).__init__()
        self._default_chunksize = chunksize
        # should be set in subclass
        self._ndim = 0

    def dimension(self):
        return self._ndim

    @property
    def ndim(self):
        return self.dimension()

    @property
    def default_chunksize(self):
        """ How much data will be processed at once, in case no chunksize has been provided."""
        if self._default_chunksize is None:
            from pyemma import config
            from pyemma.util.units import string_to_bytes
            max_bytes = string_to_bytes(config.default_chunksize)
            itemsize = np.dtype(self.output_type()).itemsize
            # TODO: consider rounding this to some cache size of CPU? e.g py-cpuinfo can obtain it.
            max_elements = max_bytes // itemsize // self.ndim
            self._default_chunksize = max_elements
            assert self._default_chunksize > 0
        return self._default_chunksize

    @property
    def chunksize(self):
        return self.default_chunksize

    @chunksize.setter
    def chunksize(self, value):
        if self.default_chunksize < 0:
            raise ValueError("Chunksize of %s was provided, but has to be >= 0" % self.default_chunksize)
        self._default_chunksize = value

    def iterator(self, stride=1, lag=0, chunk=None, return_trajindex=True, cols=None, skip=0):
        """ creates an iterator to stream over the (transformed) data.

        If your data is too large to fit into memory and you want to incrementally compute
        some quantities on it, you can create an iterator on a reader or transformer (eg. TICA)
        to avoid memory overflows.

        Parameters
        ----------

        stride : int, default=1
            Take only every stride'th frame.
        lag: int, default=0
            how many frame to omit for each file.
        chunk: int, default=None
            How many frames to process at once. If not given obtain the chunk size
            from the source.
        return_trajindex: boolean, default=True
            a chunk of data if return_trajindex is False, otherwise a tuple of (trajindex, data).
        cols: array like, default=None
            return only the given columns.
        skip: int, default=0
            skip 'n' first frames of each trajectory.

        Returns
        -------
        iter : instance of DataSourceIterator
            a implementation of a DataSourceIterator to stream over the data

        Examples
        --------

        >>> from pyemma.coordinates import source; import numpy as np
        >>> data = [np.arange(3), np.arange(4, 7)]
        >>> reader = source(data)
        >>> iterator = reader.iterator(chunk=1)
        >>> for array_index, chunk in iterator:
        ...     print(array_index, chunk)
        0 [[0]]
        0 [[1]]
        0 [[2]]
        1 [[4]]
        1 [[5]]
        1 [[6]]
        """
        if self.in_memory:
            from pyemma.coordinates.data.data_in_memory import DataInMemory
            return DataInMemory(self._Y).iterator(
                lag=lag, chunk=chunk, stride=stride, return_trajindex=return_trajindex, skip=skip
            )
        chunk = chunk if chunk is not None else self.chunksize
        if 0 < lag <= chunk:
            it = self._create_iterator(skip=skip, chunk=chunk, stride=1,
                                       return_trajindex=return_trajindex, cols=cols)
            it.return_traj_index = True
            return _LaggedIterator(it, lag, return_trajindex, stride)
        elif lag > 0:
            it = self._create_iterator(skip=skip, chunk=chunk, stride=stride,
                                       return_trajindex=return_trajindex, cols=cols)
            it.return_traj_index = True
            it_lagged = self._create_iterator(skip=skip + lag, chunk=chunk, stride=stride,
                                              return_trajindex=True, cols=cols)
            return _LegacyLaggedIterator(it, it_lagged, return_trajindex)
        return self._create_iterator(skip=skip, chunk=chunk, stride=stride,
                                     return_trajindex=return_trajindex, cols=cols)

    @abstractmethod
    def _create_iterator(self, skip=0, chunk=0, stride=1, return_trajindex=True, cols=None):
        """
        Should be implemented by non-abstract subclasses. Creates an instance-independent iterator.
        :param skip: How many frames to skip before streaming.
        :param chunk: The chunksize.
        :param stride: Take only every stride'th frame.
        :param return_trajindex: take the trajindex into account
        :return: a chunk of data if return_trajindex is False, otherwise a tuple of (trajindex, data).
        """
        raise NotImplementedError()

    def output_type(self):
        r""" By default transformers return single precision floats. """
        return np.float32

    def __iter__(self):
        return self.iterator()


class _LaggedIterator(object):
    """ _LaggedIterator

    avoids double IO, by switching the chunksize on the given Iterable instance and
    remember an overlap.

    Parameters
    ----------
    it: instance of Iterable (stride=1)
    lag : int
        lag time
    actual_stride: int
        stride
    return_trajindex: bool
        whether to return the current trajectory index during iteration (itraj).
    """
    def __init__(self, it, lag, return_trajindex, actual_stride):
        self._it = it
        self._lag = lag
        assert is_int(lag)
        self._return_trajindex = return_trajindex
        self._overlap = None
        self._actual_stride = actual_stride
        self._sufficently_long_trajectories = [i for i, x in
                                               enumerate(self._it._data_source.trajectory_lengths(1, 0))
                                               if x > lag]

    @property
    def n_chunks(self):
        cs = self._it.chunksize
        n1 = self._it._data_source.n_chunks(cs, stride=self._actual_stride, skip=self._lag)
        n2 = self._it._data_source.n_chunks(cs, stride=self._actual_stride, skip=0)
        return min(n1, n2)

    def __len__(self):
        n1 = self._it._data_source.trajectory_lengths(self._actual_stride, 0).min()
        n2 = self._it._data_source.trajectory_lengths(self._actual_stride, self._lag).min()
        return min(n1, n2)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        itraj_changed = False
        while (self._it._itraj not in self._sufficently_long_trajectories
               and self._it.number_of_trajectories() > self._it.current_trajindex):
            self._it._itraj += 1
            self._overlap = None
            itraj_changed = True
        # ensure file next handle gets opened.
        if itraj_changed:
            self._it._select_file(self._it._itraj)

        if self._overlap is None:
            with attribute(self._it, 'chunksize', self._lag):
                _, self._overlap = self._it.next()
                self._overlap = self._overlap[::self._actual_stride]

        with attribute(self._it, 'chunksize', self._it.chunksize * self._actual_stride):

            itraj, data_lagged = self._it.next()
            frag = data_lagged[:min(self._it.chunksize - self._lag, len(data_lagged)), :]
            data = np.concatenate((self._overlap, frag[(self._actual_stride - self._lag)
                                                       % self._actual_stride::self._actual_stride]), axis=0)

            offset = min(self._it.chunksize - self._lag, len(data_lagged))
            self._overlap = data_lagged[offset::self._actual_stride, :]

            data_lagged = data_lagged[::self._actual_stride]

        if self._it._last_chunk_in_traj:
            self._overlap = None

        if data.shape[0] > data_lagged.shape[0]:
            # data chunk is bigger, truncate it to match data_lagged's shape
            data = data[:data_lagged.shape[0]]
        elif data.shape[0] < data_lagged.shape[0]:
            raise RuntimeError("chunk was smaller than time-lagged chunk (%s < %s), that should not happen!"
                               % (data.shape[0], data_lagged.shape[0]))

        if self._return_trajindex:
            return itraj, data, data_lagged
        return data, data_lagged

    def __enter__(self):
        self._it.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._it.__exit__(exc_type, exc_val, exc_tb)


class _LegacyLaggedIterator(object):
    """ _LegacyLaggedIterator uses two iterators to build time-lagged chunks.

    Parameters
    ----------
    it: DataSource, skip=0
    it_lagged: DataSource, skip=lag
    return_trajindex: bool
        whether to return the current trajectory index during iteration (itraj).
    """
    def __init__(self, it, it_lagged, return_trajindex):
        self._it = it
        self._it_lagged = it_lagged
        self._return_trajindex = return_trajindex

    @property
    def n_chunks(self):
        n1 = self._it.n_chunks
        n2 = self._it_lagged.n_chunks
        return min(n1, n2)

    def __len__(self):
        return min(self._it.trajectory_lengths().min(), self._it_lagged.trajectory_lengths().min())

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        itraj, data = self._it.next()
        itraj_lag, data_lagged = self._it_lagged.next()

        while itraj < itraj_lag:
            itraj, data = self._it.next()
        assert itraj == itraj_lag

        if data.shape[0] > data_lagged.shape[0]:
            # data chunk is bigger, truncate it to match data_lagged's shape
            data = data[:data_lagged.shape[0]]
        elif data.shape[0] < data_lagged.shape[0]:
            raise RuntimeError("chunk was smaller than time-lagged chunk (%s < %s), that should not happen!"
                               % (data.shape[0], data_lagged.shape[0]))
        if self._return_trajindex:
            return itraj, data, data_lagged
        return data, data_lagged

    def __enter__(self):
        self._it.__enter__()
        self._it_lagged.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._it.__exit__(exc_type, exc_val, exc_tb)
        self._it_lagged.__exit__(exc_type, exc_val, exc_tb)
