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

from abc import ABCMeta, abstractmethod
import six
import numpy as np

from pyemma._base.logging import Loggable
from pyemma._base.progress import ProgressReporter


class Iterable(six.with_metaclass(ABCMeta, ProgressReporter, Loggable)):

    def __init__(self, chunksize=1000):
        super(Iterable, self).__init__()
        self._default_chunksize = chunksize
        if self.default_chunksize < 0:
            raise ValueError("Chunksize of %s was provided, but has to be >= 0" % self.default_chunksize)
        self._in_memory = False
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
        return self._default_chunksize

    @property
    def chunksize(self):
        return self._default_chunksize

    @chunksize.setter
    def chunksize(self, value):
        self._default_chunksize = value

    @property
    def in_memory(self):
        r"""are results stored in memory?"""
        return self._in_memory

    @in_memory.setter
    def in_memory(self, op_in_mem):
        r"""
        If set to True, the output will be stored in memory.
        """
        old_state = self._in_memory
        if not old_state and op_in_mem:
            self._in_memory = op_in_mem
            self._Y = []
            self._map_to_memory()
        elif not op_in_mem and old_state:
            self._clear_in_memory()
            self._in_memory = op_in_mem

    def _clear_in_memory(self):
        if self._logger_is_active(self._loglevel_DEBUG):
            self._logger.debug("clear memory")
        assert self.in_memory, "tried to delete in memory results which are not set"
        self._Y = None
        self._Y_source = None

    def _map_to_memory(self, stride=1):
        r"""Maps results to memory. Will be stored in attribute :attr:`_Y`."""
        if self._logger_is_active(self._loglevel_DEBUG):
            self._logger.debug("mapping to mem")
        assert self._in_memory
        self._mapping_to_mem_active = True
        self._Y = self.get_output(stride=stride)
        from pyemma.coordinates.data import DataInMemory
        self._Y_source = DataInMemory(self._Y)
        self._mapping_to_mem_active = False

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
        chunk = chunk if chunk is not None else self.default_chunksize
        it = self._create_iterator(skip=skip, chunk=chunk, stride=stride,
                                   return_trajindex=return_trajindex, cols=cols)
        if lag > 0:
            it.return_traj_index = True
            it_lagged = self._create_iterator(skip=skip+lag, chunk=chunk, stride=stride,
                                              return_trajindex=True, cols=cols)
            return LaggedIterator(it, it_lagged, return_trajindex)
        return it

    def get_output(self, dimensions=slice(0, None), stride=1, skip=0, chunk=None):
        """Maps all input data of this transformer and returns it as an array or list of arrays

        Parameters
        ----------
        dimensions : list-like of indexes or slice, default=all
           indices of dimensions you like to keep.
        stride : int, default=1
           only take every n'th frame.
        skip : int, default=0
            initially skip n frames of each file.
        chunk: int, default=None
            How many frames to process at once. If not given obtain the chunk size
            from the source.

        Returns
        -------
        output : list of ndarray(T_i, d)
           the mapped data, where T is the number of time steps of the input data, or if stride > 1,
           floor(T_in / stride). d is the output dimension of this transformer.
           If the input consists of a list of trajectories, Y will also be a corresponding list of trajectories

        """
        if isinstance(dimensions, int):
            ndim = 1
            dimensions = slice(dimensions, dimensions + 1)
        elif isinstance(dimensions, (list, np.ndarray, tuple, slice)):
            if hasattr(dimensions, 'ndim') and dimensions.ndim > 1:
                raise ValueError('dimension indices can\'t have more than one dimension')
            ndim = len(np.zeros(self.ndim)[dimensions])
        else:
            raise ValueError('unsupported type (%s) of "dimensions"' % type(dimensions))

        assert ndim > 0, "ndim was zero in %s" % self.__class__.__name__

        if chunk is None:
            chunk = self.chunksize

        # create iterator
        if self.in_memory and not self._mapping_to_mem_active:
            from pyemma.coordinates.data.data_in_memory import DataInMemory
            assert self._Y is not None
            it = DataInMemory(self._Y)._create_iterator(skip=skip, chunk=chunk,
                                                        stride=stride, return_trajindex=True)
        else:
            it = self._create_iterator(skip=skip, chunk=chunk, stride=stride, return_trajindex=True)

        with it:
            # allocate memory
            try:
                # TODO: avoid having a copy here, if Y is already filled
                trajs = [np.empty((l, ndim), dtype=self.output_type())
                         for l in it.trajectory_lengths()]
            except MemoryError:
                self._logger.exception("Could not allocate enough memory to map all data."
                                       " Consider using a larger stride.")
                return

            if self._logger_is_active(self._loglevel_DEBUG):
                self._logger.debug("get_output(): dimensions=%s" % str(dimensions))
                self._logger.debug("get_output(): created output trajs with shapes: %s"
                                   % [x.shape for x in trajs])
            # fetch data
            self.logger.debug("nchunks :%s, chunksize=%s" % (it._n_chunks, it.chunksize))
            self._progress_register(it._n_chunks,
                                    description='getting output of %s' % self.__class__.__name__,
                                    stage=1)
            for itraj, chunk in it:
                L = len(chunk)
                if L > 0:
                    trajs[itraj][it.pos:it.pos + L, :] = chunk[:, dimensions]

                # update progress
                self._progress_update(1, stage=1)

        return trajs

    def write_to_csv(self, filename=None, extension='.dat', overwrite=False,
                     stride=1, chunksize=100, **kw):
        """ write all data to csv with numpy.savetxt

        Parameters
        ----------
        filename : str, optional
            filename string, which may contain placeholders {itraj} and {stride}:

            * itraj will be replaced by trajetory index
            * stride is stride argument of this method

            If filename is not given, it is being tried to obtain the filenames
            from the data source of this iterator.
        extension : str, optional, default='.dat'
            filename extension of created files
        overwrite : bool, optional, default=False
            shall existing files be overwritten? If a file exists, this method will raise.
        stride : int
            omit every n'th frame
        chunksize: int
            how many frames to process at once
        kw : dict
            named arguments passed into numpy.savetxt (header, seperator etc.)

        Example
        -------
        Assume you want to save features calculated by some FeatureReader to ASCII:
        
        >>> import numpy as np, pyemma
        >>> from pyemma.util.files import TemporaryDirectory
        >>> import os
        >>> data = [np.random.random((10,3))] * 3
        >>> reader = pyemma.coordinates.source(data)
        >>> filename = "distances_{itraj}.dat"
        >>> with TemporaryDirectory() as td:
        ...    os.chdir(td)
        ...    reader.write_to_csv(filename, header='', delimiter=';')
        ...    print(os.listdir('.'))
        ['distances_2.dat', 'distances_1.dat', 'distances_0.dat']
        """
        import os
        if not filename:
            assert hasattr(self, 'filenames')
            #    raise RuntimeError("could not determine filenames")
            filenames = []
            for f in self.filenames:
                base, _ = os.path.splitext(f)
                filenames.append(base+extension)
        elif isinstance(filename, six.string_types):
            filename = filename.replace('{stride}', str(stride))
            filenames = [filename.replace('{itraj}', str(itraj)) for itraj
                         in range(self.number_of_trajectories())]
        else:
            raise TypeError("filename should be str or None")
        self.logger.debug("write_to_csv, filenames=%s" % filenames)
        # check files before starting to write
        import errno
        for f in filenames:
            try:
                st = os.stat(f)
                raise OSError(errno.EEXIST)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    if overwrite:
                        continue
                elif e.errno == errno.ENOENT:
                    continue
                raise
            else:
                continue
        f = None
        with self.iterator(stride, chunk=chunksize, return_trajindex=False) as it:
            self._progress_register(it._n_chunks, "saving to csv")
            oldtraj = -1
            for X in it:
                if oldtraj != it.current_trajindex:
                    if f is not None:
                        f.close()
                    fn = filenames[it.current_trajindex]
                    self.logger.debug("opening file %s for writing csv." % fn)
                    f = open(fn, 'wb')
                    oldtraj = it.current_trajindex
                np.savetxt(f, X, **kw)
                f.flush()
                self._progress_update(1, 0)
        if f is not None:
            f.close()
        self._progress_force_finish(0)

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
        pass

    def output_type(self):
        r""" By default transformers return single precision floats. """
        return np.float32

    def __iter__(self):
        return self.iterator()


class LaggedIterator(object):
    def __init__(self, it, it_lagged, return_trajindex):
        self._it = it
        self._it_lagged = it_lagged
        self._return_trajindex = return_trajindex

    @property
    def _n_chunks(self):
        n1 = self._it._n_chunks
        n2 = self._it_lagged._n_chunks
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
