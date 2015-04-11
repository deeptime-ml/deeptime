'''
Created on 07.04.2015

@author: marscher
'''
import numpy as np

from pandas.io.parsers import TextFileReader
from pyemma.coordinates.data.interface import ReaderInterface
import pandas
import csv


class NumPyFileReader(ReaderInterface):

    """reads NumPy files in chunks. Supports .npy and .npz files

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

        if not isinstance(filenames, (list, tuple)):
            filenames = [filenames]
        self._filenames = filenames

        for f in self._filenames:
            if not (f.endswith('.npy') or f.endswith('.npz')):
                raise ValueError('given file "%s" is not supported'
                                 ' by this reader' % f)

        self.mmap_mode = mmap_mode

        # current storage, holds mmapped arrays
        self._data = []

        # current file handle
        self._fh = None

        self.__set_dimensions_and_lenghts()

        self._parametrized = True

    def _reset(self, stride=1):
        self._t = 0
        self._itraj = 0
        if self._fh is not None:
            self._fh.close()

    def __load_file(self, filename):
        assert filename in self._filenames

        if self._fh is not None:
            # name already open?
            if self._fh.name == filename:
                return
            else:
                self._fh.close()

        self._logger.debug("opening file %s" % filename)
        self._fh = open(filename)

        if filename.endswith('.npy'):
            x = np.load(filename, mmap_mode=self.mmap_mode)
            self._add_array_to_storage(x)

        # in this case the file might contain several arrays
        elif filename.endswith('.npz'):
            # closes file handle
            npz_file = np.load(self._fh, mmap_mode=self.mmap_mode)
            for _, arr in npz_file.items():
                self._add_array_to_storage(arr)
        else:
            raise ValueError("given file '%s' is not a NumPy array. Make sure it has"
                             " either an .npy or .npz extension" % filename)

    def __set_dimensions_and_lenghts(self):
        for f in self._filenames:
            self.__load_file(f)

        self._lengths += [np.shape(x)[0] for x in self._data]

        # ensure all trajs have same dim
        ndims = [np.shape(x)[1] for x in self._data]
        if not np.unique(ndims).size == 1:
            raise ValueError("input data has different dimensions!"
                             "Dimensions are = %s" % ndims)

        self._ndim = ndims[0]

        self._ntraj = len(self._data)

    def _next_chunk(self, lag=0, stride=1):

        if (self._t >= self.trajectory_length(self._itraj, stride=stride) and
                self._itraj < len(self._filenames) - 1):
            # close file handles and open new ones
            self._t = 0
            self._itraj += 1
            self.__load_file(self._filenames[self._itraj])
            # we open self._mditer2 only if requested due lag parameter!
            self._curr_lag = 0

        traj_len = self.trajectory_length(self._itraj, stride)
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
            upper_bound = min(self._t + self._chunksize * stride, traj_len)
            slice_x = slice(self._t, upper_bound, stride)

            X = traj[slice_x]
            last_t = self._t
            self._t += X.shape[0]

            if self._t >= traj_len:
                self._itraj += 1
                self._t = 0

            if lag == 0:
                return X
            else:
                # its okay to return empty chunks
                upper_bound = min(
                    last_t + (lag + self._chunksize) * stride, traj_len)
                slice_y = slice(last_t + lag, upper_bound, stride)

                Y = traj[slice_y]
                return X, Y


class CSVReader(ReaderInterface):

    """reads tabulated data from given files in chunked mode

    Parameters
    ----------
    filenames : list of strings
        filenames (including paths) to read
    chunksize : int
        how many lines to process at once

    kwargs : named variables
        these will be passed into pandas FileTextReader
        see : http://pandas.pydata.org/pandas-docs/dev/generated/pandas.io.parsers.read_csv.html

    """

    def __init__(self, filenames, chunksize=1000, **kwargs):
        super(CSVReader, self).__init__(chunksize=chunksize)
        self.data_producer = self

        if not isinstance(filenames, (tuple, list)):
            filenames = [filenames]
        self._filenames = filenames
        # list of boolean to indicate if file
        self._has_header = [False] * len(self._filenames)

        self._kwargs = {}

        # default arguments for underlying csv reader
        if not kwargs.get('sep'):
            self._kwargs['sep'] = '\s+'

        if not kwargs.get('comment'):
            self._kwargs['comment'] = "#"

        # user wants to skip lines, so we need to remember this for lagged
        # access
        if kwargs.get('skip'):
            self._skip = kwargs.pop('skip')
        else:
            self._skip = 0

        self._reader = None
        self._reader_lagged = None

        # determine if lagged access pattern has changed
        self._current_lag = 0

        self.__set_dimensions_and_lenghts()
        self._parametrized = True

    def __set_dimensions_and_lenghts(self):
        # number of trajectories/data sets
        self._ntraj = len(self._filenames)
        if self._ntraj == 0:
            raise ValueError("empty file list")

        ndims = []

        for ii, f in enumerate(self._filenames):
            try:
                # determine file length
                with open(f) as fh:
                    self._lengths.append(sum(1 for _ in fh))
                    fh.seek(0)
                    # determine if file has header here:
                    self._has_header[ii] = csv.Sniffer().has_header(
                        fh.read(1024))
                    fh.seek(0)
                    r = TextFileReader(fh, chunksize=2, sep="\s+")
                    dim = r.get_chunk(1).values.squeeze().shape[0]
                    ndims.append(dim)

            # parent of IOError, OSError *and* WindowsError where available
            except EnvironmentError:
                self._logger.exception()
                self._logger.error(
                    "removing %s from list, since it caused an error" % f)
                self._filenames.remove(f)

        # check all files have same dimensions
        if not len(np.unique(ndims)) == 1:
            self._logger.error("got different dims: %s" % ndims)
            raise ValueError("input files have different dims")
        else:
            self._ndim = dim

    def _reset(self, stride=1):
        self._t = 0
        self._itraj = 0
        # to reopen files
        self._reader = None
        self._reader_lagged = None

    def _open_file(self, skip=0, stride=1, lagged=False):
        fn = self._filenames[self._itraj]

        # do not open same file
        if not lagged:
            reader = self._reader
        else:
            reader = self._reader_lagged

        if reader and reader.f == fn:
            return

        if self._has_header[self._itraj]:
            # set first line to be interpreted as header(labels)
            header = 0
        else:
            header = None

        skip = self._skip + skip
        nt = self.trajectory_length(self._itraj, 1)

        # calculate an index set, which rows to skip (includes stride)
        if stride > 1 or skip > 0:
            rows_to_read = np.arange(skip, nt, stride)
            all_inds = np.arange(0, nt)
            skiprows = np.setdiff1d(all_inds, rows_to_read, True)
        else:
            skiprows = 0

        if __debug__:
            self._logger.debug("skiprows:\n%s" % skiprows)

        reader = pandas.read_csv(fn, header=header, chunksize=self.chunksize,
                                 sep='\s+', skiprows=skiprows)

        if not lagged:
            self._reader = reader
        else:
            self._reader_lagged = reader

    def _next_chunk(self, lag=0, stride=1):
        if self._reader is None:
            self._open_file(stride=stride)

        if (self._t >= self.trajectory_length(self._itraj, stride=stride) and
                self._itraj < len(self._filenames) - 1):
            # close file handles and open new ones
            self._t = 0
            self._itraj += 1

            self._open_file(stride=stride)

        if lag != self._current_lag:
            self._current_lag = lag
            self._open_file(skip=lag, stride=stride, lagged=True)

        # omit rows according to stride
        X = self._reader.get_chunk().values
        self._t += X.shape[0]

        if lag == 0:
            return X
        else:
            try:
                Y = self._reader_lagged.get_chunk().values
            except StopIteration:
                Y = np.empty(0)
            return X, Y
