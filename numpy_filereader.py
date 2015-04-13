'''
Created on 07.04.2015

@author: marscher
'''
import numpy as np

from pyemma.coordinates.data.interface import ReaderInterface


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

        self.next_ignore = 0

        self.__set_dimensions_and_lenghts()

        self._parametrized = True

    def _reset(self, stride=1):
        self._t = 0
        self._itraj = 0
        if self._fh is not None:
            self._fh.close()

        self.next_ignore = 0

    def __load_file(self, filename):
        assert filename in self._filenames

        if self._fh is not None:
            # name already open?
            if self._fh.name == filename:
                return
            else:
                self._fh.close()

        self._logger.debug("opening file %s" % filename)
        self._fh = open(filename, 'rb')

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
                self._t + (self._chunksize + 1) * stride, traj_len)
            slice_x = slice(self._t, upper_bound, stride)
            X = traj[slice_x]

            last_t = self._t
            self._t = upper_bound

            if self._t >= traj_len:
                self._itraj += 1
                self._t = 0

            if lag == 0:
                return X
            else:
                # its okay to return empty chunks
                upper_bound = min(
                    last_t + (lag + self._chunksize + 1) * stride, traj_len)
                slice_y = slice(last_t + lag, upper_bound, stride)

                Y = traj[slice_y]
                return X, Y
