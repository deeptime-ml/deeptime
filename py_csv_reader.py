'''
Created on 11.04.2015

@author: marscher
'''
from pyemma.coordinates.data.interface import ReaderInterface
import numpy as np
import csv


class _csv_chunked_numpy_iterator:

    """
    returns numpy arrays by combining multiple lines to chunks
    """

    def __init__(self, reader, chunksize=1, skiprows=None, header=None):
        self.reader = reader
        self.chunksize = chunksize

        if isinstance(skiprows, int):
            skiprows = np.arange(skiprows)
        self.skiprows = (np.empty(0) if skiprows is None
                         else np.unique(skiprows))

        self.line = 0

        # skip header in first row
        if header == 0:
            print "skipping header"
            self.reader.next()
            self.line = 1

    def get_chunk(self):
        return self.next()

    def __iter__(self):
        return self

    def _convert_to_np_chunk(self, list_of_strings):
        stack_of_strings = np.vstack(list_of_strings)
        result = stack_of_strings.astype(float)
        return result

    def next(self):
        lines = []

        for row in self.reader:
            if self.line in self.skiprows:
                # print "skip line", self.line
                self.line += 1
                continue

            self.line += 1
            lines.append(row)
            if self.line % self.chunksize == 0:
                result = self._convert_to_np_chunk(lines)
                del lines[:]
                return result

        # last chunk
        if len(lines) > 0:
            return self._convert_to_np_chunk(lines)

        raise StopIteration


class PyCSVReader(ReaderInterface):

    def __init__(self, filenames, chunksize=1000, **kwargs):
        """

        """
        super(PyCSVReader, self).__init__(chunksize=chunksize)
        self.data_producer = self

        if not isinstance(filenames, (tuple, list)):
            filenames = [filenames]
        self._filenames = filenames
        # list of boolean to indicate if file
        self._has_header = [False] * len(self._filenames)

        self._dialects = [None] * len(self._filenames)

        self._kwargs = {}

        # user wants to skip lines, so we need to remember this for lagged
        # access
        if kwargs.get('skip'):
            self._skip = kwargs.pop('skip')
        else:
            self._skip = 0

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
                    # count rows
                    self._lengths.append(sum(1 for _ in fh))
                    fh.seek(0)
                    # determine if file has header here:
                    sample = fh.read(1024)
                    self._dialects[ii] = csv.Sniffer().sniff(sample)
                    self._has_header[ii] = csv.Sniffer().has_header(sample)
                    # if we have a header subtract it from total length
                    if self._has_header[ii]:
                        self._lengths[-1] -= 1
                    fh.seek(0)
                    r = csv.reader(fh, dialect=self._dialects[ii])
                    if self._has_header[ii]:
                        next(r)
                    line = next(r)
                    arr = np.array(line).astype(float)
                    dim = arr.squeeze().shape[0]
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
        self._iter = None
        self._iter_lagged = None

    def _open_file(self, skip=0, stride=1, lagged=False):
        fn = self._filenames[self._itraj]
        self._logger.debug("opening file %s" % fn)

        # do not open same file
        if not lagged:
            reader = self._iter
        else:
            reader = self._iter_lagged

        if reader and reader.f == fn:
            return

        self._logger.debug("survived already open check")

        if self._has_header[self._itraj]:
            # set first line to be interpreted as header(labels)
            header = 0
        else:
            header = None

        skip = (self._skip + skip) * stride
        self._logger.debug("effective skip = %i; stride=%i" % (skip, stride))
        nt = self._lengths[self._itraj]
        # TODO: is this valid or even needed?
        if self._has_header[self._itraj]:
            self._logger.debug("incrementing nt, since we have a header")
            nt += 1

        # calculate an index set, which rows to skip (includes stride)
        skiprows = None

        if skip > 0:
            skiprows = np.zeros(nt)
            skiprows[:skip] = np.arange(skip)

        if stride > 1:
            all_frames = np.arange(nt)
            if skiprows is not None:
                wanted_frames = np.arange(skip-1, nt, stride)
            else:
                wanted_frames = np.arange(0, nt, stride)
            skiprows = np.setdiff1d(
                all_frames, wanted_frames, assume_unique=True)

            #np.testing.assert_equal( x[lag::stride], np.arange(nt)[np.arange(lag, nt, stride)] )
#             wanted_indices = np.arange(skip, nt, stride)
#             all_frames = np.arange(nt)
#             skiprows = np.setdiff1d(
#                 all_frames, wanted_indices, assume_unique=True)

#         if stride > 1:
#             all_frames = np.arange(nt)
#             stridden_rows_to_skip = np.arange(0, nt, stride)
#             skiprows = np.setdiff1d(
#                 all_frames, stridden_rows_to_skip, assume_unique=True)
#             self._logger.debug("stridden_rows to omit:\n%s" % skiprows)
#
#         if skip > 0:
#             self._logger.debug("skip: %i" % skip)
# lag_inds_to_skip = np.zeros(nt)
# lag_inds_to_skip[:skip] = np.arange(skip)
# #
# self._logger.debug("lag inds to skip:\n%s" % lag_inds_to_skip)
# if skiprows is not None:
# skiprows = np.union1d(skiprows, lag_inds_to_skip)
# else:
# skiprows = lag_inds_to_skip
# if skiprows is not None: # we have stride already.
# skiprows[:skip] = np.arange(skip)
# skiprows -= skip-1 # shift everything by skip to left
# lag_inds_to_skip = np.zeros(nt)
# lag_inds_to_skip[:skip] = np.arange(skip-1)
# skiprows=np.union1d(skiprows, lag_inds_to_skip)
#             else:
#                 skiprows = np.arange(skip-1)

        if __debug__:
            if isinstance(skiprows, np.ndarray):
                self._logger.debug("effective skiprows:\n%s\n%s" %
                                   (skiprows, str(skiprows.shape)))
        self._logger.debug("header: %s" % header)
        try:
            fh = open(fn)
            reader = _csv_chunked_numpy_iterator(
                csv.reader(fh, dialect=self._dialects[self._itraj]),
                chunksize=self.chunksize, skiprows=skiprows, header=header)
            reader.f = fn
        except EnvironmentError:
            self._logger.exception()
            raise

        if not lagged:
            self._iter = reader
        else:
            self._iter_lagged = reader

    def _next_chunk(self, lag=0, stride=1):
        if self._iter is None:
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

        X = self._iter.get_chunk()
        self._t += X.shape[0]

        if lag == 0:
            return X
        else:
            try:
                Y = self._iter_lagged.get_chunk()
            except StopIteration:
                Y = np.empty(0)
            return X, Y
