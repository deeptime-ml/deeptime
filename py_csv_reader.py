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

        # file handle and name
        self.fh = None
        self.f = None

        # skip header in first row
        if header == 0:
            self.reader.next()
            self.line = 1

    def get_chunk(self):
        return self.next()

    def __iter__(self):
        return self

    def close(self):
        self.fh.close()

    def _convert_to_np_chunk(self, list_of_strings):
        stack_of_strings = np.vstack(list_of_strings)
        result = stack_of_strings.astype(float)
        return result

    def next(self):
        if not self.fh:
            raise StopIteration

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

        self.fh.close()
        raise StopIteration


class PyCSVReader(ReaderInterface):

    def __init__(self, filenames, chunksize=1000, **kwargs):
        """

        """
        super(PyCSVReader, self).__init__(chunksize=chunksize)

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
        self._lagged_iter_finished = False

        self.__set_dimensions_and_lenghts()
        self._parametrized = True

    def describe(self):
        return "[CSVReader files=%s]" % self._filenames

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
        self._lagged_iter_finished = False

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

        if self._has_header[self._itraj]:
            # set first line to be interpreted as header(labels)
            header = 0
        else:
            header = None

        skip = (self._skip + skip)
        nt = self._lengths[self._itraj]

        if self._has_header[self._itraj]:
            nt += 1
            skip += 1

        # calculate an index set, which rows to skip (includes stride)
        skiprows = None

        if skip > 0:
            skiprows = np.zeros(nt)
            skiprows[:skip] = np.arange(skip)

        if stride > 1:
            all_frames = np.arange(nt)
            if skiprows is not None:
                wanted_frames = np.arange(skip, nt, stride)
            else:
                wanted_frames = np.arange(0, nt, stride)
            skiprows = np.setdiff1d(
                all_frames, wanted_frames, assume_unique=True)

        try:
            fh = open(fn)
            reader = _csv_chunked_numpy_iterator(
                csv.reader(fh, dialect=self._dialects[self._itraj]),
                chunksize=self.chunksize, skiprows=skiprows, header=header)
            reader.f = fn
            reader.fh = fh
        except EnvironmentError:
            self._logger.exception()
            raise

        if not lagged:
            self._iter = reader
        else:
            self._iter_lagged = reader

    def _close(self):
        # invalidate iterator
        if self._iter:
            self._iter.close()

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
            # Note: this ugly hack is needed, since the caller of this method
            # may try to request lagged chunks repeatedly.
            try:
                if self._lagged_iter_finished:
                    raise StopIteration
                Y = self._iter_lagged.get_chunk()
            except StopIteration:
                self._lagged_iter_finished = True
                Y = np.empty(0)
            return X, Y

    def parametrize(self, stride=1):
        if self.in_memory:
            self._map_to_memory(stride)
