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
"""
Created on 11.04.2015

@author: marscher
"""

from __future__ import absolute_import

from math import ceil
import csv
import os

from pyemma.coordinates.data._base.datasource import DataSourceIterator, DataSource
from pyemma.coordinates.data.util.traj_info_cache import TrajInfo
from six.moves import range
import numpy as np


class PyCSVIterator(DataSourceIterator):
    def __init__(self, data_source, skip=0, chunk=0, stride=1, return_trajindex=False):
        super(PyCSVIterator, self).__init__(data_source, skip=skip, chunk=chunk,
                                            stride=stride,
                                            return_trajindex=return_trajindex)
        self._open_file()
        if isinstance(self._skip_rows, int):
            self._skip_rows = np.arange(self._skip_rows)
        self._skip_rows = (np.empty(0) if self._skip_rows is None
                           else np.unique(self._skip_rows))
        self.line = 0
        self._reader = csv.reader(self._file_handle,
                                  dialect=self._data_source._get_dialect(self._itraj))

    def close(self):
        if self._file_handle is not None:
            self._file_handle.close()

    def _next_chunk(self):
        if not self._file_handle or self._itraj >= self.number_of_trajectories():
            self.close()
            raise StopIteration()

        traj_len = self.trajectory_length()
        lines = []
        for row in self._reader:
            if self.line in self._skip_rows:
                self.line += 1
                continue

            self.line += 1
            if not self.uniform_stride:
                for _ in range(0, self.ra_indices_for_traj(self._itraj).tolist().count(self.line - 1)):
                    lines.append(row)
            else:
                lines.append(row)
            if self.chunksize != 0 and len(lines) % self.chunksize == 0:
                result = self._convert_to_np_chunk(lines)
                if self._t >= traj_len:
                    self._next_traj()
                return result

        self._next_traj()
        # last chunk
        if len(lines) > 0:
            result = self._convert_to_np_chunk(lines)
            return result

        self.close()

    def _next_traj(self):
        self._itraj += 1
        while not self.uniform_stride and self._itraj not in self.traj_keys \
                and self._itraj < self.number_of_trajectories():
            self._itraj += 1
        if self._itraj < self.number_of_trajectories():
            # close current file handle
            self._file_handle.close()
            # open next one
            self._open_file()
            # reset line counter
            self.line = 0
            # reset time counter
            self._t = 0
            # get new reader
            self._reader = csv.reader(self._file_handle,
                                      dialect=self._data_source._get_dialect(self._itraj))

    def _convert_to_np_chunk(self, list_of_strings):
        stack_of_strings = np.vstack(list_of_strings)
        del list_of_strings[:]
        try:
            result = stack_of_strings.astype(float)
        except ValueError:
            fn = self._data_source.filenames[self._itraj]
            for idx, line in enumerate(list_of_strings):
                for value in line:
                    try:
                        float(value)
                    except ValueError as ve:
                        s = "Invalid entry in file %s, line %s: %s" % (fn, self._t+idx, repr(ve))
                        raise ValueError(s)
        self._t += len(list_of_strings)
        return result

    def _open_file(self):
        # only apply _skip property at the beginning of the trajectory
        skip = self._data_source._skip[self._itraj] + self.skip if self._t == 0 else 0
        nt = self._data_source._skip[self._itraj] + self._data_source._lengths[self._itraj]

        # calculate an index set, which rows to skip (includes stride)
        skip_rows = np.empty(0)

        if skip > 0:
            skip_rows = np.zeros(nt)
            skip_rows[:skip] = np.arange(skip)

        if not self.uniform_stride:
            all_frames = np.arange(nt)
            skip_rows = np.setdiff1d(all_frames, self.ra_indices_for_traj(self._itraj), assume_unique=True)
        elif self.stride > 1:
            all_frames = np.arange(nt)
            if skip_rows is not None:
                wanted_frames = np.arange(skip, nt, self.stride)
            else:
                wanted_frames = np.arange(0, nt, self.stride)
            skip_rows = np.setdiff1d(
                    all_frames, wanted_frames, assume_unique=True)

        self._skip_rows = skip_rows
        try:
            fh = open(self._data_source.filenames[self._itraj])
            self._file_handle = fh
        except EnvironmentError:
            self._logger.exception()
            raise


class PyCSVReader(DataSource):
    r""" reads tabulated ASCII data

    Parameters
    ----------
    filenames: str or list of str
        files to be read

    chunksize: int, optional
        how much lines to process at once

    delimiters: str, list of str or None

        1. if str (eg. '\t'), then this delimiter is used for all filenames.
        2. list of delimiter strings, the length has to match the length of filenames.
        3. if not given, it will be guessed (may fail eg. for 1 dimensional data).

    comments: str, list of str or None, default='#'
        Lines starting with this char will be ignored, except for first line (header)

    converters : dict, optional
        A dictionary mapping column number to a function that will convert
        that column to a float.  E.g., if column 0 is a date string:
        ``converters = {0: datestr2num}``.  Converters can also be used to
        provide a default value for missing data:
        ``converters = {3: lambda s: float(s.strip() or 0)}``.

    """
    def __init__(self, filenames, chunksize=1000, delimiters=None, comments='#',
                 converters=None, **kwargs):
        super(PyCSVReader, self).__init__(chunksize=chunksize)
        self._is_reader = True

        n = len(filenames)
        self._comments = self.__parse_args(comments, '#', n)
        self._delimiters = self.__parse_args(delimiters, None, n)
        self._converters = converters

        # mapping of boolean to indicate if file has an header and csv dialect
        # self._has_headers = [False] * n
        self._dialects = [None] * n

        self._skip = np.zeros(n, dtype=int)
        # invoke filename setter
        self.filenames = filenames

    def __parse_args(self, arg, default, n):
        if arg is None:
            return [default]*n
        if isinstance(arg, (list, tuple)):
            assert len(arg) == n
            return arg
        return [arg]*n

    def _create_iterator(self, skip=0, chunk=0, stride=1, return_trajindex=True):
        return PyCSVIterator(self, skip=skip, chunk=chunk, stride=stride,
                             return_trajindex=return_trajindex)

    def _get_dialect(self, itraj):
        fn_idx = self.filenames.index(self.filenames[itraj])
        return self._dialects[fn_idx]

    def describe(self):
        return "[CSVReader files=%s]" % self._filenames

    def _get_traj_info(self, filename):
        idx = self.filenames.index(filename)

        def new_size(x):
            return int(ceil(x * 1.2))
        # how to handle mode?
        """
        On Windows, tell() can return illegal values (after an fgets()) when
        reading files with Unix-style line-endings. Use binary mode ('rb') to
        circumvent this problem.
        """
        with open(filename, 'rb') as fh:
            # approx by filesize / (first line + 20%)
            size = new_size(os.stat(filename).st_size / len(fh.readline()))
            assert size > 0
            fh.seek(0)
            offsets = np.empty(size, dtype=np.int64)
            offsets[0] = 0
            i = 1
            for _ in fh:  # for line in fh
                offsets[i] = fh.tell()
                i += 1
                if i >= len(offsets):
                    offsets = np.resize(offsets, new_size(len(offsets)))
            offsets = offsets[:i]
            length = len(offsets) - 1
            fh.seek(0)

            if not self._delimiters[idx]:
                # determine delimiter
                sample = fh.read(2048)
                sniffer = csv.Sniffer()
                self._dialects[idx] = sniffer.sniff(sample)
                if sniffer.has_header(sample):
                    self._skip[idx] += 1
                    length -= 1
            else:
                class custom_dialect(csv.Dialect):
                    delimiter = self._delimiters[idx]
                    quotechar = '"'
                    # TODO: this may cause problems if newline is only \r or \n
                    lineterminator = '\r\n'
                    quoting = csv.QUOTE_MINIMAL
                d = custom_dialect()
                d.delimiter = self._delimiters[idx]

                # determine header
                hdr = False
                while True:
                    line = fh.readline()
                    if line == '':
                        break
                    if line[0] == self._comments[idx]:
                        hdr += 1
                        continue

                self._skip[idx] += hdr
                length -= hdr

                self._dialects[idx] = d
            # if we have a header subtract it from total length
            fh.seek(0)
            r = csv.reader(fh, dialect=self._dialects[idx])
            for _ in range(self._skip[idx]+1):
                line = next(r)
            try:
                # determine file length
                with open(f) as fh:
                    # count rows
                    self._lengths.append(sum(1 for _ in fh))
                    fh.seek(0)
                    # determine if file has header here:
                    sample = fh.read(2048)
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

            try:
                arr = np.array(line).astype(float)
            except ValueError as ve:
                s = 'could not parse first line of data in file "%s"' % filename
                raise ValueError(s, ve)
            s = arr.squeeze().shape
            if len(s) == 1:
                ndim = s[0]
            else:
                ndim = 1

        return TrajInfo(ndim, length, offsets)
