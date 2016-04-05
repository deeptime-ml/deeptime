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

import six

from pyemma._base.logging import Loggable
from pyemma.coordinates.data._base.datasource import DataSourceIterator, DataSource
from pyemma.coordinates.data.util.traj_info_cache import TrajInfo
from six.moves import range
import numpy as np


class PyCSVIterator(DataSourceIterator):
    def __init__(self, data_source, skip=0, chunk=0, stride=1, return_trajindex=False, cols=None):
        # do not pass cols, because we want to handle in this impl, not in DataSourceIterator
        super(PyCSVIterator, self).__init__(data_source, skip=skip, chunk=chunk,
                                            stride=stride,
                                            return_trajindex=return_trajindex)
        self._custom_cols = cols
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
                del lines[:]  # free some space
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
        # filter empty strings
        list_of_strings = list(filter(bool, list_of_strings))
        stack_of_strings = np.vstack(list_of_strings)
        if self._custom_cols:
            stack_of_strings = stack_of_strings[:, self._custom_cols]
        try:
            result = stack_of_strings.astype(float)
        except ValueError:
            fn = self._file_handle.name
            dialect_str = _dialect_to_str(self._reader.dialect)
            for idx, line in enumerate(list_of_strings):
                for value in line:
                    try:
                        float(value)
                    except ValueError as ve:
                        s = str("Invalid entry in file {fn}, line {line}: {error}."
                                " Used dialect to parse: {dialect}").format(fn=fn, line=self._t + idx,
                                                                            error=repr(ve),
                                                                            dialect=dialect_str)
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
            fh = open(self._data_source.filenames[self._itraj],
                      mode=self._data_source.DEFAULT_OPEN_MODE)
            self._file_handle = fh
        except EnvironmentError:
            self._logger.exception()
            raise


def _dialect_to_str(dialect):
    from io import StringIO
    s = StringIO()
    s.write("[CSVDialect ")
    fields = str("delimiter='{delimiter}', lineterminator='{lineterminator}',"
                 " skipinitialspace={skipinitialspace}, quoting={quoting},"
                 " quotechar={quotechar}, doublequote={doublequote}]")
    s.write(fields.format(delimiter=dialect.delimiter,
                          lineterminator=dialect.lineterminator,
                          skipinitialspace=dialect.skipinitialspace,
                          quoting=dialect.quoting,
                          quotechar=dialect.quotechar,
                          doublequote=dialect.doublequote))
    s.seek(0)
    return str(s.read())


class PyCSVReader(DataSource):
    r""" Reader for tabulated ASCII data

    This class uses numpy to interpret string data to array data.

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

    converters : dict, optional (Not yet implemented)
        A dictionary mapping column number to a function that will convert
        that column to a float.  E.g., if column 0 is a date string:
        ``converters = {0: datestr2num}``.  Converters can also be used to
        provide a default value for missing data:
        ``converters = {3: lambda s: float(s.strip() or 0)}``.

    Notes
    -----
    For reading files with only one column, one needs to specify a delimter...
    """
    DEFAULT_OPEN_MODE = 'r'  # read in text-mode

    def __init__(self, filenames, chunksize=1000, delimiters=None, comments='#',
                 converters=None, **kwargs):
        super(PyCSVReader, self).__init__(chunksize=chunksize)
        self._is_reader = True

        if isinstance(filenames, (tuple, list)):
            n = len(filenames)
        elif isinstance(filenames, six.string_types):
            n = 1
        else:
            raise TypeError("'filenames' argument has to be list, tuple or string")
        self._comments = PyCSVReader.__parse_args(comments, '#', n)
        self._delimiters = PyCSVReader.__parse_args(delimiters, None, n)
        self._converters = converters

        # mapping of boolean to indicate if file has an header and csv dialect
        # self._has_headers = [False] * n
        self._dialects = [None] * n

        self._skip = np.zeros(n, dtype=int)
        # invoke filename setter
        self.filenames = filenames

    @staticmethod
    def __parse_args(arg, default, n):
        if arg is None:
            return [default] * n
        if isinstance(arg, (list, tuple)):
            assert len(arg) == n
            return arg
        return [arg] * n

    def _create_iterator(self, skip=0, chunk=0, stride=1, return_trajindex=True, cols=None):
        return PyCSVIterator(self, skip=skip, chunk=chunk, stride=stride,
                             return_trajindex=return_trajindex, cols=cols)

    def _get_dialect(self, itraj):
        fn_idx = self.filenames.index(self.filenames[itraj])
        return self._dialects[fn_idx]

    def describe(self):
        return "[CSVReader files=%s]" % self._filenames

    def _determine_dialect(self, fh, length):
        """
        Parameters
        ----------
        fh : file handle
            file handle for which the dialect should be determined.

        length : int
            the previously obtained file length (from _get_file_info)

        Returns
        -------
        dialect : csv.Dialect
            an Dialect instance which holds things like delimiter etc.
        length : int
            length with respect to eventually found header (one or multiple lines)

        Notes
        -----
        As a side effect this method sets the dialect for the given file handle in
        self._dialects[idx] where idx = self.filenames.index(fh.name)
        """
        filename = fh.name
        idx = self.filenames.index(filename)
        fh.seek(0)
        skip = 0  # rows to skip (for a eventually found header)
        # auto detect delimiter with csv.Sniffer
        if self._delimiters[idx] is None:
            # use a sample of three lines
            sample = ''.join(fh.readline() for _ in range(3))
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(sample)
            except csv.Error as e:
                s = ('During handling of file "%s" following error occurred:'
                     ' "%s". Sample was "%s"' % (filename, e, sample))
                raise RuntimeError(s)
            if sniffer.has_header(sample):
                skip += 1
        else:
            sample = fh.readline()
            fh.seek(0)

            class custom_dialect(csv.Dialect):
                delimiter = self._delimiters[idx]
                quotechar = '"'
                if sample[-2] == '\r' and sample[-1] == '\n':
                    lineterminator = '\r\n'
                else:
                    lineterminator = '\n'
                quoting = csv.QUOTE_MINIMAL

            dialect = custom_dialect()
            dialect.delimiter = self._delimiters[idx]
            # determine header (multi-line)
            hdr = False
            for line in fh:
                if line.startswith(self._comments[idx]):
                    hdr += 1
                    continue
                else:
                    break

            skip += hdr
        length -= skip

        self._dialects[idx] = dialect
        self._skip[idx] = skip

        return dialect, length, skip

    @staticmethod
    def _calc_offsets(fh):
        """ determines byte offsets between all lines
        Parameters
        ----------
        fh : file handle
        file handle to obtain byte offsets from.

        Returns
        -------
        lengths : int
            number of valid (non-empty) lines
        offsets : ndarray(dtype=int64)
            byte offsets
        """

        def new_size(x):
            return int(ceil(x * 1.2))

        filename = fh.name
        fh.seek(0)
        # approx by filesize / (first line + 20%)
        fh.readline()  # skip first line, because it may contain a much shorter header, which will give a bad estimate
        size = new_size(os.stat(filename).st_size / len(fh.readline()))
        offsets = np.empty(size, dtype=np.int64)
        offsets[0] = 0
        i = 1
        # re-open in binary mode to circumvent a bug in Py3.5 win, where the first offset reported by tell
        # overflows int64.
        with open(filename, 'rb') as fh:
            while fh.readline():
                offsets[i] = fh.tell()
                i += 1
                if i >= len(offsets):
                    offsets = np.resize(offsets, new_size(len(offsets)))
        offsets = offsets[:i]

        # filter empty lines (offset between two lines is only 1 or 2 chars)
        # insert an diff of 2 at the beginning to match the amount of indices
        diff = np.diff(offsets)
        mask = diff > 2
        mask = np.insert(mask, 0, True)
        offsets = offsets[mask]
        length = len(offsets) - 1

        return length, offsets

    @staticmethod
    def _get_dimension(fh, dialect, skip):
        fh.seek(0)
        # if we have a header subtract it from total length
        r = csv.reader(fh, dialect=dialect)
        for _ in range(skip + 1):
            line = next(r)

        # obtain dimension from first valid row
        try:
            arr = np.array(line).astype(float)
        except ValueError as ve:
            s = 'could not parse first line of data in file "%s"' % fh.name
            raise ValueError(s, ve)
        s = arr.squeeze().shape
        if len(s) == 1:
            ndim = s[0]
        else:
            ndim = 1

        return ndim

    def _get_traj_info(self, filename):
        # calc byte offsets, csv dialect and dimension (elements in first valid row)

        with open(filename, self.DEFAULT_OPEN_MODE) as fh:
            length, offsets = PyCSVReader._calc_offsets(fh)
            dialect, length, skip = self._determine_dialect(fh, length)
            ndim = PyCSVReader._get_dimension(fh, dialect, skip)

        return TrajInfo(ndim, length, offsets)
