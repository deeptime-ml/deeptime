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

import csv

import numpy as np
from six.moves import range

from pyemma.coordinates.data.datasource import DataSource, DataSourceIterator


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
                                  dialect=self._data_source._dialects[self._itraj])

    def close(self):
        if self._file_handle is not None:
            self._file_handle.close()
        raise StopIteration()

    def next_chunk(self):
        if not self._file_handle or self._itraj >= self.number_of_trajectories():
            self.close()
        traj_len = self.trajectory_lengths()[self._itraj]
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
            if self.chunksize != 0 and self.line % self.chunksize == 0:
                result = self._convert_to_np_chunk(lines)
                del lines[:]
                if self._t >= traj_len:
                    self._next_traj()
                return result

        self._next_traj()
        # last chunk
        if len(lines) > 0:
            result = self._convert_to_np_chunk(lines)
            del lines[:]
            return result

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
            self._reader = csv.reader(self._file_handle, dialect=self._data_source._dialects[self._itraj])

    def _convert_to_np_chunk(self, list_of_strings):
        self._t += len(list_of_strings)
        stack_of_strings = np.vstack(list_of_strings)
        result = stack_of_strings.astype(float)
        return result

    def _open_file(self):
        fn = self._data_source._filenames[self._itraj]

        # only apply _skip property at the beginning of the trajectory
        skip = self._data_source._skip + self.skip if self._t == 0 else 0
        nt = self._data_source._lengths[self._itraj]

        if self._data_source._has_header[self._itraj]:
            nt += 1
            skip += 1

        # calculate an index set, which rows to skip (includes stride)
        skip_rows = []

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
            fh = open(fn)
            self._file_handle = fh
        except EnvironmentError:
            self._logger.exception()
            raise


class PyCSVReader(DataSource):
    def _create_iterator(self, skip=0, chunk=0, stride=1, return_trajindex=True):
        return PyCSVIterator(self, skip=skip, chunk=chunk, stride=stride,
                             return_trajindex=return_trajindex)

    def __init__(self, filenames, chunksize=1000, **kwargs):
        super(PyCSVReader, self).__init__(chunksize=chunksize)

        if not isinstance(filenames, (tuple, list)):
            filenames = [filenames]
        self._filenames = filenames
        # list of boolean to indicate if file
        self._has_header = [False] * len(self._filenames)

        self._dialects = [None] * len(self._filenames)

        # user wants to skip lines, so we need to remember this for lagged
        # access
        if kwargs.get('skip'):
            self._skip = kwargs.pop('skip')
        else:
            self._skip = 0

        self.__set_dimensions_and_lenghts()

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
            self._ndim = ndims[0]
