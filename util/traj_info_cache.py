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
Created on 30.04.2015

@author: marscher
'''

from __future__ import absolute_import

import hashlib
import os
import sys
import warnings
from io import BytesIO
from logging import getLogger

import numpy as np

from pyemma.util import config

logger = getLogger(__name__)


__all__ = ('TrajectoryInfoCache', 'TrajInfo')


class UnknownDBFormatException(KeyError):
    pass


class TrajInfo(object):

    def __init__(self, ndim=0, length=0, offsets=None):
        self._ndim = ndim
        self._length = length
        if offsets is None:
            self._offsets = []
        else:
            self._offsets = offsets

        self._version = 1
        self._hash = -1
        self._abs_path = None

    @property
    def version(self):
        return self._version

    @property
    def ndim(self):
        return self._ndim

    @property
    def length(self):
        return self._length

    @property
    def offsets(self):
        return self._offsets

    @offsets.setter
    def offsets(self, value):
        self._offsets = np.asarray(value, dtype=np.int64)

    @property
    def hash_value(self):
        return self._hash

    @hash_value.setter
    def hash_value(self, val):
        self._hash = val

    @property
    def abs_path(self):
        return self._abs_path

    @abs_path.setter
    def abs_path(self, val):
        self._abs_path = val

    def offsets_to_bytes(self):
        assert self.hash_value != -1
        fh = BytesIO()
        np.savez_compressed(fh, offsets=self.offsets)
        fh.seek(0)
        return fh.read()

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.version == other.version
                and self.hash_value == other.hash_value
                and self.ndim == other.ndim
                and self.length == other.length
                and np.all(self.offsets == other.offsets)
                )

    def __str__(self):
        return "[TrajInfo hash={hash}, len={len}, dim={dim}, path={path}". \
            format(hash=self.hash_value, len=self.length, dim=self.ndim, path=self.abs_path)


class TrajectoryInfoCache(object):

    """ stores trajectory lengths associated to a file based hash (mtime, name, 1mb of data)

    Parameters
    ----------
    database_filename : str (optional)
        if given the cache is being made persistent to this file. Otherwise the
        cache is lost after the process has finished.

    Notes
    -----
    Do not instantiate this yourself, but use the instance provided by this
    module.

    """
    _instance = None
    DB_VERSION = 2

    @staticmethod
    def instance():
        """ :returns the TrajectoryInfoCache singleton instance"""
        if TrajectoryInfoCache._instance is None:
            # if we do not have a configuration director yet, we do not want to store
            if not config.cfg_dir:
                filename = None
            else:
                filename = os.path.join(config.cfg_dir, "traj_info.sqlite3")
            TrajectoryInfoCache._instance = TrajectoryInfoCache(filename)

        return TrajectoryInfoCache._instance

    def __init__(self, database_filename=None):
        self.database_filename = database_filename

        # have no filename, use in memory sqlite db
        # have no sqlite module, use dict
        # have sqlite and file, create db with given filename

        try:
            import sqlite3
            from pyemma.coordinates.data.util.traj_info_backends import SqliteDB
            self._database = SqliteDB(self.database_filename)
        except ImportError:
            warnings.warn("sqlite3 package not available, persistant storage of trajectory info not possible!")
            from pyemma.coordinates.data.util.traj_info_backends import DictDB
            self._database = DictDB()

    @property
    def current_db_version(self):
        return self._database.db_version

    @property
    def num_entries(self):
        return self._database.num_entries

    def _handle_csv(self, reader, filename, length):
        # this is maybe a bit ugly, but so far we do not store the dialect of csv files in
        # the database, so we need to re-do this step in case of a cache hit.
        from pyemma.coordinates.data import PyCSVReader
        if not isinstance(reader, PyCSVReader):
            return
        with open(filename, PyCSVReader.DEFAULT_OPEN_MODE) as fh:
            reader._determine_dialect(fh, length)

    def __getitem__(self, filename_reader_tuple):
        filename, reader = filename_reader_tuple
        abs_path = os.path.abspath(filename)
        key = self._get_file_hash_v2(filename)
        try:
            info = self._database.get(key)
            if not isinstance(info, TrajInfo):
                raise KeyError()
            self._handle_csv(reader, filename, info.length)
            # if path has changed, update it
            if not info.abs_path == abs_path:
                info.abs_path = abs_path
                self._database.update(info)
        # handle cache misses and not interpretable results by re-computation.
        # Note: this also handles UnknownDBFormatExceptions!
        except KeyError:
            info = reader._get_traj_info(filename)
            info.hash_value = key
            info.abs_path = abs_path
            # store info in db
            self.__setitem__(info)

            # save forcefully now
            if hasattr(self._database, 'sync'):
                self._database.sync()

        return info

    def _get_file_hash(self, filename):
        statinfo = os.stat(filename)

        # only remember file name without path, to re-identify it when its
        # moved
        hash_value = hash(os.path.basename(filename))
        hash_value ^= hash(statinfo.st_mtime)
        hash_value ^= hash(statinfo.st_size)

        # now read the first megabyte and hash it
        with open(filename, mode='rb') as fh:
            data = fh.read(1024)

        hash_value ^= hash(data)
        return str(hash_value)

    def _get_file_hash_v2(self, filename):
        statinfo = os.stat(filename)
        # now read the first megabyte and hash it
        with open(filename, mode='rb') as fh:
            data = fh.read(1024)

        if sys.version_info > (3,):
            long = int

        hasher = hashlib.md5()
        hasher.update(os.path.basename(filename).encode('utf-8'))
        hasher.update(str(statinfo.st_mtime).encode('ascii'))
        hasher.update(str(statinfo.st_size).encode('ascii'))
        hasher.update(data)
        return hasher.hexdigest()

    def __setitem__(self, traj_info):
        self._database.set(traj_info)

    def clear(self):
        self._database.clear()

    def close(self):
        """ you most likely never want to call this! """
        self._database.close()
