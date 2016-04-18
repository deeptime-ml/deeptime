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

from io import BytesIO
from logging import getLogger
import os
from threading import Semaphore

from pyemma.util import config
import six
import numpy as np
if six.PY2:
    import dumbdbm
else:
    from dbm import dumb as dumbdbm

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

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.version == other.version
                and self.hash_value == other.hash_value
                and self.ndim == other.ndim
                and self.length == other.length
                and np.all(self.offsets == other.offsets)
                )


def create_traj_info(db_val):
    assert isinstance(db_val, (six.string_types, bytes))
    if six.PY3 and isinstance(db_val, six.string_types):
        db_val = bytes(db_val.encode('utf-8', errors='ignore'))
    fh = BytesIO(db_val)

    try:
        arr = np.load(fh)['data']
        info = TrajInfo()
        header = arr[0]

        version = header['data_format_version']
        info._version = version
        if version == 1:
            info._hash = header['filehash']
            info._ndim = arr[1]
            info._length = arr[2]
            info._offsets = arr[3]
        else:
            raise ValueError("unknown version %s" % version)
        return info
    except Exception as ex:
        raise UnknownDBFormatException(ex)


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
    DB_VERSION = '1'

    @staticmethod
    def instance():
        if TrajectoryInfoCache._instance is None:
            # singleton pattern
            filename = os.path.join(config.cfg_dir, "trajlen_cache")
            TrajectoryInfoCache._instance = TrajectoryInfoCache(filename)

            # sync db to hard drive at exit.
            if hasattr(TrajectoryInfoCache._instance._database, 'sync'):
                import atexit
                @atexit.register
                def write_at_exit():
                    TrajectoryInfoCache._instance._database.sync()

        return TrajectoryInfoCache._instance

    def __init__(self, database_filename=None):
        # for now we disable traj info cache persistence!
        database_filename = None
        self.database_filename = database_filename
        if database_filename is not None:
            try:
                self._database = dumbdbm.open(database_filename, flag="c")
            except dumbdbm.error as e:
                try:
                    os.unlink(database_filename)
                    self._database = dumbdbm.open(database_filename, flag="n")
                    # persist file right now, since it was broken
                    self._set_curr_db_version(TrajectoryInfoCache.DB_VERSION)
                    # close and re-open to ensure file exists
                    self._database.close()
                    self._database = dumbdbm.open(database_filename, flag="w")
                except OSError:
                    raise RuntimeError('corrupted database in "%s" could not be deleted'
                                       % os.path.abspath(database_filename))
        else:
            self._database = {}

        self._set_curr_db_version(TrajectoryInfoCache.DB_VERSION)
        self._write_protector = Semaphore()

    @property
    def current_db_version(self):
        return self._current_db_version

    def _set_curr_db_version(self, val):
        self._database['db_version'] = val
        self._current_db_version = val

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
        key = self._get_file_hash(filename)
        result = None
        try:
            result = str(self._database[key])
            info = create_traj_info(result)

            self._handle_csv(reader, filename, info.length)

        # handle cache misses and not interpretable results by re-computation.
        # Note: this also handles UnknownDBFormatExceptions!
        except KeyError:
            info = reader._get_traj_info(filename)
            info.hash_value = key
            # store info in db
            result = self.__setitem__(filename, info)

            # save forcefully now
            if hasattr(self._database, 'sync'):
                logger.debug("sync db after adding new entry")
                self._database.sync()

        return info

    def __format_value(self, traj_info):
        assert traj_info.hash_value != -1
        fh = BytesIO()

        header = {'data_format_version': 1,
                  'filehash': traj_info.hash_value,  # back reference to file by hash
                  }

        array = np.empty(4, dtype=object)

        array[0] = header
        array[1] = traj_info.ndim
        array[2] = traj_info.length
        array[3] = traj_info.offsets

        np.savez_compressed(fh, data=array)
        fh.seek(0)
        return fh.read()

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

    def __setitem__(self, filename, traj_info):
        dbval = self.__format_value(traj_info)

        self._write_protector.acquire()
        self._database[str(traj_info.hash_value)] = dbval
        self._write_protector.release()

        return dbval

    def clear(self):
        self._database.clear()
