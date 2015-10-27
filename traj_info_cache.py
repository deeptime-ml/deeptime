
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
from six import PY2
if PY2:
    import anydbm
else:
    import dbm as anydbm

import os
import numpy as np
import mdtraj

from mdtraj.formats.registry import _FormatRegistry as md_registry
from threading import Semaphore
from pyemma.util.config import conf_values

__all__ = ('TrajectoryInfoCache')

# TODO: add complete shape info to use this also for numpy/csv files
class _TrajectoryInfoCache(object):

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

    def __init__(self, database_filename=None):
        if database_filename is not None:
            self._database = anydbm.open(database_filename, flag="c")
        else:
            self._database = {}
        self._write_protector = Semaphore()

    def __getitem__(self, filename):
        key = self.__get_file_hash(filename)
        result = None
        try:
            result = self._database[key]
        except KeyError:
            result = self.__determine_len(filename)
            self.__setitem__(filename, result, key=key)
        return int(result)

    def __determine_len(self, filename):
        _, ext = os.path.splitext(filename)
        if ext in md_registry.loaders:
            with mdtraj.open(filename) as fh:
                return len(fh)
        elif ext in ('.npy'):
            x = np.load(filename, mmap_mode='r')
            return len(x)
        else:
            raise ValueError('file %s is unsupported.')

    def __format_value(self, filename, length):
        return str(length)

    def __get_file_hash(self, filename):
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

    def __setitem__(self, filename, n_frames, key=None):
        if not key:
            key = self.__get_file_hash(filename)
        self._write_protector.acquire()
        self._database[key] = self.__format_value(filename, n_frames)
        self._write_protector.release()


# singleton pattern
cfg_dir = conf_values['pyemma']['cfg_dir']
filename = os.path.join(cfg_dir, "trajlen_cache")
TrajectoryInfoCache = _TrajectoryInfoCache(filename)