'''
Created on 30.04.2015

@author: marscher
'''
import anydbm
import os
import mdtraj
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
        with mdtraj.open(filename) as fh:
            return len(fh)

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
