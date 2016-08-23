
# This file is part of PyEMMA.
#
# Copyright (c) 2016 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
Created on 25.05.2016

@author: marscher
'''

import itertools
import os
import time
import warnings
from io import BytesIO
from operator import itemgetter

import numpy as np
from six import StringIO

from pyemma.coordinates.data.util.traj_info_cache import (UnknownDBFormatException,
                                                          TrajInfo,
                                                          TrajectoryInfoCache,
                                                          logger)
from pyemma.util import config


class AbstractDB(object):
    def set(self, value):
        # value: TrajInfo
        pass

    def update(self, value):
        pass

    def close(self):
        pass

    def sync(self):
        pass

    def get(self, key):
        # should raise KeyError in case of non existent key
        pass

    @property
    def db_version(self):
        pass

    @db_version.setter
    def db_version(self, val):
        pass


class DictDB(AbstractDB):
    def __init__(self):
        self._db = {}
        self.db_version = TrajectoryInfoCache.DB_VERSION

    def set(self, value):
        self._db[value.hash_value] = value

    def update(self, value):
        self._db[value.hash_value] = value

    @property
    def db_version(self):
        return self._db['version']

    @db_version.setter
    def db_version(self, version):
        self._db['version'] = version

    @property
    def num_entries(self):
        return len(self._db) - 1  # substract field for db_version


class SqliteDB(AbstractDB):
    def __init__(self, filename=None, clean_n_entries=30):
        """
        :param filename: path to database file
        :param clean_n_entries: during cleaning delete n % entries.
        """
        self.clean_n_entries = clean_n_entries
        import sqlite3

        # register numpy array conversion functions
        # uses "np_array" type in sql tables
        def adapt_array(arr):
            out = BytesIO()
            np.savez_compressed(out, offsets=arr)
            out.seek(0)
            return out.read()

        def convert_array(text):
            out = BytesIO(text)
            out.seek(0)
            npz = np.load(out)
            arr = npz['offsets']
            npz.close()
            return arr
        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, adapt_array)

        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("NPARRAY", convert_array)
        self._database = sqlite3.connect(filename if filename is not None else ":memory:",
                                         detect_types=sqlite3.PARSE_DECLTYPES, timeout=1000*1000,
                                         isolation_level=None)
        self.filename = filename

        try:
            cursor = self._database.execute("select num from version")
            row = cursor.fetchone()
            if not row:
                self.db_version = TrajectoryInfoCache.DB_VERSION
                version = self.db_version
            else:
                version = row[0]
            if version != TrajectoryInfoCache.DB_VERSION:
                # drop old db? or try to convert?
                self._create_new_db()
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                self._create_new_db()
                self.db_version = TrajectoryInfoCache.DB_VERSION
        except sqlite3.DatabaseError:
            bak = filename + ".bak"
            warnings.warn("TrajInfo database corrupted. Backing up file to %s and start with new one." % bak)
            self._database.close()
            import shutil
            shutil.move(filename, bak)
            SqliteDB.__init__(self, filename)

    def _create_new_db(self):
        # assumes self.database is a sqlite3.Connection
        create_version_table = "CREATE TABLE version (num INTEGER PRIMARY KEY);"
        create_info_table = """CREATE TABLE traj_info(
            hash VARCHAR(64) PRIMARY KEY,
            length INTEGER,
            ndim INTEGER,
            offsets NPARRAY,
            abs_path VARCHAR(4096) null,
            version INTEGER,
            lru_db INTEGER
        );
        """
        self._database.execute(create_version_table)
        self._database.execute(create_info_table)
        self._database.commit()

    def close(self):
        self._database.close()

    @property
    def db_version(self):
        cursor = self._database.execute("select num from version")
        row = cursor.fetchone()
        if not row:
            raise RuntimeError("unknown db version")
        return row[0]

    @db_version.setter
    def db_version(self, val):
        self._database.execute("insert into version VALUES (?)", [val])
        self._database.commit()

    @property
    def num_entries(self):
        # cursor = self._database.execute("SELECT hash FROM traj_info;")
        # return len(cursor.fetchall())
        c = self._database.execute("SELECT COUNT(hash) from traj_info;").fetchone()
        return int(c[0])

    def set(self, traj_info):
        import sqlite3
        values = (
            traj_info.hash_value, traj_info.length, traj_info.ndim,
            np.array(traj_info.offsets), traj_info.abs_path, TrajectoryInfoCache.DB_VERSION,
            # lru db
            self._database_from_key(traj_info.hash_value)
        )
        statement = ("INSERT INTO traj_info (hash, length, ndim, offsets, abs_path, version, lru_db)"
                     "VALUES (?, ?, ?, ?, ?, ?, ?)", values)
        try:
             self._database.execute(*statement)
        except sqlite3.IntegrityError as ie:
            logger.exception("insert failed: %s " % ie)
            return
        self._database.commit()

        self._update_time_stamp(hash_value=traj_info.hash_value)

        if self.filename is not None:
            current_size = os.stat(self.filename).st_size
            if (self.num_entries >= config.traj_info_max_entries or
                    # current_size is in bytes, while traj_info_max_size is in MB
                    1.*current_size / 1024**2 >= config.traj_info_max_size):
                logger.info("Cleaning database because it has too much entries or is too large.\n"
                            "Entries: %s. Size: %.2fMB. Configured max_entires: %s. Max_size: %sMB"
                            % (self.num_entries, (current_size*1.0 / 1024**2),
                               config.traj_info_max_entries, config.traj_info_max_size))
                self._clean(n=self.clean_n_entries)

    def get(self, key):
        cursor = self._database.execute("SELECT * FROM traj_info WHERE hash=?", (key,))
        row = cursor.fetchone()
        if not row:
            raise KeyError()
        info = self._create_traj_info(row)
        self._update_time_stamp(key)
        return info

    def _database_from_key(self, key):
        """
        gets the database name for the given key. Should ensure a uniform spread
        of keys over the databases in order to minimize waiting times. Since the
        database has to be locked for updates and multiple processes want to write,
        each process has to wait until the lock has been released.

        By default the LRU databases will be stored in a sub directory "traj_info_usage"
        lying next to the main database.

        :param key: hash of the TrajInfo instance
        :return: str, database path
        """
        if not self.filename:
            return None

        from pyemma.util.files import mkdir_p
        hash_value_long = int(key, 16)
        # bin hash to one of either 10 different databases
        # TODO: make a configuration parameter out of this number
        db_name = str(hash_value_long)[-1] + '.db'
        directory = os.path.dirname(self.filename) + os.path.sep + 'traj_info_usage'
        mkdir_p(directory)
        return os.path.join(directory, db_name)

    def _update_time_stamp(self, hash_value):
        """ timestamps are being stored distributed over several lru databases.
        The timestamp is a time.time() snapshot (float), which are seconds since epoch."""
        db_name = self._database_from_key(hash_value)
        if not db_name:
            db_name=':memory:'

        import sqlite3

        with sqlite3.connect(db_name) as conn:
            """ last_read is a result of time.time()"""
            conn.execute('CREATE TABLE IF NOT EXISTS usage '
                         '(hash VARCHAR(32), last_read FLOAT)')
            conn.commit()
            cur = conn.execute('select * from usage where hash=?', (hash_value,))
            row = cur.fetchone()
            if not row:
                conn.execute("insert into usage(hash, last_read) values(?, ?)", (hash_value, time.time()))
            else:
                conn.execute("update usage set last_read=? where hash=?", (time.time(), hash_value))
            conn.commit()

    @staticmethod
    def _create_traj_info(row):
        # convert a database row to a TrajInfo object
        try:
            hash = row[0]
            length = row[1]
            ndim = row[2]
            offsets = row[3]
            assert isinstance(offsets, np.ndarray)
            abs_path = row[4]
            version = row[5]

            info = TrajInfo()
            info._version = version
            if version == 2:
                info._hash = hash
                info._ndim = ndim
                info._length = length
                info._offsets = offsets
                info._abs_path = abs_path
            else:
                raise ValueError("unknown version %s" % version)
            return info
        except Exception as ex:
            logger.exception(ex)
            raise UnknownDBFormatException(ex)

    @staticmethod
    def _format_tuple_for_sql(value):
        value = tuple(str(v) for v in value)
        return repr(value)[1:-2 if len(value) == 1 else -1]

    def _clean(self, n):
        """
        obtain n% oldest entries by looking into the usage databases. Then these entries
        are deleted first from the traj_info db and afterwards from the associated LRU dbs.

        :param n: delete n% entries in traj_info db [and associated LRU (usage) dbs].
        """
        # delete the n % oldest entries in the database
        import sqlite3
        num_delete = int(self.num_entries / 100.0 * n)
        logger.debug("removing %i entries from db" % num_delete)
        lru_dbs = self._database.execute("select hash, lru_db from traj_info").fetchall()
        lru_dbs.sort(key=itemgetter(1))
        hashs_by_db = {}
        age_by_hash = []
        for k, v in itertools.groupby(lru_dbs, key=itemgetter(1)):
            hashs_by_db[k] = list(x[0] for x in v)

        # debug: distribution
        len_by_db = {os.path.basename(db): len(hashs_by_db[db]) for db in hashs_by_db.keys()}
        logger.debug("distribution of lru: %s" % str(len_by_db))
        ### end dbg

        self.lru_timeout = 1000 #1 sec

        # collect timestamps from databases
        for db in hashs_by_db.keys():
            with sqlite3.connect(db, timeout=self.lru_timeout) as conn:
                rows = conn.execute("select hash, last_read from usage").fetchall()
                for r in rows:
                    age_by_hash.append((r[0], float(r[1]), db))

        # sort by age
        age_by_hash.sort(key=itemgetter(1))
        if len(age_by_hash)>=2:
            assert[age_by_hash[-1] > age_by_hash[-2]]
        ids = map(itemgetter(0), age_by_hash[:num_delete])
        ids = tuple(map(str, ids))

        sql_compatible_ids = SqliteDB._format_tuple_for_sql(ids)

        stmnt = "DELETE FROM traj_info WHERE hash in (%s)" % sql_compatible_ids
        cur = self._database.execute(stmnt)
        self._database.commit()
        assert cur.rowcount == len(ids), "deleted not as many rows(%s) as desired(%s)" %(cur.rowcount, len(ids))

        # iterate over all LRU databases and delete those ids, we've just deleted from the main db.
        age_by_hash.sort(key=itemgetter(2))
        for db, values in itertools.groupby(age_by_hash, key=itemgetter(2)):
            values = tuple(v[0] for v in values)
            with sqlite3.connect(db, timeout=self.lru_timeout) as conn:
                    stmnt = "DELETE FROM usage WHERE hash IN (%s)" \
                            % SqliteDB._format_tuple_for_sql(values)
                    curr = conn.execute(stmnt)
                    assert curr.rowcount == len(values), curr.rowcount
