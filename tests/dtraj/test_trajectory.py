
# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group
#
# MSMTools is free software: you can redistribute it and/or modify
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

r"""This module contains unit tests for the trajectory module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import os
import unittest

import numpy as np

from os.path import abspath, join
from os import pardir

from msmtools.dtraj import read_discrete_trajectory, write_discrete_trajectory, \
    load_discrete_trajectory, save_discrete_trajectory

testpath = abspath(join(abspath(__file__), pardir)) + '/testfiles/'
filename_csv = testpath + 'dtraj.dat'
filename_npy = testpath + 'dtraj.npy'
assert os.path.exists(filename_csv)
assert os.path.exists(filename_npy)


class TestReadDiscreteTrajectory(unittest.TestCase):

    def test_read_discrete_trajectory(self):
        dtraj_np = np.loadtxt(filename_csv, dtype=int)
        dtraj = read_discrete_trajectory(filename_csv)
        self.assertTrue(np.all(dtraj_np == dtraj))


class TestWriteDiscreteTrajectory(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.filename = tempfile.mktemp(suffix='.dat')
        self.dtraj = np.arange(10000)

    def tearDown(self):
        os.remove(self.filename)

    def test_write_discrete_trajectory(self):
        write_discrete_trajectory(self.filename, self.dtraj)
        dtraj_n = np.loadtxt(self.filename)
        self.assertTrue(np.all(dtraj_n == self.dtraj))


class TestLoadDiscreteTrajectory(unittest.TestCase):
    def test_load_discrete_trajectory(self):
        dtraj_n = np.load(filename_npy)
        dtraj = load_discrete_trajectory(filename_npy)
        self.assertTrue(np.all(dtraj_n == dtraj))


class TestSaveDiscreteTrajectory(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.filename = tempfile.mktemp(suffix='.npy')
        self.dtraj = np.arange(10000)

    def tearDown(self):
        os.remove(self.filename)

    def test_save_discrete_trajectory(self):
        save_discrete_trajectory(self.filename, self.dtraj)
        dtraj_n = np.load(self.filename)
        self.assertTrue(np.all(dtraj_n == self.dtraj))


if __name__ == "__main__":
    unittest.main()
