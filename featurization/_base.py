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
Created on 15.02.2016

@author: marscher
'''
import six

class Feature(object):

    @property
    def dimension(self):
        return self._dim

    @dimension.setter
    def dimension(self, val):
        assert isinstance(val, int)
        self._dim = val

    @property
    def top(self):
        return self._top

    @top.setter
    def top(self, value):
        import mdtraj
        if isinstance(value, six.string_types):
            value = mdtraj.load(value).top
        assert isinstance(value, mdtraj.Topology)
        self._top = value

    def _ensure_topfile(self):
        if not hasattr(self.top, 'fname') or self.top.fname is None:
            raise ValueError("not file name for topology available")

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __repr__(self):
        return str(self.describe())
