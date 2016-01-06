
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
Created on 22.01.2015

@author: marscher
'''

from __future__ import absolute_import

import numpy as np
from pyemma.coordinates.transform.transformer import Transformer

# TODO: this should be either a new type (data_consumer, eg. end point) or
# deleted


class WriterCSV(Transformer):

    '''
    write to csv files via numpy
    '''

    def __init__(self, filename, stride=1):
        '''
        Constructor
        '''
        super(WriterCSV, self).__init__()

        # filename should be obtained from source trajectory filename,
        # eg suffix it to given filename
        self.filename = filename
        self.stride = stride
        self._last_frame = False

        self._reset()

    def describe(self):
        return "[Writer filename='%s']" % self.filename

    def dimension(self):
        return self.data_producer.dimension()

    def _reset(self, stride=1):
        try:
            self._fh.close()
            self._logger.debug('closed file')
        except EnvironmentError:
            self._logger.exception('during close')
        except AttributeError:
            # no file handle exists yet
            pass

        try:
            self._fh = open(self.filename, 'wb')
        except EnvironmentError:
            self._logger.exception('could not open file "%s" for writing.')
            raise

    def _transform_array(self, X):
        raise

    def _estimate(self, iterable, **kw):
        it = iterable.iterator(stride=self.stride, return_trajindex=False)
        for X in it:
            np.savetxt(self._fh, X)
        self._fh.close()

        return self
