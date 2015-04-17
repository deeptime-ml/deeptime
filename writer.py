# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Created on 22.01.2015

@author: marscher
'''

import numpy as np
from pyemma.coordinates.transform.transformer import Transformer


class WriterCSV(Transformer):

    '''
    shall write to csv files
    '''

    def __init__(self, filename):
        '''
        Constructor
        '''
        super(WriterCSV, self).__init__()

        # filename should be obtained from source trajectory filename,
        # eg suffix it to given filename
        self.filename = filename
        self._last_frame = False

        self._reset()

    def describe(self):
        return "[Writer filename='%s']" % self.filename

    def dimension(self):
        return self.data_producer.dimension()

    def _map_array(self, X):
        pass

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
            self._fh = open(self.filename, 'w')
        except EnvironmentError:
            self._logger.exception('could not open file "%s" for writing.')
            raise

    def _param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=None, stride=1):
        np.savetxt(self._fh, X)
        if last_chunk:
            self._logger.debug("closing file")
            self._fh.close()
            return True  # finished
