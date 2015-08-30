
'''
Created on 22.01.2015

@author: marscher
'''

from __future__ import absolute_import

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
            self._fh = open(self.filename, 'wb')
        except EnvironmentError:
            self._logger.exception('could not open file "%s" for writing.')
            raise

    def _param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=None, stride=1):
        np.savetxt(self._fh, X)
        if last_chunk:
            self._logger.debug("closing file")
            self._fh.close()
            return True  # finished
