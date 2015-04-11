'''
Created on 22.01.2015

@author: marscher
'''
from pyemma.util.log import getLogger

import numpy as np
from pyemma.coordinates.transform.transformer import Transformer


log = getLogger('WriterCSV')
__all__ = ['WriterCSV']


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
        self.last_frame = False

        self._reset()

    def describe(self):
        return "[Writer filename='%s']" % self.filename

    def _get_constant_memory(self):
        return 0

    def dimension(self):
        return self.data_producer.dimension()

    def _reset(self, stride=1):
        try:
            self.fh.close()
            log.debug('closed file')
        except IOError:
            log.exception('during close')
        except AttributeError:
            pass

        try:
            self.fh = open(self.filename, 'w')
        except IOError:
            log.exception('could not open file "%s" for writing.')
            raise

    def _param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=None, stride=1):
        np.savetxt(self.fh, X)
        if last_chunk:
            log.debug("closing file")
            self.fh.close()
            return True  # finished
