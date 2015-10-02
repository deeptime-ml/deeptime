'''
Created on 22.07.2015

@author: marscher
'''

import numpy as np
from pyemma.coordinates.transform.transformer import Transformer


class Sparsifier(Transformer):

    def __init__(self, rtol=1e-2):
        Transformer.__init__(self)
        self._varying_indices = None
        self._first_frame = None
        self._rtol = rtol

    @property
    def rtol(self):
        return self._rtol

    def _param_init(self):
        # TODO: determine if final data_producer is a feature_reader and if so
        # it has a data structure providing insight into sparse features, so
        # we do not have to calculate them again...
        self._varying_indices = []

    def describe(self):
        return self.__class__.__name__ + 'dim: %i' % self.dimension() if self._parametrized else ''

    def dimension(self):
        if not self._parametrized:
            raise RuntimeError(
                "Sparsifier does not know its output dimension yet.")
        dim = len(self._varying_indices)
        return dim

    def _param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj,
                        last_chunk, ipass, Y=None, stride=1):

        if ipass == 0:
            if t == 0:
                self._first_frame = X[0]

            close = np.isclose(X, self._first_frame, rtol=self.rtol)
            not_close = np.logical_not(close)
            close_cols = np.argwhere(not_close)[:, 1]
            var_inds = np.unique(close_cols)
            self._varying_indices = np.union1d(var_inds, self._varying_indices)

            if last_chunk:
                return True

        return False

    def _param_finish(self):
        self._parametrized = True
        self._logger.warning("Detected and eliminated %i constant features"
                             % (self.data_producer.dimension() - self.dimension()))
        self._varying_indices = np.array(self._varying_indices, dtype=int)

    def _transform_array(self, X):
        return X[:, self._varying_indices]
