import numpy as np

from pyemma._base.serialization.serialization import SerializableMixIn
from pyemma.coordinates.data._base.datasource import DataSourceIterator, DataSource

__author__ = 'marscher'


class SourcesMerger(DataSource, SerializableMixIn):
    __serialize_version = 0
    """ Combines multiple data sources to stream from.

    Note that you are responsible you only join matching (meaningful) data sets. If one trajectory is for instance
    shorter than the another, the longer one will be truncated during iteration.

    Parameters
    ----------
    sources : list, tuple
        list of DataSources (Readers, StreamingTransformers etc.) to combine for streaming access.

    chunk: int or None
        chunk size to use for underlying iterators.
    """
    def __init__(self, sources, chunk=None):
        super(SourcesMerger, self).__init__(chunksize=chunk)
        self.sources = sources
        self._is_reader = True
        self._ndim = sum(s.ndim for s in sources)
        ntraj = sources[0].ntraj
        for s in sources[1:]:
            if s.ntraj != ntraj:
                raise ValueError('different amount of trajectories to merge: {}'
                                 .format([(s, s.ntraj) for s in sources]))
        self._ntraj = ntraj
        self._filenames = []
        for s in sources:
            self._filenames += s.filenames
        import itertools
        for pair in itertools.combinations((s.trajectory_lengths() for s in sources), 2):
            if np.any(pair[0] != pair[1]):
                raise ValueError("currently only implemented for matching datasets.")
        self._lengths = sources[0].trajectory_lengths()

    def _create_iterator(self, skip=0, chunk=0, stride=1, return_trajindex=True, cols=None):
        return _JoiningIterator(self, self.sources, skip=skip, chunk=chunk, stride=stride,
                                return_trajindex=return_trajindex, cols=cols)

    def __reduce__(self):
        return SourcesMerger, (self.sources, self.chunksize)


class _JoiningIterator(DataSourceIterator):

    def __init__(self, src, sources, skip=0, chunk=0, stride=1, return_trajindex=False, cols=None):
        super(_JoiningIterator, self).__init__(src, skip, chunk,
                                               stride, return_trajindex, cols)
        self._iterators = [s.iterator(skip=skip, chunk=chunk, stride=stride,
                                      return_trajindex=return_trajindex, cols=cols)
                           for s in sources]
        self._selected_itraj = -1
        self.sources = sources

    def close(self):
        for it in self._iterators:
            it.close()

    def _next_chunk(self):
        # This method assumes that invoking next(iterator) handles file selections properly.
        # If one iterator raises stop iteration, this is propagated.
        chunks = []
        for it in self._iterators:
            if it.return_traj_index:
                itraj, X = next(it)
                assert itraj == self._itraj
            else:
                X = next(it)
            chunks.append(X)

        res = np.hstack(chunks)
        self._t += len(res)

        if self._t >= self.trajectory_length() and self._itraj < self._data_source.ntraj -1:
            self._itraj += 1
            self._select_file(self._itraj)

        return res

    def _select_file(self, itraj):
        if itraj != self._selected_itraj:
            self._t = 0
            self._itraj = itraj
            self._selected_itraj = itraj
            if __debug__:
                for it in self._iterators:
                    assert it._itraj == itraj
                    assert it._selected_itraj == itraj
                    assert it._t == self._t
