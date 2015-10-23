from pyemma.coordinates.data.interface import ReaderInterface

from six import string_types
from pyemma.coordinates.api import source

import numpy as np


class FragmentedTrajectoryReader(ReaderInterface):

    def __init__(self, trajectories, topologyfile=None, chunksize=100, featurizer=None):
        # sanity checks
        assert isinstance(trajectories, (list, tuple)), "input trajectories should be of list or tuple type"
        # call super
        super(FragmentedTrajectoryReader, self).__init__(chunksize=chunksize)
        # create input sources for the fragments
        readers = []
        for input_item in trajectories:
            reader = source(input_item, features=featurizer, top=topologyfile, chunk_size=chunksize)
            readers.append(reader)
        # store readers
        self._readers = readers
        # store trajectory files
        self._trajectories = trajectories
        # one (composite) trajectory
        self._ntraj = 1
        # lengths array per reader
        self._reader_lengths = [reader.trajectory_length(0, 1) for reader in self._readers]
        # composite trajectory length
        self._lengths = [sum(self._reader_lengths)]
        # mapping reader_index -> cumulative length
        self._cumulative_lengths = np.cumsum(self._reader_lengths)

    def describe(self):
        return "[FragmentedTrajectoryReader files=%s]" % self._trajectories

    def dimension(self):
        return self._readers[0].dimension()

    def _close(self):
        for reader in self._readers:
            reader._close()

    def _reset(self, context=None):
        for reader in self._readers:
            reader._reset(context)
            reader._skip = 0
        self._skip = 0

    def _next_chunk(self, ctx):
        if self._itraj > 0:
            self._reset(ctx)
        skip = self._skip if self._t == 0 else 0
        if self._chunksize == 0:
            if not ctx.uniform_stride:
                raise ValueError("fragmented trajectory implemented for random access")
            else:
                X = None
                overlap = skip
                for idx, r in enumerate(self._readers):
                    r._skip = overlap
                    out = r.get_output(stride=ctx.stride)[0]
                    X = np.vstack((X, out)) if X is not None else out
                    # if stride doesn't divide length, one has to offset the next trajectory
                    #overlap = ctx.stride - (((self._reader_lengths[idx] - overlap) % ctx.stride) + 1)
                    overlap = ctx.stride * ((self._reader_lengths[idx] - overlap - 1) // ctx.stride + 1) - self._reader_lengths[idx] - overlap
                self._itraj += 1
                return X

#    def trajectory_lengths(self, stride=1):
#        return np.array([(self._lengths[0] - 1) // stride + 1 - self._skip], dtype=int)

    def parametrize(self, stride=1):
        for reader in self._readers:
            reader.parametrize(stride)
        self._parametrized = True

    def _index_to_reader_index(self, index):
        """
        Accepts an index parameter in [0, sum(reader_lenghts)) and returns a tuple (reader_index, local_index),
        where the tuple (reader_index, local_index) corresponds to the global frame index of the fragmented trajectory.
        :param index: the global index
        :return: a tuple (reader_index, local_index)
        """
        prevLen = 0
        for readerIndex, len in enumerate(self._cumulative_lengths):
            if prevLen <= index < len:
                return (readerIndex, index - prevLen)
            prevLen = len
        raise ValueError("Requested index %s was out of bounds [0,%s)" % (index, self._cumulative_lengths[-1]))