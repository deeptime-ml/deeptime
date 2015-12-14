from copy import copy
from itertools import chain

from pyemma.coordinates.api import source

import numpy as np
from pyemma.coordinates.data.datasource import DataSource


class _FragmentedTrajectoryIterator(object):
    def __init__(self, fragmented_reader, readers, chunksize, stride, skip):
        # global time variable
        self._t = 0
        # time variable on a per-reader basis
        self._reader_t = 0
        self._readers = readers
        self._chunksize = chunksize
        self._stride = stride
        self._skip = skip
        self._frag_reader = fragmented_reader
        # lengths array per reader
        self._reader_lengths = [reader.trajectory_length(0, 1) for reader in self._readers]
        # composite trajectory length
        self._lengths = [sum(self._reader_lengths)]
        # mapping reader_index -> cumulative length
        self._cumulative_lengths = np.cumsum(self._reader_lengths)
        # current reader index
        self._reader_at = 0
        self._done = False
        self._ctx = None

    def __iter__(self):
        return self

    @property
    def ctx(self):
        return self._ctx

    @ctx.setter
    def ctx(self, value):
        ctx_copy = copy(value)
        ctx_copy.lag = 0
        self._ctx = ctx_copy

    def __next__(self):
        skip = self._skip if self._t == 0 else 0
        if self._chunksize == 0:
            X = self._read_full(skip)
            return X
        else:
            # first chunk
            if self._reader_at == 0 and self._t == 0:
                self._reader_it = self._readers[self._reader_at].iterator(self._stride)
                self._reader_overlap = skip
            self._readers[self._reader_at]._skip = self._reader_overlap
            # set original chunksize
            self._readers[self._reader_at].chunksize = self._chunksize
            # chunk is contained in current reader
            if self._readers[self._reader_at].trajectory_length(0, self._stride, self._skip if
                    self._reader_at == 0 else 0) - self._reader_t - self._chunksize > 0:
                self._t += self._chunksize
                self._reader_t += self._chunksize
                X = self._readers[self._reader_at]._next_chunk(self.ctx)
                return X
            # chunk has to be collected from subsequent readers
            else:
                ndim = len(np.zeros(self._readers[0].dimension())[0::])
                expected_length = min(self._chunksize, sum(self._traj_lengths(self._stride)) - self._t)
                X = np.empty((expected_length, ndim), dtype=self._frag_reader.output_type())
                read = 0
                while read < expected_length:
                    # reader has data left:
                    if self._readers[self._reader_at].trajectory_length(0, self._stride, self._skip if
                            self._reader_at == 0 else 0) - self._reader_t > 0:
                        chunk = self._readers[self._reader_at]._next_chunk(self.ctx)
                        L = chunk.shape[0]
                        X[read:read + L, :] = chunk[:]
                        read += L
                        self._reader_t += L
                    # need new reader
                    if read < expected_length:
                        self._reader_at += 1
                        self._reader_t = 0
                        self._reader_it = self._readers[self._reader_at].iterator(self._stride)
                        self._reader_overlap = self._calculate_new_overlap(self._stride,
                                                                           self._reader_lengths[self._reader_at - 1],
                                                                           self._reader_overlap)
                        self._readers[self._reader_at]._skip = self._reader_overlap
                        if expected_length - read > 0:
                            self._readers[self._reader_at].chunksize = expected_length - read
                self._t += read
                return X

    def next(self):
        return self.__next__()

    def _read_full(self, skip):
        overlap = skip
        self._skip = overlap
        ndim = len(np.zeros(self._readers[0].dimension())[0::])
        length = sum(self._traj_lengths(self._stride))
        X = np.empty((length, ndim), dtype=self._frag_reader.output_type())
        for idx, r in enumerate(self._readers):
            r._skip = overlap
            # if stride doesn't divide length, one has to offset the next trajectory
            overlap = self._calculate_new_overlap(self._stride, self._reader_lengths[idx], overlap)
            r.chunksize = min(length, r.trajectory_length(0, self._stride))
            for itraj, data in r.iterator(stride=self._stride):
                L = data.shape[0]
                if L > 0:
                    X[self._t:self._t + L, :] = data[:]
                self._t += L
        return X

    def _traj_lengths(self, stride):
        return np.array([(l - (self._skip) - 1) // stride + 1 for idx, l in enumerate(self._lengths)], dtype=int)

    @staticmethod
    def _calculate_new_overlap(stride, traj_len, skip):
        """
        Given two trajectories T_1 and T_2, this function calculates for the first trajectory an overlap, i.e.,
        a skip parameter for T_2 such that the trajectory fragments T_1 and T_2 appear as one under the given stride.

        Idea for deriving the formula: It is

        K = ((traj_len - skip - 1) // stride + 1) = #(data points in trajectory of length (traj_len - skip)).

        Therefore, the first point's position that is not contained in T_1 anymore is given by

        pos = skip + s * K.

        Thus the needed skip of T_2 such that the same stride parameter makes T_1 and T_2 "look as one" is

        overlap = pos - traj_len.

        :param stride: the (global) stride parameter
        :param traj_len: length of T_1
        :param skip: skip of T_1
        :return: skip of T_2
        """
        overlap = stride * ((traj_len - skip - 1) // stride + 1) - traj_len + skip
        return overlap


class FragmentedTrajectoryReader(DataSource):
    def __init__(self, trajectories, topologyfile=None, chunksize=100, featurizer=None):
        # sanity checks
        assert isinstance(trajectories, (list, tuple)), "input trajectories should be of list or tuple type"
        # if it contains no further list: treat as single trajectory
        if not any([isinstance(traj, (list, tuple)) for traj in trajectories]):
            trajectories = [trajectories]
        # if not list of lists, treat as single-element-fragment-trajectory
        trajectories = [traj if isinstance(traj, (list, tuple)) else [traj] for traj in trajectories]
        # some trajectory should be provided
        assert len(trajectories) > 0, "no input trajectories provided"
        # call super
        super(FragmentedTrajectoryReader, self).__init__(chunksize=chunksize)
        # number of trajectories
        self._ntraj = len(trajectories)
        # store readers
        self._readers = [[source(input_item, features=featurizer, top=topologyfile, chunk_size=chunksize)
                          for input_item in trajectories[itraj]] for itraj in range(0, self._ntraj)]
        # store lagged (lazy) readers
        self._readers_lagged = [[source(input_item, features=featurizer, top=topologyfile, chunk_size=chunksize)
                                 for input_item in trajectories[itraj]] for itraj in range(0, self._ntraj)]
        # lengths array per reader
        self._reader_lengths = [[reader.trajectory_length(0, 1)
                                 for reader in self._readers[itraj]] for itraj in range(0, self._ntraj)]
        # composite trajectory length
        self._lengths = [sum(self._reader_lengths[itraj]) for itraj in range(0, self._ntraj)]
        # mapping reader_index -> cumulative length
        self._cumulative_lengths = [np.cumsum(self._reader_lengths[itraj]) for itraj in range(0, self._ntraj)]
        # store trajectory files
        self._trajectories = trajectories
        # reader iterator
        self._it = None
        # lagged reader iterator
        self._it_lagged = None

    def describe(self):
        return "[FragmentedTrajectoryReader files=%s]" % self._trajectories

    def dimension(self):
        return self._readers[0][0].dimension()

    def _close(self):
        for reader in chain(self._readers, self._readers_lagged):
            reader._close()

    def _reset(self, context=None):
        super(FragmentedTrajectoryReader, self)._reset(context)
        for itraj in range(0, self._ntraj):
            for reader in chain(self._readers[itraj], self._readers_lagged[itraj]):
                reader._reset(context)
                reader._skip = 0
        self._skip = 0
        self._itraj = 0
        self._t = 0
        self._it = None
        self._it_lagged = None

    def _next_chunk(self, ctx):
        if self._itraj > self._ntraj:
            self._reset(ctx)
        if not self._it:
            self._it = _FragmentedTrajectoryIterator(self, self._readers[self._itraj], self.chunksize, ctx.stride, self._skip)
        if ctx.lag > 0 and not self._it_lagged:
            self._it_lagged = _FragmentedTrajectoryIterator(self, self._readers_lagged[self._itraj], self.chunksize,
                                                            ctx.stride, self._skip + ctx.lag)
        if not ctx.uniform_stride:
            raise ValueError("fragmented trajectory implemented for random access")
        else:
            self._it.ctx = ctx
            X = next(self._it, None)
            self._t += X.shape[0]
            remove_lagged_iterator = False
            if self._t >= self.trajectory_length(self._itraj, stride=ctx.stride):
                self._itraj += 1
                self._it = None
                self._t = 0
                remove_lagged_iterator = True
            if ctx.lag == 0:
                return X
            else:
                self._it_lagged.ctx = ctx
                Y = next(self._it_lagged, None)
                if remove_lagged_iterator:
                    self._it_lagged = None
                return X, Y

    def parametrize(self, stride=1):
        for itraj in range(0, self._ntraj):
            for reader in self._readers[itraj]:
                reader.parametrize(stride)
        self._parametrized = True

    def _index_to_reader_index(self, index, itraj):
        """
        Accepts an index parameter in [0, sum(reader_lenghts)) and returns a tuple (reader_index, local_index),
        where the tuple (reader_index, local_index) corresponds to the global frame index of the fragmented trajectory.
        :param index: the global index
        :return: a tuple (reader_index, local_index)
        """
        prev_len = 0
        for readerIndex, length in enumerate(self._cumulative_lengths[itraj]):
            if prev_len <= index < length:
                return readerIndex, index - prev_len
            prev_len = length
        raise ValueError("Requested index %s was out of bounds [0,%s)" % (index, self._cumulative_lengths[itraj][-1]))
