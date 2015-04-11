__author__ = 'noe'

import numpy as np
import mdtraj

from pyemma.coordinates.util import patches
from pyemma.coordinates.data.interface import ReaderInterface
from pyemma.coordinates.data.featurizer import MDFeaturizer

__all__ = ['FeatureReader']


class FeatureReader(ReaderInterface):

    """
    Reads features from MD data.

    To select a feature, access the :attr:`featurizer` and call a feature
    selecting method (e.g) distances.

    Parameters
    ----------
    trajectories: list of strings
        paths to trajectory files

    topologyfile: string
        path to topology file (e.g. pdb)

    Examples
    --------

    Iterator access:

    >>> reader = FeatureReader('mytraj.xtc', 'my_structure.pdb')
    >>> chunks = []
    >>> for itraj, X in reader:
    >>>     chunks.append(X)


    Extract backbone torsion angles of protein during feature reading:

    >>> reader = FeatureReader('mytraj.xtc', 'my_structure.pdb')
    >>> reader.featurizer.add_backbone_torsions()
    >>> chunks = []
    >>> for _, X in reader:
    ...     chunks.append(X)

    """

    def __init__(self, trajectories, topologyfile=None, chunksize=100, featurizer=None):
        assert (topologyfile is not None) or (featurizer is not None), \
            "Needs either a topology file or a featurizer for instantiation"
        # init with chunksize 100
        super(FeatureReader, self).__init__(chunksize=chunksize)
        self.data_producer = self

        # files
        if isinstance(trajectories, basestring):
            trajectories = [trajectories]
        self.trajfiles = trajectories
        self.topfile = topologyfile

        # featurizer
        if topologyfile and featurizer:
            self._logger.warning("Both a topology file and a featurizer were given as arguments. "
                                 "Only featurizer gets respected in this case.")
        if not featurizer:
            self.featurizer = MDFeaturizer(topologyfile)
        else:
            self.featurizer = featurizer
            self.topfile = featurizer.topologyfile

        # iteration
        self._mditer = None
        # current lag time
        self._curr_lag = 0
        # time lagged iterator
        self._mditer2 = None

        # cache size
        self.in_memory = False
        self.Y = None

        self.__set_dimensions_and_lenghts()
        self._parametrized = True

    # @classmethod
    # def init_from_featurizer(cls, trajectories, featurizer):
    #     if not isinstance(featurizer, MDFeaturizer):
    #         raise ValueError("given featurizer is not of type Featurizer, but is %s"
    #                          % type(featurizer))
    #     cls.featurizer = featurizer
    #     return cls(trajectories, featurizer.topologyfile)

    def __set_dimensions_and_lenghts(self):
        self._ntraj = len(self.trajfiles)
        # basic statistics
        for traj in self.trajfiles:
            sum_frames = sum(t.n_frames for t in self._create_iter(traj))
            self._lengths.append(sum_frames)

        # number of trajectories/data sets
        if self._ntraj == 0:
            raise ValueError("no valid data")

        # note: dimension is a custom impl in this class

    def describe(self):
        """
        Returns a description of this transformer

        :return:
        """
        return "Feature reader, features = ", self.featurizer.describe()

    def parametrize(self, stride=1):
        """
        Parametrizes this transformer

        :return:
        """
        if self.in_memory:
            self._map_to_memory(stride=stride)

    def dimension(self):
        """
        Returns the number of output dimensions

        :return:
        """
        if len(self.featurizer.active_features) == 0:
            # special case: cartesion coordinates
            return self.featurizer.topology.n_atoms * 3
        else:
            # general case
            return self.featurizer.dimension()

    def _get_memory_per_frame(self):
        """
        Returns the memory requirements per frame, in bytes

        :return:
        """
        return 4 * self.dimension()

    def _get_constant_memory(self):
        """
        Returns the constant memory requirements, in bytes

        :return:
        """
        return 0

    def _map_to_memory(self, stride=1):
        #TODO: stride is currently not implemented
        if stride > 1: 
            raise NotImplementedError('stride option for FeatureReader._map_to_memory is currently not implemented')

        self._reset()
        # iterate over trajectories
        last_chunk = False
        itraj = 0
        while not last_chunk:
            last_chunk_in_traj = False
            t = 0
            while not last_chunk_in_traj:
                y = self._next_chunk()
                assert y is not None
                L = np.shape(y)[0]
                # last chunk in traj?
                last_chunk_in_traj = (t + L >= self.trajectory_length(itraj))
                # last chunk?
                last_chunk = (
                    last_chunk_in_traj and itraj >= self.number_of_trajectories() - 1)
                # write
                self.Y[itraj][t:t + L] = y
                # increment time
                t += L
            # increment trajectory
            itraj += 1

    def _create_iter(self, filename, skip=0, stride=1):
        return patches.iterload(filename, chunk=self.chunksize,
                                top=self.topfile, skip=skip, stride=stride)

    def _reset(self, stride=1):
        """
        resets the chunk reader
        """
        self._itraj = 0
        self._curr_lag = 0
        if len(self.trajfiles) >= 1:
            self._t = 0
            self._mditer = self._create_iter(self.trajfiles[0], stride=stride)

    def _next_chunk(self, lag=0, stride=1):
        """
        gets the next chunk. If lag > 0, we open another iterator with same chunk
        size and advance it by one, as soon as this method is called with a lag > 0.

        :return: a feature mapped vector X, or (X, Y) if lag > 0
        """
        chunk = self._mditer.next()
        shape = chunk.xyz.shape

        if lag > 0:
            if self._curr_lag == 0:
                # lag time or trajectory index changed, so open lagged iterator
                if __debug__:
                    self._logger.debug("open time lagged iterator for traj %i with lag %i"
                                       % (self._itraj, self._curr_lag))
                self._curr_lag = lag
                self._mditer2 = self._create_iter(self.trajfiles[self._itraj],
                                                  skip=self._curr_lag*stride, stride=stride) 
            try:
                adv_chunk = self._mditer2.next()
            except StopIteration:
                # When _mditer2 ran over the trajectory end, return empty chunks.
                adv_chunk = mdtraj.Trajectory(
                              np.empty((0, shape[1], shape[2]), np.float32),
                              chunk.topology)

        self._t += shape[0]

        if (self._t >= self.trajectory_length(self._itraj, stride=stride) and
                self._itraj < len(self.trajfiles) - 1):
            if __debug__:
                self._logger.debug('closing current trajectory "%s"'
                                   % self.trajfiles[self._itraj])
            self._mditer.close()
            self._t = 0
            self._itraj += 1
            self._mditer = self._create_iter(self.trajfiles[self._itraj], stride=stride)
            # we open self._mditer2 only if requested due lag parameter!
            self._curr_lag = 0

        # map data
        if lag == 0:
            if len(self.featurizer.active_features) == 0:
                shape_2d = (shape[0], shape[1] * shape[2])
                return chunk.xyz.reshape(shape_2d)
            else:
                return self.featurizer.map(chunk)
        else:
            if len(self.featurizer.active_features) == 0:
                shape_Y = adv_chunk.xyz.shape

                X = chunk.xyz.reshape((shape[0], shape[1] * shape[2]))
                Y = adv_chunk.xyz.reshape((shape_Y[0], shape_Y[1] * shape_Y[2]))
            else:
                X = self.featurizer.map(chunk)
                Y = self.featurizer.map(adv_chunk)
            return X, Y
