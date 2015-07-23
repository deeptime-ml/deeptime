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
import numpy as np
import mdtraj

from pyemma.coordinates.util import patches
from pyemma.coordinates.data.interface import ReaderInterface
from pyemma.coordinates.data.featurizer import MDFeaturizer
from pyemma import config

__author__ = 'noe, marscher'
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

    >>> reader = FeatureReader('mytraj.xtc', 'my_structure.pdb') # doctest: +SKIP
    >>> chunks = [] # doctest: +SKIP
    >>> for itraj, X in reader: # doctest: +SKIP
    ...     chunks.append(X) # doctest: +SKIP

    """

    def __init__(self, trajectories, topologyfile=None, chunksize=100, featurizer=None):
        assert (topologyfile is not None) or (featurizer is not None), \
            "Needs either a topology file or a featurizer for instantiation"

        super(FeatureReader, self).__init__(chunksize=chunksize)

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

        self.__set_dimensions_and_lenghts()
        self._parametrized = True

    def __set_dimensions_and_lenghts(self):
        self._ntraj = len(self.trajfiles)
        # lookups pre-computed lengths, or compute it on the fly and store it in db.
        if config['use_trajectory_lengths_cache'] == 'True':
            from pyemma.coordinates.data.traj_info_cache import TrajectoryInfoCache
            for traj in self.trajfiles:
                self._lengths.append(TrajectoryInfoCache[traj])
        else:
            for traj in self.trajfiles:
                with mdtraj.open(traj, mode='r') as fh:
                    self._lengths.append(len(fh))

        # number of trajectories/data sets
        if self._ntraj == 0:
            raise ValueError("no valid data")

        # note: dimension is a custom impl in this class

    def describe(self):
        """
        Returns a description of this transformer

        :return:
        """
        return ["Feature reader with following features"] + self.featurizer.describe()

    def dimension(self):
        """
        Returns the number of output dimensions

        :return:
        """
        if len(self.featurizer.active_features) == 0:
            # special case: Cartesian coordinates
            return self.featurizer.topology.n_atoms * 3
        else:
            # general case
            return self.featurizer.dimension()

    def _create_iter(self, filename, skip=0, stride=1):
        return patches.iterload(filename, chunk=self.chunksize,
                                top=self.topfile, skip=skip, stride=stride)

    def _close(self):
        try:
            if self._mditer:
                self._mditer.close()
            if self._mditer2:
                self._mditer2.close()
        except:
            self._logger.exception("something went wrong closing file handles")

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
                                       % (self._itraj, lag))
                self._curr_lag = lag
                self._mditer2 = self._create_iter(self.trajfiles[self._itraj],
                                                  skip=self._curr_lag*stride, stride=stride) 
            try:
                adv_chunk = self._mditer2.next()
            except StopIteration:
                # When _mditer2 ran over the trajectory end, return empty chunks.
                adv_chunk = mdtraj.Trajectory(np.empty((0, shape[1], shape[2]), np.float32), chunk.topology)

        self._t += shape[0]

        if (self._t >= self.trajectory_length(self._itraj, stride=stride) and
                self._itraj < len(self.trajfiles) - 1):
            if __debug__:
                self._logger.debug('closing current trajectory "%s"'
                                   % self.trajfiles[self._itraj])
            self._close()

            self._t = 0
            self._itraj += 1
            self._mditer = self._create_iter(self.trajfiles[self._itraj], stride=stride)
            # we open self._mditer2 only if requested due lag parameter!
            self._curr_lag = 0

        if (self._t >= self.trajectory_length(self._itraj, stride=stride) and
                self._itraj == len(self.trajfiles) - 1):
            if __debug__:
                self._logger.debug('closing last trajectory "%s"'
                                   % self.trajfiles[self._itraj])
            self._mditer.close()
            if self._curr_lag != 0:
                self._mditer2.close()

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

    def parametrize(self, stride=1):
        if self.in_memory:
            self._map_to_memory(stride)
