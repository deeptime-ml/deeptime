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


from __future__ import absolute_import

import mdtraj
import six

from pyemma import config
from pyemma.coordinates.data.datasource import DataSourceIterator, DataSource
from pyemma.coordinates.data.featurizer import MDFeaturizer
from pyemma.coordinates.util import patches

__author__ = 'noe, marscher'
__all__ = ['FeatureReader']


class FeatureReader(DataSource):
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
    >>> from pyemma.datasets import get_bpti_test_data

    Iterator access:

    >>> reader = FeatureReader(get_bpti_test_data()['trajs'], get_bpti_test_data()['top'])

    Optionally set a chunksize

    >>> reader.chunksize = 300

    Store chunks by their trajectory index

    >>> chunks = {i : [] for i in range(reader.number_of_trajectories())}
    >>> for itraj, X in reader:
    ...     chunks[itraj].append(X)


    Calculate some distances of protein during feature reading:

    >>> reader.featurizer.add_distances([[0, 3], [10, 15]])
    >>> X = reader.get_output()

    """

    def __init__(self, trajectories, topologyfile=None, chunksize=100, featurizer=None):
        assert (topologyfile is not None) or (featurizer is not None), \
            "Needs either a topology file or a featurizer for instantiation"

        super(FeatureReader, self).__init__(chunksize=chunksize)
        self._is_reader = True

        # files
        if isinstance(trajectories, six.string_types):
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

        # Check that the topology and the files in the filelist can actually work together
        self._assert_toptraj_consistency()

        self.__set_dimensions_and_lengths()

    def __set_dimensions_and_lengths(self):
        self._ntraj = len(self.trajfiles)

        # workaround NotImplementedError __len__ for xyz files
        # Github issue: markovmodel/pyemma#621
        if six.PY2:
            from mock import patch
        else:
            from unittest.mock import patch
        from mdtraj.formats import XYZTrajectoryFile
        def _make_len_func(top):
            def _len_xyz(self):
                assert isinstance(self, XYZTrajectoryFile)
                assert hasattr(self, '_filename'), "structual change in xyzfile class!"
                import warnings
                from pyemma.util.exceptions import EfficiencyWarning
                warnings.warn("reading all of your data,"
                              " just to determine number of frames." +
                              " Happens only once, because this is cached."
                              if config['use_trajectory_lengths_cache'] else "",
                              EfficiencyWarning)
                # obtain len by reading whole file!
                mditer = mdtraj.iterload(self._filename, top=top)
                return sum(t.n_frames for t in mditer)

            return _len_xyz

        f = _make_len_func(self.topfile)

        # lookups pre-computed lengths, or compute it on the fly and store it in db.
        with patch.object(XYZTrajectoryFile, '__len__', f):
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

    def _create_iterator(self, skip=0, chunk=0, stride=1, return_trajindex=True):
        return FeatureReaderIterator(self, skip=skip, chunk=chunk, stride=stride,
                                     return_trajindex=return_trajindex)

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

    def _assert_toptraj_consistency(self):
        r""" Check if the topology and the trajfiles of the reader have the same n_atoms"""
        traj = mdtraj.load_frame(self.trajfiles[0], index=0, top=self.topfile)
        desired_n_atoms = self.featurizer.topology.n_atoms
        assert traj.xyz.shape[1] == desired_n_atoms, "Mismatch in the number of atoms between the topology" \
                                                     " and the first trajectory file, %u vs %u" % \
                                                     (desired_n_atoms, traj.xyz.shape[1])


class FeatureReaderIterator(DataSourceIterator):

    def __init__(self, data_source, skip=0, chunk=0, stride=1, return_trajindex=False):
        super(FeatureReaderIterator, self).__init__(
                data_source, skip=skip, chunk=chunk, stride=stride, return_trajindex=return_trajindex
        )
        self._create_mditer()

    def close(self):
        if self._mditer is not None:
            self._mditer.close()
        raise StopIteration()

    def _next_chunk(self):
        """
        gets the next chunk. If lag > 0, we open another iterator with same chunk
        size and advance it by one, as soon as this method is called with a lag > 0.

        :return: a feature mapped vector X, or (X, Y) if lag > 0
        """
        chunk = next(self._mditer)
        shape = chunk.xyz.shape

        self._t += shape[0]

        if self._t >= self.trajectory_length() and self._itraj < len(self._data_source.trajfiles) - 1:
            if __debug__:
                # self._logger.debug('closing current trajectory "%s"' % self.trajfiles[self._itraj])
                pass
            self._mditer.close()

            self._t = 0
            self._itraj += 1
            self._create_mditer()

        if not self.uniform_stride:
            traj_len = self.ra_trajectory_length(self._itraj)
        else:
            traj_len = self.trajectory_length()
        if self._t >= traj_len and self._itraj == len(self._data_source.trajfiles) - 1:
            if __debug__:
                # self._logger.debug('closing last trajectory "%s"' % self.trajfiles[self._itraj])
                pass
            self._mditer.close()

        # map data
        if len(self._data_source.featurizer.active_features) == 0:
            shape_2d = (shape[0], shape[1] * shape[2])
            return chunk.xyz.reshape(shape_2d)
        else:
            return self._data_source.featurizer.transform(chunk)

    def _create_mditer(self):
        if not self.uniform_stride:
            while self._itraj not in self.traj_keys and self._itraj < self.number_of_trajectories():
                self._itraj += 1
            self._mditer = self._create_patched_iter(
                    self._data_source.trajfiles[self._itraj], stride=self.ra_indices_for_traj(self._itraj)
            )
        else:
            self._mditer = self._create_patched_iter(
                    self._data_source.trajfiles[self._itraj], skip=self.skip, stride=self.stride
            )

    def _create_patched_iter(self, filename, skip=0, stride=1, atom_indices=None):
        return patches.iterload(filename, chunk=self.chunksize, top=self._data_source.topfile,
                                skip=skip, stride=stride, atom_indices=atom_indices)