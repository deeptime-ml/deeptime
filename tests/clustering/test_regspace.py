
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


'''
Created on 26.01.2015

@author: marscher
'''

from __future__ import absolute_import
import itertools
import unittest

from pyemma.coordinates.clustering.regspace import RegularSpaceClustering
from pyemma.coordinates.data.data_in_memory import DataInMemory
from pyemma.coordinates.api import cluster_regspace

import numpy as np
import pyemma.util.types as types


class RandomDataSource(DataInMemory):

    __serialize_version = 0

    def __init__(self, a=None, b=None, chunksize=100, n_samples=1000, dim=3):
        """
        creates random values in interval [a,b]
        """
        data = np.random.random((n_samples, dim))
        if a is not None and b is not None:
            data *= (b - a)
            data += a
        super(RandomDataSource, self).__init__(data, chunksize=chunksize)


class TestRegSpaceClustering(unittest.TestCase):

    def setUp(self):
        self.dmin = 0.3
        self.clustering = RegularSpaceClustering(dmin=self.dmin)
        self.src = RandomDataSource()

    def test_algorithm(self):
        self.clustering.estimate(self.src)

        # correct type of dtrajs
        assert types.is_int_vector(self.clustering.dtrajs[0])

        # assert distance for each centroid is at least dmin
        for c in itertools.combinations(self.clustering.clustercenters, 2):
            if np.allclose(c[0], c[1]):  # skip equal pairs
                continue

            dist = np.linalg.norm(c[0] - c[1], 2)

            self.assertGreaterEqual(dist, self.dmin,
                                    "centroid pair\n%s\n%s\n has smaller"
                                    " distance than dmin(%f): %f"
                                    % (c[0], c[1], self.dmin, dist))

    def test_assignment(self):
        self.clustering.estimate(self.src)

        assert len(self.clustering.clustercenters) > 1

        # num states == num _clustercenters?
        self.assertEqual(len(np.unique(self.clustering.dtrajs)),  len(
            self.clustering.clustercenters), "number of unique states in dtrajs"
            " should be equal.")

        data_to_cluster = np.random.random((1000, 3))

        self.clustering.assign(data_to_cluster, stride=1)

    def test_spread_data(self):
        src =  RandomDataSource(a=-2, b=2)
        self.clustering.dmin = 2
        self.clustering.estimate(src)

    def test1d_data(self):
        data = np.random.random(100)
        cluster_regspace(data, dmin=0.3)

    def test_non_existent_metric(self):
        src = RandomDataSource(a=-2, b=2)
        self.clustering.dmin = 2
        self.clustering.metric = "non_existent_metric"
        with self.assertRaises(ValueError):
            self.clustering.estimate(src)

    def test_minRMSD_metric(self):
        src = RandomDataSource(a=-2, b=2)
        self.clustering.dmin = 2
        self.clustering.metric = "minRMSD"
        self.clustering.estimate(src)

        data_to_cluster = np.random.random((1000, 3))

        self.clustering.assign(data_to_cluster, stride=1)

    def test_too_small_dmin_should_warn(self):
        self.clustering.dmin = 1e-8
        max_centers = 50
        self.clustering.max_centers = max_centers
        import warnings
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            self.clustering.estimate(self.src)
            assert w
            assert len(w) == 1

            assert len(self.clustering.clustercenters) == max_centers

            # assign data
            out = self.clustering.get_output()
            assert len(out) == self.clustering.number_of_trajectories()
            assert len(out[0]) == self.clustering.trajectory_lengths()[0]

    def test_regspace_nthreads(self):
        for metric in ('euclidean', 'minRMSD'):
            self.clustering.estimate(self.src, n_jobs=1, dmin=self.dmin, metric=metric)
            cl2 = cluster_regspace(self.src, n_jobs=2, dmin=self.dmin, metric=metric)
            np.testing.assert_equal(self.clustering.clustercenters, cl2.clustercenters)


if __name__ == "__main__":
    unittest.main()
