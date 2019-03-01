
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
import unittest
from unittest import TestCase
import numpy as np
from pyemma.coordinates.api import cluster_mini_batch_kmeans


class TestMiniBatchKmeans(TestCase):
    def test_3gaussian_1d_singletraj(self):
        # generate 1D data from three gaussians
        X = [np.random.randn(200) - 2.0,
             np.random.randn(300),
             np.random.randn(400) + 2.0]
        X = np.hstack(X)
        kmeans = cluster_mini_batch_kmeans(X, batch_size=0.5, k=100, max_iter=10000)
        cc = kmeans.clustercenters
        assert (np.any(cc < 1.0))
        assert (np.any((cc > -1.0) * (cc < 1.0)))
        assert (np.any(cc > -1.0))

    def test_3gaussian_2d_multitraj(self):
        # generate 1D data from three gaussians
        X1 = np.zeros((200, 2))
        X1[:, 0] = np.random.randn(200) - 2.0
        X2 = np.zeros((300, 2))
        X2[:, 0] = np.random.randn(300)
        X3 = np.zeros((400, 2))
        X3[:, 0] = np.random.randn(400) + 2.0
        X = [X1, X2, X3]
        kmeans = cluster_mini_batch_kmeans(X, batch_size=0.5, k=100, max_iter=10000)
        cc = kmeans.clustercenters
        assert (np.any(cc < 1.0))
        assert (np.any((cc > -1.0) * (cc < 1.0)))
        assert (np.any(cc > -1.0))


class TestMiniBatchKmeansResume(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from pyemma.util.contexts import numpy_random_seed
        with numpy_random_seed(32):
            # three gaussians
            X = [np.random.randn(1000)-2.0,
                 np.random.randn(1000),
                 np.random.randn(1000)+2.0]
            cls.X = np.hstack(X)

    def test_resume(self):
        """ check that we can continue with the iteration by passing centers"""
        # centers are far off
        initial_centers = np.array([[1, 2, 3]]).T
        cl = cluster_mini_batch_kmeans(self.X, clustercenters=initial_centers,
                                       max_iter=1, k=3)

        resume_centers = cl.clustercenters
        cl.estimate(self.X, clustercenters=resume_centers, max_iter=50)
        new_centers = cl.clustercenters

        true = np.array([[-2, 0, 2]]).T
        d0 = true - resume_centers
        d1 = true - new_centers

        diff = np.linalg.norm(d0)
        diff_next = np.linalg.norm(d1)

        self.assertLess(diff_next, diff, 'resume_centers=%s, new_centers=%s' % (resume_centers, new_centers))

if __name__ == '__main__':
    unittest.main()
