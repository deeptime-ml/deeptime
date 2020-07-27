
# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# MSMTools is free software: you can redistribute it and/or modify
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
Created on 07.10.2013

@author: marscher
'''
import unittest
import numpy as np
from msmtools.util.birth_death_chain import BirthDeathChain

from msmtools.analysis.dense import assessment


class RateMatrixTest(unittest.TestCase):

    def setUp(self):
        self.A = np.array(
            [[-3, 3, 0, 0],
             [3, -5, 2, 0],
             [0, 3, -5, 2],
             [0, 0, 3, -3]]
        )

    def testIsRateMatrix(self):
        self.assertTrue(assessment.is_rate_matrix(self.A), 'A should be a rate matrix')

        # manipulate matrix so it isn't a rate matrix any more
        self.A[0][0] = 3
        self.assertFalse(assessment.is_rate_matrix(self.A), 'matrix is not a rate matrix')


class ReversibleTest(unittest.TestCase):

    def setUp(self):
        p = np.zeros(10)
        q = np.zeros(10)
        p[0:-1] = 0.5
        q[1:] = 0.5
        p[4] = 0.01
        q[6] = 0.1

        self.bdc = BirthDeathChain(q, p)
        self.T = self.bdc.transition_matrix()
        self.mu = self.bdc.stationary_distribution()

    def testIsReversible(self):
        # create a reversible matrix
        self.assertTrue(assessment.is_reversible(self.T, self.mu), "T should be reversible")


if __name__ == "__main__":
    unittest.main()
