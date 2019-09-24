
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
import numpy as np

from sktime.markovprocess import MarkovStateModel

__author__ = 'noe, marscher'


class DoubleWell_Discrete_Data(object):
    """ MCMC process in a symmetric double well potential, spatially discretized to 100 bins """

    def __init__(self):
        import os
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'double_well_discrete.npz')
        with np.load(filename) as datafile:
            self._dtraj_T100K_dt10 = datafile['dtraj']
            self._P = datafile['P']
        self._msm = MarkovStateModel(self._P)

    @property
    def dtraj_T100K_dt10(self):
        """ 100K frames trajectory at timestep 10, 100 microstates (not all are populated). """
        return self._dtraj_T100K_dt10

    @property
    def dtraj_T100K_dt10_n2good(self):
        """ 100K frames trajectory at timestep 10, good 2-state discretization (at transition state). """
        return self.dtraj_T100K_dt10_n([50])

    @property
    def dtraj_T100K_dt10_n2bad(self):
        """ 100K frames trajectory at timestep 10, bad 2-state discretization (off transition state). """
        return self.dtraj_T100K_dt10_n([40])

    def dtraj_T100K_dt10_n2(self, divide):
        """ 100K frames trajectory at timestep 10, arbitrary 2-state discretization. """
        return self.dtraj_T100K_dt10_n([divide])

    @property
    def dtraj_T100K_dt10_n6good(self):
        """ 100K frames trajectory at timestep 10, good 6-state discretization. """
        return self.dtraj_T100K_dt10_n([40, 45, 50, 55, 60])

    def dtraj_T100K_dt10_n(self, divides):
        """ 100K frames trajectory at timestep 10, arbitrary n-state discretization. """
        disc = np.zeros(100, dtype=int)
        divides = np.concatenate([divides, [100]])
        for i in range(len(divides)-1):
            disc[divides[i]:divides[i+1]] = i+1
        return disc[self.dtraj_T100K_dt10]

    @property
    def transition_matrix(self):
        """ Exact transition matrix used to generate the data """
        return self.msm.transition_matrix

    @property
    def msm(self):
        """ Returns an MSM object with the exact transition matrix """
        return self._msm

    def generate_traj(self, N, start=None, stop=None, dt=1):
        """ Generates a random trajectory of length N with time step dt """
        return self.msm.generate_traj(N, start=start, stop=stop, dt=dt)

    def generate_trajs(self, M, N, start=None, stop=None, dt=1):
        """ Generates M random trajectories of length N each with time step dt """
        return self.msm.generate_trajs(M, N, start=start, stop=stop, dt=dt)
