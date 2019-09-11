import numpy as np


def load_2well_discrete():
    class DoubleWell_Discrete_Data(object):
        """ MCMC process in a symmetric double well potential, spatially discretized to 100 bins """

        def __init__(self):
            from pkg_resources import resource_filename
            from sktime.markovprocess import MarkovStateModel

            filename = resource_filename(__name__, 'data/double_well_discrete.npz')
            with np.load(filename) as datafile:
                self._dtraj_T100K_dt10 = datafile['dtraj']
                P = datafile['P']
            self._msm = MarkovStateModel(transition_matrix=P)

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
            for i in range(len(divides) - 1):
                disc[divides[i]:divides[i + 1]] = i + 1
            return disc[self.dtraj_T100K_dt10]

        @property
        def msm(self):
            """ Returns an MSM object with the exact transition matrix """
            return self._msm

    return DoubleWell_Discrete_Data()
