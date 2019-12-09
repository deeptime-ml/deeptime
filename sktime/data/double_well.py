import functools
from typing import List

import numpy as _np
from sktime.markovprocess import MarkovStateModel as _MarkovStateModel

__author__ = 'noe, marscher, clonker'


@functools.lru_cache(maxsize=1)
def _load_double_well_discrete():
    import os
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'double_well_discrete.npz')
    with _np.load(filename) as datafile:
        dtraj = datafile['dtraj']
        transition_matrix = datafile['P']
    msm = _MarkovStateModel(transition_matrix)
    return dtraj, msm


class DoubleWellDiscrete(object):
    """ MCMC process in a symmetric double well potential, spatially discretized to 100 bins """

    def __init__(self):
        dtraj, msm = _load_double_well_discrete()
        self._dtraj = dtraj
        self._dtraj.flags.writeable = False
        self._msm = msm

    @property
    def dtraj(self):
        """ 100K frames trajectory at timestep 10, 100 microstates (not all are populated). """
        return self._dtraj

    @property
    def dtraj_n2good(self):
        """ 100K frames trajectory at timestep 10, good 2-state discretization (at transition state). """
        return self.dtraj_n([50])

    @property
    def dtraj_n2bad(self):
        """ 100K frames trajectory at timestep 10, bad 2-state discretization (off transition state). """
        return self.dtraj_n([40])

    def dtraj_n2(self, divide):
        """ 100K frames trajectory at timestep 10, arbitrary 2-state discretization. """
        return self.dtraj_n([divide])

    @property
    def dtraj_n6good(self):
        """ 100K frames trajectory at timestep 10, good 6-state discretization. """
        return self.dtraj_n([40, 45, 50, 55, 60])

    def dtraj_n(self, divides):
        """ 100K frames trajectory at timestep 10, arbitrary n-state discretization. """
        disc = _np.zeros(100, dtype=int)
        divides = _np.concatenate([divides, [100]])
        for i in range(len(divides) - 1):
            disc[divides[i]:divides[i + 1]] = i + 1
        return disc[self.dtraj]

    @property
    def transition_matrix(self):
        """ Exact transition matrix used to generate the data """
        return self.msm.transition_matrix

    @property
    def msm(self):
        """ Returns an MSM object with the exact transition matrix """
        return self._msm

    def simulate_trajectory(self, n_steps, start=None, stop=None, dt=1) -> _np.ndarray:
        """
        Generates a discrete trajectory of length less or equal n_steps.
        Parameters
        ----------
        n_steps: number of steps to simulate
        start: starting hidden state (optional)
        stop: stopping hidden set (optional), if not None this can lead to fewer than n_steps steps
        dt: time step
        Returns
        -------
        a discrete trajectory
        """
        return self.msm.simulate(n_steps, start=start, stop=stop, dt=dt)

    def simulate_trajectories(self, n_trajectories: int, n_steps: int,
                              start=None, stop=None, dt=1) -> List[_np.ndarray]:
        """
        Generates n_trajectories random trajectories of length n_steps each with time step dt
        Parameters
        ----------
        n_trajectories: number of trajectories
        n_steps: number of steps per trajectory
        start: starting hidden state, sampled from stationary distribution of hidden transition matrix if None
        stop: stopping hidden set, optional
        dt: discrete time step
        Returns
        -------
        discrete trajectories: a list of discrete trajectories
        """
        return [self.simulate_trajectory(n_steps, start=start, stop=stop, dt=dt) for _ in range(n_trajectories)]
