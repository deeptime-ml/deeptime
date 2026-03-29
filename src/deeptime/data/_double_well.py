import functools
from typing import List

import numpy as _np

__author__ = 'noe, marscher, clonker'


@functools.lru_cache(maxsize=1)
def _load_double_well_discrete():
    import os
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'double_well_discrete.npz')
    with _np.load(filename) as datafile:
        dtraj = datafile['dtraj']
        transition_matrix = datafile['P']
    # avoid side effects, since we are caching these arrays!
    dtraj.flags.writeable = False
    transition_matrix.flags.writeable = False

    from deeptime.markov.msm import MarkovStateModel
    msm = MarkovStateModel(transition_matrix)
    return dtraj, msm


class DoubleWellDiscrete:
    r""" MCMC process in a symmetric double well potential, spatially discretized to 100 bins.

    Encapsulates discrete trajectories and
    markov state model (see :class:`MarkovStateModel <deeptime.markov.msm.MarkovStateModel>`) with exact
    transition matrix.
    """

    def __init__(self):
        dtraj, msm = _load_double_well_discrete()
        self._dtraj = dtraj
        self._analytic_msm = msm

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
        r""" 100K frames trajectory at timestep 10, arbitrary n-state discretization.

        Parameters
        ----------
        divides : (n, dtype=int) ndarray
            The state boundaries.

        Returns
        -------
        dtraj : (T,) ndarray
            Discrete trajectory with :code:`len(divides)` states.
        """
        disc = _np.zeros(100, dtype=int)
        divides = _np.concatenate([divides, [100]])
        for i in range(len(divides) - 1):
            disc[divides[i]:divides[i + 1]] = i + 1
        return disc[self.dtraj]

    @property
    def transition_matrix(self):
        """ Exact transition matrix used to generate the data """
        return self.analytic_msm.transition_matrix

    @property
    def analytic_msm(self):
        """ Returns a :class:`MarkovStateModel <deeptime.markov.msm.MarkovStateModel>` instance with
        the exact transition matrix. """
        return self._analytic_msm

    def simulate_trajectory(self, n_steps, start=None, stop=None, dt=1) -> _np.ndarray:
        """ Generates a discrete trajectory of length less or equal n_steps.

        Parameters
        ----------
        n_steps : int
            maximum number of steps to simulate
        start : int, optional, default=None
            Starting state. If None is given, it is sampled from the stationary distribution.
        stop : int, optional, default=None
            Stopping state. If not None and encountered, stops the simulation. This can lead to fewer
            than n_steps steps.
        dt : int, optional, default=1
            Time step to apply when simulating the trajectory.

        Returns
        -------
        dtraj : (T, 1) ndarray
            A discrete trajectory.
        """
        return self.analytic_msm.simulate(n_steps, start=start, stop=stop, dt=dt)

    def simulate_trajectories(self, n_trajectories: int, n_steps: int,
                              start=None, stop=None, dt=1) -> List[_np.ndarray]:
        """
        Simulates :code:`n_trajectories` discrete trajectories. For a more detailed description of the arguments, see
        :meth:`simulate_trajectory`.
        """
        return [self.simulate_trajectory(n_steps, start=start, stop=stop, dt=dt) for _ in range(n_trajectories)]
