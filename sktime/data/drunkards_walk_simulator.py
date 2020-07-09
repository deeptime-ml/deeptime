from typing import Tuple

import numpy as np


class DrunkardsWalk(object):
    r""" This example dataset simulates the steps a drunkard living in a two-dimensional plane takes finding
    either the bar or the home as two absorbing states.

    The drunkard can take steps in a 3x3 stencil with uniform probability (as possible, in the corners the only
    possibilities are the ones that do not lead out of the grid). The transition matrix
    :math:`P\in\mathbb{R}^{nm\times nm}`  possesses one absorbing state for home and bar, respectively,
    and uniform two-dimensional jump probabilities in between. The grid is of size :math:`n\times m` and a point
    :math:`(i,j)` is identified with state :math:`i+nj` in the transition matrix.
    """

    def __init__(self, grid_size: Tuple[int, int], bar_location: Tuple[int, int], home_location: Tuple[int, int]):
        r""" Creates a new drunkard's walk instance on a two-dimensional grid with predefined bar and home locations.

        Parameters
        ----------
        grid_size : tuple
            The grid size, must be tuple of length two.
        bar_location : tuple
            The bar location, must be valid coordinate and tuple of length two.
        home_location : tuple
            The home location, must be valid coordinate and tuple of length two.
        """
        self.n_states = grid_size[0] * grid_size[1]
        self.grid_size = grid_size
        self.bar_location = bar_location
        self.bar_state = self.coordinate_to_state(self.bar_location)
        self.home_location = home_location
        self.home_state = self.coordinate_to_state(self.home_location)

        transition_matrix = np.zeros((self.n_states, self.n_states), dtype=np.float32)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                coord = (i, j)
                state = self.coordinate_to_state(coord)  # row in the transition matrix
                if state == self.home_state or state == self.bar_state:
                    transition_matrix[state, state] = 1.
                else:
                    next_steps = []
                    for offset_i in [-1, 0, 1]:
                        for offset_j in [-1, 0, 1]:
                            if offset_i != 0 or offset_j != 0:
                                next_step = (coord[0] + offset_i, coord[1] + offset_j)
                                if self.is_valid_coordinate(next_step):
                                    next_steps.append(next_step)
                    #  uniform probability
                    p = 1. / len(next_steps)
                    for step in next_steps:
                        transition_matrix[state, self.coordinate_to_state(step)] = p

        from sktime.markov.msm import MarkovStateModel
        self._msm = MarkovStateModel(transition_matrix)

    def coordinate_to_state(self, coord: Tuple[int, int]) -> int:
        r""" Transforms a two-dimensional grid point (i, j) to a one-dimensional state.

        Parameters
        ----------
        coord : (i, j) tuple
            The grid point.

        Returns
        -------
        state : int
            The state corresponding to the grid point.
        """
        return coord[0] + self.grid_size[0] * coord[1]

    def state_to_coordinate(self, state: int) -> Tuple[int, int]:
        r""" Inverse operation to :meth:`coordinate_to_state`. Transforms state to corresponding coordinate (i,j).

        Parameters
        ----------
        state : int
            The state.

        Returns
        -------
        coordinate : (i, j) tuple
            The corresponding coordinate.
        """
        coord_j = np.floor(state / (self.grid_size[0]))
        coord_i = state - coord_j * self.grid_size[0]
        return coord_i, coord_j

    def is_valid_coordinate(self, coord: Tuple[int, int]) -> bool:
        r""" Validates if a coordinate is within bounds.

        Parameters
        ----------
        coord : (i, j) tuple
            The coordinate.

        Returns
        -------
        is_valid : bool
            Whether the coordinate is within bounds.
        """
        return (0 <= coord[0] < self.grid_size[0]) and (0 <= coord[1] < self.grid_size[1])

    @property
    def msm(self):
        r""" Yields a :class:`MSM <sktime.markov.msm.MarkovStateModel>` which is parameterized with a transition matrix
        corresponding to this setup.

        Returns
        -------
        msm : sktime.markov.msm.MarkovStateModel
            The corresponding Markov state model.
        """
        return self._msm

    def walk(self, start: Tuple[int, int], n_steps: int, seed: int = -1):
        r""" Simulates a random walk on the grid.

        Parameters
        ----------
        start : (i, j) tuple
            Start coordinate on the grid.
        n_steps : int
            Number of steps to simulate.
        seed : int, default=-1
            Random seed.

        Returns
        -------
        random_walk : (n_steps, 2) ndarray
            A random walk in coordinate space.
        """
        assert self.is_valid_coordinate(start), "Start must be within bounds."
        states = self.msm.simulate(n_steps, start=self.coordinate_to_state(start), seed=seed)
        return np.array([self.state_to_coordinate(state) for state in states])
