from typing import Tuple

import numpy as np


class DrunkardsWalk(object):
    r""" This example dataset simulates the steps a drunkard living in a two-dimensional plane takes finding
    either the bar or the home as two absorbing states.

    The drunkard can take steps up/down/left/right with uniform probability (as possible, in the corners the only
    possibilities are the ones that do not lead out of the grid). The transition matrix
    :math:`P\in\mathbb{R}^{nm\times nm}  possesses one absorbing state for home and bar, respectively,
    and uniform two-dimensional jump probabilities in between. The grid is of size :math:`n\times m` and a point
    :math:`(i,j)` is identified with state :math:`i+nj` in the transition matrix.
    """

    def __init__(self, grid_size: Tuple[int, int], bar_location: Tuple[int, int], home_location: Tuple[int, int]):
        self.n_states = grid_size[0] * grid_size[1]  # the bar and home plus the intermediate steps
        self.grid_size = grid_size
        self.bar_location = bar_location
        self.bar_state = self._coord_to_ix(self.bar_location)
        self.home_location = home_location
        self.home_state = self._coord_to_ix(self.home_location)

        transition_matrix = np.zeros((self.n_states, self.n_states), dtype=np.float32)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                coord = (i, j)
                state = self._coord_to_ix(coord)  # row in the transition matrix
                if state == self.home_state or state == self.bar_state:
                    transition_matrix[state, state] = 1.
                else:
                    pass

    def _coord_to_ix(self, coord: Tuple[int, int]):
        return coord[0] + self.grid_size[0]*coord[1]
