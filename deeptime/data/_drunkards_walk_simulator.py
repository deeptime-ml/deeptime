import itertools
from typing import Tuple, List, Optional

import numpy as np

from deeptime.util.decorators import plotting_function

Coordinate = Tuple[int, int]


class DrunkardsWalk:
    r""" This example dataset simulates the steps a drunkard living in a two-dimensional plane takes finding
    either the bar or the home as two absorbing states.

    The drunkard can take steps in a 3x3 stencil with uniform probability (as possible, in the corners the only
    possibilities are the ones that do not lead out of the grid). The transition matrix
    :math:`P\in\mathbb{R}^{nm\times nm}`  possesses one absorbing state for home and bar, respectively,
    and uniform two-dimensional jump probabilities in between. The grid is of size :math:`n\times m` and a point
    :math:`(i,j)` is identified with state :math:`i+nj` in the transition matrix.

    Parameters
    ----------
    grid_size : tuple
        The grid size, must be tuple of length two.
    bar_location : tuple
        The bar location, must be valid coordinate and tuple of length two.
    home_location : tuple
        The home location, must be valid coordinate and tuple of length two.
    barriers : List of tuple of two integers or None, default=None
        Initial barrier locations. Can also be added post-hoc by calling :meth:`add_barrier`.
    """

    def __init__(self, grid_size: Tuple[int, int], bar_location: List[Coordinate], home_location: List[Coordinate],
                 barriers=None):
        if barriers is None:
            barriers = []
        self.n_states = grid_size[0] * grid_size[1]
        self.grid_size = grid_size
        self.bar_location = bar_location if isinstance(bar_location, (tuple, list, np.ndarray)) else [bar_location]
        self.bar_location = np.atleast_2d(self.bar_location)
        self.bar_state = [self.coordinate_to_state(state) for state in self.bar_location]
        self.home_location = home_location if isinstance(home_location, (tuple, list, np.ndarray)) else [home_location]
        self.home_location = np.atleast_2d(self.home_location)
        self.home_state = [self.coordinate_to_state(state) for state in self.home_location]
        self.barriers = barriers
        self.barrier_states = [self.coordinate_to_state(barrier) for barrier in self.barriers]
        self.barrier_weights = [None]*len(barriers)

        from deeptime.markov.msm import MarkovStateModel
        self._msm = MarkovStateModel(transition_matrix=np.eye(self.n_states, dtype=np.float64))
        self._update_transition_matrix()

    def _is_home_or_bar_state(self, state):
        return state in self.home_state or state in self.bar_state

    def _is_home_or_bar_coord(self, coord):
        return self._is_home_or_bar_state(self.coordinate_to_state(coord))

    def _update_transition_matrix(self) -> None:
        r"""Updates the MSM so that the state of the simulator is reflected in the transition matrix.
        """
        transition_matrix = self.msm.transition_matrix.copy()
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                coord = (i, j)
                state = self.coordinate_to_state(coord)  # row in the transition matrix
                next_steps = []
                for offset_i in [-1, 0, 1]:
                    for offset_j in [-1, 0, 1]:
                        if offset_i != 0 or offset_j != 0:
                            next_step = (coord[0] + offset_i, coord[1] + offset_j)
                            if self.is_valid_coordinate(next_step):
                                next_steps.append(next_step)
                #  uniform probability
                probabilities = []
                for next_step in next_steps:
                    next_state = self.coordinate_to_state(next_step)
                    if self.barriers is not None and next_step in self.barriers:
                        barrier_ix = self.barriers.index(next_step)
                        weight = self.barrier_weights[barrier_ix]
                        if weight is None:
                            probabilities.append(0.)
                        else:
                            probabilities.append(1./weight)
                    else:
                        if self._is_home_or_bar_state(state) and self._is_home_or_bar_state(next_state):
                            probabilities.append(100.)
                        else:
                            probabilities.append(1.)
                if self._is_home_or_bar_state(state) or \
                        (state in self.barrier_states and
                         self.barrier_weights[self.barrier_states.index(state)] is None):
                    # high probability to stay in home/bar
                    next_steps.append(coord)
                    probabilities.append(100.)
                else:
                    next_steps.append(coord)
                    probabilities.append(0.)
                probabilities = np.array(probabilities) / np.sum(probabilities)
                for p, step in zip(probabilities, next_steps):
                    transition_matrix[state, self.coordinate_to_state(step)] = p
        self.msm.update_transition_matrix(transition_matrix)

    def add_barrier(self, begin: Coordinate, end: Coordinate, weight: Optional[float] = None):
        r""" Adds a barrier to the grid by assigning probabilities

        .. math::
            P_{ij} = \mathbb{P}(X_{n+1} = j\in\mathrm{barriers} : X_n=i\text{ next to barrier}) = 0.

        The barrier is interpreted as a straight line between begin and end, discretized onto states using
        Bresenham's line algorithm :footcite:`bresenham1965algorithm`.

        Parameters
        ----------
        begin : tuple of two integers
            Begin coordinate of the barrier.
        end : tuple of two integers
            End coordinate of the barrier.
        weight : float or None, default=None
            If not None, the probability of jumping onto the barrier is set to :math:`1/w` before row-normalization.

        References
        ----------
        .. footbibliography::
        """
        barrier = []
        x0, y0 = begin
        x1, y1 = end

        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        while True:
            barrier.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2*err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
            if dy <= e2 <= dx:
                # introduced a 'diagonal' jump, append block so that diagonal is 'filled'
                barrier.append((x0, y0 - sy))

        self.barriers = list(itertools.chain(self.barriers, barrier))
        self.barrier_weights = list(itertools.chain(self.barrier_weights, [weight]*len(barrier)))
        self.barrier_states = [self.coordinate_to_state(barrier) for barrier in self.barriers]
        self._update_transition_matrix()

    def coordinate_to_state(self, coord: Coordinate) -> int:
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

    def state_to_coordinate(self, state: int) -> Coordinate:
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

    def is_valid_coordinate(self, coord: Coordinate) -> bool:
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
        r""" Yields a :class:`MSM <deeptime.markov.msm.MarkovStateModel>` which is parameterized with a transition matrix
        corresponding to this setup.

        Returns
        -------
        msm : deeptime.markov.msm.MarkovStateModel
            The corresponding Markov state model.
        """
        return self._msm

    def walk(self, start: Coordinate, n_steps: int, stop: bool = True, return_states: bool = False, seed: int = -1):
        r""" Simulates a random walk on the grid.

        Parameters
        ----------
        start : (i, j) tuple
            Start coordinate on the grid.
        n_steps : int
            Maximum number of steps to simulate.
        stop : bool, default=False
            Whether to stop the simulation once home or bar have been reached
        return_states : bool, default=False
            Whether to return states instead of coordinates. Default is False.
        seed : int, default=-1
            Random seed.

        Returns
        -------
        random_walk : (n_steps, 2) ndarray
            A random walk in coordinate space.
        """
        assert self.is_valid_coordinate(start), "Start must be within bounds."
        if stop:
            stopping_states = np.concatenate([self.home_state, self.bar_state])
        else:
            stopping_states = None
        states = self.msm.simulate(n_steps, start=self.coordinate_to_state(start), stop=stopping_states, seed=seed)
        if not return_states:
            return np.array([self.state_to_coordinate(state) for state in states])
        else:
            return states

    @staticmethod
    @plotting_function
    def plot_path(ax, path, intermediates: bool = True, color_lerp: bool = True, **plot_kw):  # pragma: no cover
        r""" Plots a path onto a drunkard's walk map. The path is interpolated with splines and, if desired, has a
        linearly interpolated color along its path.

        Parameters
        ----------
        ax : matplotlib axis
            The axis to plot onto.
        path : (N, 2) ndarray
            The path in terms of 2-dimensional coordinates.
        intermediates : bool, default=True
            Whether to scatterplot the states that were visited in the path.
        color_lerp : bool, default=True
            Whether to perform a linear interpolation along a colormap while the path progresses through the map.
        plot_kw
            Additional args that go into either a `ax.plot` call (`lerp=False`) or into a `LineCollection(...)` call
            (`lerp=True`).
        """
        import scipy.interpolate as interp
        from matplotlib.collections import LineCollection

        path = np.asarray(path)

        x = np.r_[path[:, 0]]
        y = np.r_[path[:, 1]]
        f, u = interp.splprep([x, y], s=0, per=False)
        xint, yint = interp.splev(np.linspace(0, 1, 50000), f)
        if intermediates:
            ax.scatter(x, y, label='Visited intermediates')

        if color_lerp:
            points = np.stack([xint, yint]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            coll = LineCollection(segments, cmap='cool', linestyle='dotted', **plot_kw)
            coll.set_array(np.linspace(0, 1, num=len(points), endpoint=True))
            coll.set_linewidth(2)
            ax.add_collection(coll)
        else:
            ax.plot(xint, yint, **plot_kw)

    @plotting_function
    def plot_network(self, ax, F, cmap=None, connection_threshold: float = 0.):  # pragma: no cover
        r""" Plots a flux network onto a matplotlib axis assuming that the states are ordered according to a iota
        range (i.e., {0, 1, ..., n_states-1}) on the grid.

        Parameters
        ----------
        ax : matplotlib axis
            The axis to plot onto.
        F : (n_states, n_states) ndarray
            The flux network, assigning to a state tuple `F[i,j]` a current.
        cmap : matplotlib colormap, default=None
            A colormap to use for the edges.
        connection_threshold : float, default=0
            A connection threshold in terms of minimal current to exceed to draw an edge.

        Returns
        -------
        edge_colors : list
            List of edge colors.
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        if cmap is None:
            cmap = plt.cm.jet

        G = nx.DiGraph()
        for state in range(self.n_states):
            G.add_node(state)

        edge_colors = []
        for state in range(self.n_states):
            for other_state in range(self.n_states):
                w = F[state, other_state]
                if w > connection_threshold:
                    G.add_edge(state, other_state, weight=w)
                    edge_colors.append(F[state, other_state])

        positions = np.zeros((self.n_states, 2))
        for i in range(self.n_states):
            positions[i] = self.state_to_coordinate(i)
        layout = {k: positions[k] for k in G.nodes.keys()}

        nx.draw_networkx_nodes(G, layout, node_size=30, ax=ax)
        nx.draw_networkx_edges(G, layout, ax=ax, arrowstyle='->', edge_cmap=cmap, edge_color=edge_colors,
                               connectionstyle='arc3, rad=0.1')
        return edge_colors

    @plotting_function
    def plot_2d_map(self, ax, barriers: bool = True, barrier_mode: str = 'filled'):  # pragma: no cover
        r""" Plots the drunkard's map onto an already existing matplotlib axis. One can select whether to draw barriers
        and if barriers should be drawn, whether they have a 'filled' face or be 'hollow' with just the border
        being colored.

        Parameters
        ----------
        ax : matplotlib axis
            The axis.
        barriers : bool, default=True
            Whether to draw the barriers.
        barrier_mode : str, default='filled'
            How to draw barriers - can be one of 'filled' and 'hollow'.

        Returns
        -------
        handles, labels
            Handles and labels for a matplotlib legend.
        """
        import numpy as np
        from matplotlib.patches import Rectangle

        valid_barrier_modes = ('filled', 'hollow')
        if barrier_mode not in valid_barrier_modes:
            raise ValueError(f"Barrier mode '{barrier_mode}' not supported, valid modes are {valid_barrier_modes}.")

        ax.scatter(*self.home_location.T, marker='*', label='Home', c='red', s=150, zorder=5)
        ax.scatter(*self.bar_location.T, marker='*', label='Bar', c='orange', s=150, zorder=5)

        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))
        ax.set_xlabel('coordinate x')
        ax.set_ylabel('coordinate y')

        rect = None
        for state in range(self.n_states):
            coord = self.state_to_coordinate(state)
            if self._is_home_or_bar_coord(coord):
                ax.add_patch(Rectangle((coord[0] - .5, coord[1] - .5), 1., 1., alpha=.3, color='green'))
            elif barriers and coord in self.barriers:
                barrier_ix = self.barriers.index(coord)
                weight = self.barrier_weights[barrier_ix]
                if weight is None:
                    rect = Rectangle((coord[0] - .5, coord[1] - .5), 1., 1., alpha=.5, color='red', lw=3.)
                    if barrier_mode == 'hollow':
                        rect.set_facecolor('none')
                    ax.add_patch(rect)
                else:
                    rect = Rectangle((coord[0] - .5, coord[1] - .5), 1., 1., alpha=.2, color='red', lw=3.)
                    if barrier_mode == 'hollow':
                        rect.set_facecolor('none')
                    ax.add_patch(rect)

        for grid_point in np.arange(-.5, self.grid_size[0] + .5, 1):
            ax.axhline(grid_point, linestyle='-', color='grey', lw=.5)

        for grid_point in np.arange(-.5, self.grid_size[1] + .5, 1):
            ax.axvline(grid_point, linestyle='-', color='grey', lw=.5)

        handles, labels = ax.get_legend_handles_labels()
        if rect is not None:
            handles.append(rect)
            labels.append("Barrier")

        ax.set_xlim([-.5, self.grid_size[0] - .5])
        ax.set_ylim([-.5, self.grid_size[1] - .5])

        return handles, labels
