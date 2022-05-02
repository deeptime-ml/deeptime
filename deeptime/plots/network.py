from typing import Optional, Union, Dict, Tuple

import numpy as np
from scipy.sparse import issparse

from deeptime.plots.util import default_image_cmap, default_line_width
from deeptime.util.decorators import plotting_function
from deeptime.util.types import ensure_number_array


class NetworkPlot:
    r"""Plot of a network with nodes and arcs.

    Parameters
    ----------
    adjacency_matrix : ndarray
        weight matrix or adjacency matrix of the network to visualize
    pos : ndarray or dict[int, ndarray]
        user-defined positions as (n,2) array

    Examples
    --------
    We define first define a reactive flux by taking the following transition
    matrix and computing TPT from state 2 to 3.

    >>> import numpy as np
    >>> P = np.array([[0.8,  0.15, 0.05,  0.0,  0.0],
    ...               [0.1,  0.75, 0.05, 0.05, 0.05],
    ...               [0.05,  0.1,  0.8,  0.0,  0.05],
    ...               [0.0,  0.2, 0.0,  0.8,  0.0],
    ...               [0.0,  0.02, 0.02, 0.0,  0.96]])
    >>> from deeptime.markov.msm import MarkovStateModel
    >>> flux = MarkovStateModel(P).reactive_flux([2], [3])

    now plot the gross flux
    >>> import networkx as nx
    >>> positions = nx.spring_layout(nx.from_numpy_array(flux.gross_flux))
    >>> NetworkPlot(flux.gross_flux, positions).plot_network() # doctest: +ELLIPSIS
    <...Figure...
    """

    def __init__(self, adjacency_matrix, pos):
        self.adjacency_matrix = adjacency_matrix
        self.pos = pos

    @property
    def pos(self) -> np.ndarray:
        return self._pos

    @pos.setter
    def pos(self, value: Union[np.ndarray, Dict[int, np.ndarray]]):
        if len(value) < self.n_nodes:
            raise ValueError(f'Given less positions ({len(value)}) than states ({self.n_nodes})')
        if isinstance(value, dict):
            value = np.stack((value[i] for i in range(len(value))))
        self._pos = value

    @property
    def n_nodes(self):
        return self.adjacency_matrix.shape[0]

    @property
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return ((np.min(self.pos[:, 0]), np.max(self.pos[:, 0])),
                (np.min(self.pos[:, 1]), np.max(self.pos[:, 1])))

    @property
    def d_x(self):
        return self.bounds[0][1] - self.bounds[0][0]

    @property
    def d_y(self):
        return self.bounds[1][1] - self.bounds[1][0]

    @staticmethod
    def _draw_arrow(ax, pos_1, pos_2, label="", width=1.0, arrow_curvature=1.0, color="grey",
                    patchA=None, patchB=None, shrinkA=2, shrinkB=1, arrow_label_size=None, arrow_label_location=.55):
        r"""
        Draws a slightly curved arrow from (x1,y1) to (x2,y2).
        Will allow the given patches at start and end.
        """
        from matplotlib import patches

        # set arrow properties
        dist = np.linalg.norm(pos_2 - pos_1)
        arrow_curvature *= 0.075  # standard scale
        rad = arrow_curvature / dist
        tail_width = width
        head_width = max(2., 2 * width)
        # head_length = max(1.5, head_width)

        arrow_style = patches.ArrowStyle.Simple(head_length=head_width, head_width=head_width, tail_width=tail_width)
        connection_style = patches.ConnectionStyle.Arc3(rad=-rad)
        arr = patches.FancyArrowPatch(posA=pos_1, posB=pos_2, arrowstyle=arrow_style,
                                      connectionstyle=connection_style, color=color,
                                      shrinkA=shrinkA, shrinkB=shrinkB, patchA=patchA, patchB=patchB,
                                      zorder=0, transform=ax.transData)
        ax.add_patch(arr)

        # Bezier control point
        control_vertex = np.array(arr.get_connectionstyle().connect(pos_1, pos_2).vertices[1])
        # quadratic Bezier at slightly shifted midpoint t = arrow_label_location
        t = arrow_label_location  # shorthand
        ptext = (1 - t)**2 * pos_1 + 2 * (1 - t) * t * control_vertex + t**2 * pos_2

        ax.text(
            ptext[0], ptext[1], label, size=arrow_label_size,
            horizontalalignment='center', verticalalignment='center', zorder=1,
            transform=ax.transData)

    def plot_network(self, ax=None, state_sizes=None, state_scale=1.0, state_colors='#ff5500', state_labels='auto',
                     arrow_scale=1.0, arrow_curvature=1.0, arrow_labels='weights', arrow_label_format='{:.1e}',
                     arrow_label_location=0.55, cmap=None, **textkwargs):
        r"""Draws a network using discs and curved arrows.

        The thicknesses and labels of the arrows are taken from the off-diagonal matrix elements
        in A.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional, default=None
            The axes to plot on.
        state_sizes : ndarray, optional, default=None
            List of state sizes to plot, must be of length `n_states`. It effectively evaluates as

            .. math::
                \frac{\mathrm{state\_scale} \min (d_x, d_y)^2}{2 n_\mathrm{nodes}}
                  \frac{\mathrm{state_sizes}}{\| \mathrm{state_sizes} \|_{\max}}

            with :math:`\mathrm{state\_sizes}` interpreted as `1` in case of `None`. In particular this means that
            the states scale their size with respect to the volume. I.e., `state_scale=[1, 2]` leads to the second
            state drawn as a circle with twice the volume of the first.
        state_scale : float, default=1.
            Uniform scaling factor for `state_sizes`.
        state_colors : str or list of float, default='#ff5500'
            The color to use for drawing states. If given as list of float, uses the colormap (`cmap` argument) to
            determine the color.
        state_labels : None or 'auto' or list of str, default='auto'
            The state labels. If 'auto', just enumerates the states. In case of `None` no state labels are depicted,
            otherwise assigns each state its label based on the list.
        """
        # Set the default values for the text dictionary
        from matplotlib import pyplot as plt
        from matplotlib import colors
        from matplotlib import patches
        if ax is None:
            ax = plt.gca()
        if cmap is None:
            cmap = default_image_cmap()
        textkwargs.setdefault('size', None)
        textkwargs.setdefault('horizontalalignment', 'center')
        textkwargs.setdefault('verticalalignment', 'center')
        textkwargs.setdefault('color', 'black')
        # remove the temporary key 'arrow_label_size' as it cannot be parsed by plt.text!
        arrow_label_size = textkwargs.pop('arrow_label_size', textkwargs['size'])
        # sizes of nodes
        if state_sizes is None:
            state_sizes = 0.5 * state_scale * min(self.d_x, self.d_y) ** 2 * np.ones(self.n_nodes) / float(self.n_nodes)
        else:
            state_sizes = 0.5 * state_scale * min(self.d_x, self.d_y) ** 2 * state_sizes \
                          / (np.max(state_sizes) * float(self.n_nodes))
        # automatic arrow rescaling
        diag_mask = np.logical_not(np.eye(self.adjacency_matrix.shape[0], dtype=bool))
        default_lw = default_line_width()
        arrow_scale *= 2 * default_lw / np.max(self.adjacency_matrix[diag_mask & (self.adjacency_matrix > 0)])

        # set node labels
        if state_labels is None:
            pass
        elif isinstance(state_labels, str) and state_labels == 'auto':
            state_labels = [str(i) for i in np.arange(self.n_nodes)]
        else:
            if len(state_labels) != self.n_nodes:
                raise ValueError(f"length of state_labels({len(state_labels)}) has to match "
                                 f"length of states({self.n_nodes}).")
        # set node colors
        if isinstance(state_colors, str):
            state_colors = np.array([colors.to_rgb(state_colors)] * self.n_nodes)
        else:
            state_colors = ensure_number_array(state_colors, ndim=1)
            if not isinstance(cmap, colors.Colormap):
                cmap = plt.get_cmap(cmap)
            assert isinstance(cmap, colors.Colormap)
            state_colors = np.array([cmap(x) for x in state_colors])
        if len(state_colors) != self.n_nodes:
            raise ValueError(f"Mismatch between n_states and #state_colors ({self.n_nodes} vs {len(state_colors)}).")

        # set arrow labels
        if isinstance(arrow_labels, np.ndarray):
            L = arrow_labels
            if isinstance(arrow_labels[0, 0], str):
                arrow_label_format = '{}'
        elif isinstance(arrow_labels, str) and arrow_labels.lower() == 'weights':
            L = np.copy(self.adjacency_matrix)
        elif arrow_labels is None:
            L = np.full(self.adjacency_matrix.shape, fill_value='', dtype=object)
            arrow_label_format = '{}'
        else:
            raise ValueError('invalid arrow labels')

        # draw circles
        circles = []
        for i in range(self.n_nodes):
            circles.append(
                patches.Circle(self.pos[i], radius=np.sqrt(0.5 * state_sizes[i]) / 2.0,
                               color=state_colors[i], zorder=2)
            )
            ax.add_patch(circles[-1])

            # add annotation
            if state_labels is not None:
                ax.text(self.pos[i][0], self.pos[i][1], state_labels[i], zorder=3, **textkwargs)

        assert len(circles) == self.n_nodes, f"{len(circles)} != {self.n_nodes}"

        # draw arrows
        for i, j in zip(*np.triu_indices(self.n_nodes, k=1)):  # upper triangular indices with 0 <= i < j < n_nodes
            if abs(self.adjacency_matrix[i, j]) > 0:
                self._draw_arrow(
                    ax, self.pos[i], self.pos[j],
                    label=arrow_label_format.format(L[i, j]), width=arrow_scale * self.adjacency_matrix[i, j],
                    arrow_curvature=arrow_curvature, patchA=circles[i], patchB=circles[j],
                    shrinkA=3, shrinkB=0, arrow_label_size=arrow_label_size, arrow_label_location=arrow_label_location)
            if abs(self.adjacency_matrix[j, i]) > 0:
                self._draw_arrow(
                    ax, self.pos[j], self.pos[i],
                    label=arrow_label_format.format(L[j, i]), width=arrow_scale * self.adjacency_matrix[j, i],
                    arrow_curvature=arrow_curvature, patchA=circles[j], patchB=circles[i],
                    shrinkA=3, shrinkB=0, arrow_label_size=arrow_label_size, arrow_label_location=arrow_label_location)

        ax.autoscale_view()
        return ax


@plotting_function(requires_networkx=True)
def plot_adjacency(adjacency_matrix, positions: Optional[np.ndarray] = None, layout=None, ax=None, node_size=None):
    import networkx as nx
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()
    if positions is not None:
        if positions.ndim != 2 or positions.shape[0] != adjacency_matrix.shape[0] or positions.shape[1] != 2:
            raise ValueError(f"Unsupported positions array. Has to be ({adjacency_matrix.shape[0]}, 2)-shaped but "
                             f"was of shape {positions.shape}.")

    if issparse(adjacency_matrix):
        graph = nx.from_scipy_sparse_matrix(adjacency_matrix)
    else:
        graph = nx.from_numpy_matrix(adjacency_matrix)
    if positions is not None:
        def layout(g):
            return dict(zip(g.nodes(), positions))
    else:
        layout = nx.spring_layout if layout is None else layout
    assert layout is not None
    pos = layout(graph)

    plot = NetworkPlot(adjacency_matrix, pos=pos)
    ax = plot.plot_network(ax=ax, state_sizes=node_size)
    return ax, graph
