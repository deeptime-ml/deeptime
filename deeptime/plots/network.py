from typing import Optional, Union, Dict, Tuple, List

import numpy as np
import scipy
from scipy.sparse import issparse

from deeptime.markov import ReactiveFlux
from deeptime.markov.msm import MarkovStateModel
from deeptime.plots.util import default_image_cmap, default_line_width
from deeptime.util.decorators import plotting_function
from deeptime.util.types import ensure_number_array


class Network:
    r"""Plot of a network with nodes and arcs.

    Parameters
    ----------
    adjacency_matrix : ndarray
        weight matrix or adjacency matrix of the network to visualize
    pos : ndarray or dict[int, ndarray]
        user-defined positions as (n,2) array
    cmap : matplotlib.colors.Colormap or str, default=None
        The colormap for `state_color`s.
    state_sizes : ndarray, optional, default=None
        List of state sizes to plot, must be of length `n_states`. It effectively evaluates as

        .. math::
            \frac{\mathrm{state\_scale} \cdot (\min (d_x, d_y))^2}{2 n_\mathrm{nodes}}
              \frac{\mathrm{state\_sizes}}{\| \mathrm{state\_sizes} \|_{\max}}

        with :math:`\mathrm{state\_sizes}` interpreted as `1` in case of `None`. In particular this means that
        the states scale their size with respect to the volume. I.e., `state_sizes=[1, 2]` leads to the second
        state drawn as a circle with twice the volume of the first.
    state_scale : float, default=1.
        Uniform scaling factor for `state_sizes`.
    state_colors : str or list of float, default='#ff5500'
        The color to use for drawing states. If given as list of float, uses the colormap (`cmap` argument) to
        determine the color.
    state_labels : None or 'auto' or list of str, default='auto'
        The state labels. If 'auto', just enumerates the states. In case of `None` no state labels are depicted,
        otherwise assigns each state its label based on the list.
    edge_scale : float, optional, default=1.
        Linear scaling coefficient for all arrow widths. Takes the default line width `rcParams['lines.linewidth']`
        into account.
    edge_curvature : float, optional, default=1.
        Linear scaling coefficient for arrow curvature. Setting it to `0` produces straight arrows.
    edge_labels : 'weights' or ndarray or None, default='weights'
        If 'weights', arrows obtain labels according to the weights in the adjacency matrix. If ndarray the dtype
        is expected to be object and the argument should be a `(n, n)` matrix with labels. If None, no labels are
        printed.
    edge_label_format : str, default='{:.1e}'
        Format string for arrow labels. Only has an effect if arrow_labels is set to `weights`.
    edge_label_location : float, default=0.55
        Location of the arrow labels on the curve. Should be between 0 (meaning on the source state) and 1 (meaning
        on the target state). Defaults to 0.55, i.e., slightly shifted toward the target state from the midpoint.

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

    Now plot the gross flux using networkx spring layout.

    >>> import networkx as nx
    >>> positions = nx.spring_layout(nx.from_numpy_array(flux.gross_flux))
    >>> Network(flux.gross_flux, positions).plot() # doctest: +ELLIPSIS
    <...Axes...
    """

    def __init__(self, adjacency_matrix, pos, cmap=None,
                 state_sizes=None, state_scale=1.0, state_colors='#ff5500', state_labels='auto',
                 edge_scale: float = 1., edge_curvature: float = 1.0,
                 edge_labels: Optional[Union[str, np.ndarray]] = 'weights', edge_label_format: str = '{:.1e}',
                 edge_label_location: float = 0.55):
        self.adjacency_matrix = adjacency_matrix
        self.pos = pos
        self.edge_scale = edge_scale
        self.edge_curvature = edge_curvature
        self.edge_labels = edge_labels
        self.edge_label_format = edge_label_format
        self.edge_label_location = edge_label_location
        self.cmap = cmap
        self.state_sizes = state_sizes
        self.state_scale = state_scale
        self.state_colors = state_colors
        self.state_labels = state_labels

    @property
    def adjacency_matrix(self):
        r""" The adjacency matrix. Can be sparse. """
        return self._adjacency_matrix

    @adjacency_matrix.setter
    def adjacency_matrix(self, value):
        if issparse(value):
            self._adjacency_matrix = scipy.sparse.csr_matrix(value)
        else:
            self._adjacency_matrix = value

    @property
    def pos(self) -> np.ndarray:
        r""" Position array. If the object was constructed with a dict-style layout (as generated by networkx),
        the positions are converted back into an array format.

        :getter: Yields the positions.
        :setter: Sets the positions, can also be provided as dict.
        :type: ndarray
        """
        return self._pos

    @pos.setter
    def pos(self, value: Union[np.ndarray, Dict[int, np.ndarray]]):
        if len(value) < self.n_nodes:
            raise ValueError(f'Given less positions ({len(value)}) than states ({self.n_nodes})')
        if isinstance(value, dict):
            value = np.stack([value[i] for i in range(len(value))])
        self._pos = value

    @property
    def n_nodes(self):
        r""" Number of nodes in the network. """
        return self.adjacency_matrix.shape[0]

    @property
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        r""" The bounds of node positions. Yields `((xmin, xmax), (ymin, ymax))`. """
        return ((np.min(self.pos[:, 0]), np.max(self.pos[:, 0])),
                (np.min(self.pos[:, 1]), np.max(self.pos[:, 1])))

    @property
    def d_x(self):
        r""" Width of the network. """
        return self.bounds[0][1] - self.bounds[0][0]

    @property
    def d_y(self):
        r""" Height of the network. """
        return self.bounds[1][1] - self.bounds[1][0]

    @property
    def cmap(self):
        r""" The colormap to use for states if colors are given as one-dimensional array of floats.

        :type: matplotlib.colors.Colormap
        """
        return self._cmap

    @cmap.setter
    def cmap(self, value):
        from matplotlib import colors
        import matplotlib.pyplot as plt
        if value is None:
            value = default_image_cmap()
        if not isinstance(value, colors.Colormap):
            value = plt.get_cmap(value)
        self._cmap = value

    @staticmethod
    def _draw_arrow(ax, pos_1, pos_2, label="", width=1.0, arrow_curvature=1.0, color="grey",
                    patchA=None, patchB=None, shrinkA=2, shrinkB=1, arrow_label_size=None, arrow_label_location=.55):
        r""" Draws a slightly curved arrow from (x1,y1) to (x2,y2). Will allow the given patches at start and end. """
        from matplotlib import patches

        # set arrow properties
        dist = np.linalg.norm(pos_2 - pos_1)
        arrow_curvature *= 0.075  # standard scale
        rad = arrow_curvature / dist
        tail_width = width
        head_width = max(2., 2 * width)

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
        ptext = (1 - t) ** 2 * pos_1 + 2 * (1 - t) * t * control_vertex + t ** 2 * pos_2

        ax.text(*ptext, label, size=arrow_label_size, horizontalalignment='center', verticalalignment='center',
                zorder=3, transform=ax.transData)

    @property
    def edge_base_scale(self):
        r""" Base scale for edges depending on the matplotlib default line width and the maximum off-diagonal
        element in the adjacency matrix. Returns default line width if all off-diagonal elements are zero. """
        if issparse(self.adjacency_matrix):
            mat = self.adjacency_matrix.tocoo()
            max_off_diag = 0
            for i, j, v in zip(mat.row, mat.col, mat.data):
                if i != j:
                    max_off_diag = max(max_off_diag, abs(v))
        else:
            diag_mask = np.logical_not(np.eye(self.adjacency_matrix.shape[0], dtype=bool))
            max_off_diag = np.max(np.abs(self.adjacency_matrix[diag_mask]))
        default_lw = default_line_width()
        return 2 * default_lw / max_off_diag if max_off_diag > 0 else default_lw

    @property
    def edge_labels(self) -> Optional[np.ndarray]:
        r""" Edge labels. Can be left None for no labels, otherwise must be matrix of same shape as adjacency matrix.
        containing numerical values (in conjunction with :attr:`edge_label_format`) or strings."""
        return self._edge_labels

    @edge_labels.setter
    def edge_labels(self, value):
        if isinstance(value, np.ndarray):
            if value.shape != self.adjacency_matrix.shape:
                raise ValueError(f"Arrow labels matrix has shape {value.shape} =/= {self.adjacency_matrix.shape}")
            self._edge_labels = value
        elif isinstance(value, str) and value.lower() == 'weights':
            self._edge_labels = self.adjacency_matrix
        elif value is None:
            self._edge_labels = None
        else:
            raise ValueError("Invalid edge labels, should be 'weights', ndarray of strings or None.")

    @property
    def state_sizes(self) -> np.ndarray:
        r""" Sizes of states. Can be left `None` which amounts to state sizes of 1. """
        return self._state_sizes

    @state_sizes.setter
    def state_sizes(self, value: Optional[np.ndarray]):
        if value is not None and len(value) != self.n_nodes:
            raise ValueError(f"State sizes must correspond to states (# = {self.n_nodes}) but was of "
                             f"length {len(value)}")
        self._state_sizes = np.asarray(value, dtype=float) if value is not None else np.ones(self.n_nodes)

    @property
    def node_sizes(self):
        r""" The effective node sizes. Rescales to account for size of plot. """
        return 0.5 * self.state_scale * min(self.d_x, self.d_y) ** 2 * self.state_sizes \
               / (np.max(self.state_sizes) * float(self.n_nodes))

    @property
    def state_labels(self) -> List[str]:
        r""" State labels. """
        return self._state_labels

    @state_labels.setter
    def state_labels(self, value):
        if isinstance(value, str) and value == 'auto':
            value = [str(i) for i in np.arange(self.n_nodes)]
        else:
            if len(value) != self.n_nodes:
                raise ValueError(f"length of state_labels({len(value)}) has to match "
                                 f"length of states({self.n_nodes}).")
        self._state_labels = value

    @property
    def state_colors(self) -> np.ndarray:
        """ The state colors in (N, rgb(a))-shaped array. """
        return self._state_colors

    @state_colors.setter
    def state_colors(self, value):
        # set node colors
        from matplotlib import colors
        if isinstance(value, str):
            state_colors = np.array([colors.to_rgba(value)] * self.n_nodes)
        else:
            state_colors = ensure_number_array(value)
            if state_colors.ndim == 1:
                state_colors = np.array([self.cmap(x) for x in state_colors])
            elif state_colors.ndim == 2 and state_colors.shape[1] in (3, 4):
                pass  # ok: rgb(a) values
            else:
                raise ValueError(f"state color(s) can only be individual color or float range or rgb(a) values "
                                 f"but was {state_colors}")
        if len(state_colors) != self.n_nodes:
            raise ValueError(f"Mismatch between n_states and #state_colors ({self.n_nodes} vs {len(state_colors)}).")
        self._state_colors = state_colors

    def edge_label(self, i, j) -> str:
        r""" Yields the formatted edge label for edge i->j.

        Parameters
        ----------
        i : int
            edge i
        j : int
            edge j

        Returns
        -------
        label : str
            The edge label.
        """
        if self.edge_labels is None:
            return ""
        else:
            fmt = self.edge_label_format if np.issubdtype(self.edge_labels.dtype, np.number) else "{}"
            return fmt.format(self.edge_labels[i, j])

    def plot(self, ax=None, **textkwargs):
        r"""Draws a network using discs and curved arrows.

        The thicknesses and labels of the arrows are taken from the off-diagonal matrix elements
        in A.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional, default=None
            The axes to plot on.
        **textkwargs
            Optional arguments for state labels.
        """
        # Set the default values for the text dictionary
        from matplotlib import pyplot as plt
        from matplotlib import patches
        if ax is None:
            ax = plt.gca()
        textkwargs.setdefault('size', None)
        textkwargs.setdefault('horizontalalignment', 'center')
        textkwargs.setdefault('verticalalignment', 'center')
        textkwargs.setdefault('color', 'black')
        # remove the temporary key 'arrow_label_size' as it cannot be parsed by plt.text!
        arrow_label_size = textkwargs.pop('arrow_label_size', textkwargs['size'])
        # automatic arrow rescaling
        edge_scale = self.edge_base_scale * self.edge_scale

        # draw circles
        node_sizes = self.node_sizes
        circles = []
        for i in range(self.n_nodes):
            circles.append(
                patches.Circle(self.pos[i], radius=np.sqrt(0.5 * node_sizes[i]) / 2.0,
                               color=self.state_colors[i], zorder=2)
            )
            ax.add_patch(circles[-1])

            # add annotation
            if self.state_labels is not None:
                ax.text(self.pos[i][0], self.pos[i][1], self.state_labels[i], zorder=3, **textkwargs)

        # draw arrows
        for i, j in zip(*np.triu_indices(self.n_nodes, k=1)):  # upper triangular indices with 0 <= i < j < n_nodes
            if self.adjacency_matrix[i, j] != 0:
                label = self.edge_label(i, j)
                width = edge_scale * self.adjacency_matrix[i, j]
                self._draw_arrow(ax, self.pos[i], self.pos[j], label=label, width=width,
                                 arrow_curvature=self.edge_curvature, patchA=circles[i], patchB=circles[j],
                                 shrinkA=3, shrinkB=0,
                                 arrow_label_size=arrow_label_size, arrow_label_location=self.edge_label_location)
            if self.adjacency_matrix[j, i] != 0:
                label = self.edge_label(j, i)
                width = edge_scale * self.adjacency_matrix[j, i]
                self._draw_arrow(ax, self.pos[j], self.pos[i], label=label, width=width,
                                 arrow_curvature=self.edge_curvature, patchA=circles[j], patchB=circles[i],
                                 shrinkA=3, shrinkB=0,
                                 arrow_label_size=arrow_label_size, arrow_label_location=self.edge_label_location)

        ax.autoscale_view()
        return ax


@plotting_function(requires_networkx=True)
def plot_adjacency(adjacency_matrix, positions: Optional[np.ndarray] = None, layout=None, ax=None, scale_states=True):
    r"""Plot an adjacency matrix. The edges are scaled according to the respective values. For more fine-grained
    control use :class:`Network`.

    .. plot::

        import numpy as np
        P = np.array([[0.8, 0.15, 0.05, 0.0],
                      [0.1, 0.75, 0.05, 0.05],
                      [0.05, 0.1, 0.8, 0.0],
                      [0.0, 0.2, 0.0, 0.8]])

        from deeptime.plots import plot_adjacency
        ax = plot_adjacency(P, positions=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
        ax.set_aspect('equal')


    Parameters
    ----------
    adjacency_matrix : ndarray or scipy sparse matrix
        Adjacency matrix to plot. Could be, e.g., a MSM transition matrix.
    positions : ndarray, optional, default=None
        A (N, 2)-shaped ndarray containing positions for the nodes of the adjacency matrix. If left as `None`, the
        layout is algorithmically determined (based on the algorithm specified in the `layout` parameter).
    layout : callable, optional, default=None
        The automatic layout to use. Only has an effect, if `positions` is `None`. In that case, it defaults to
        the `networkx.spring_layout`.
        Can be any callable which takes a networkx graph as first argument and yields a position array or dict.
    ax : matplotlib.axes.Axes, optional, default=None
        The axes to plot on. Otherwise, uses the current axes (via `plt.gca()`).
    scale_states : bool, default=True
        Whether to scale nodes according to the value on the diagonal of the adjacency matrix.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes that were plotted on.

    See Also
    --------
    Network
    """
    import networkx as nx
    if positions is not None:
        if positions.ndim != 2 or positions.shape[0] != adjacency_matrix.shape[0] or positions.shape[1] != 2:
            raise ValueError(f"Unsupported positions array. Has to be ({adjacency_matrix.shape[0]}, 2)-shaped but "
                             f"was of shape {positions.shape}.")

    if issparse(adjacency_matrix):
        graph = nx.from_scipy_sparse_array(adjacency_matrix)
    else:
        graph = nx.from_numpy_array(adjacency_matrix)
    if positions is not None:
        def layout(g):
            return dict(zip(g.nodes(), positions))
    else:
        layout = nx.spring_layout if layout is None else layout
    assert layout is not None
    pos = layout(graph)

    if scale_states:
        state_sizes = adjacency_matrix.diagonal() if issparse(adjacency_matrix) else np.diag(adjacency_matrix)
    else:
        state_sizes = None
    plot = Network(adjacency_matrix, pos=pos, state_sizes=state_sizes)
    ax = plot.plot(ax=ax)
    return ax


@plotting_function(requires_networkx=True)
def plot_markov_model(msm: Union[MarkovStateModel, np.ndarray], pos=None, state_sizes=None, state_scale=1.0,
                      state_colors='#ff5500', state_labels='auto',
                      minflux=1e-6, edge_scale=1.0, edge_curvature=1.0, edge_labels='weights',
                      edge_label_format='{:.2e}', ax=None, **textkwargs):
    r"""Network representation of MSM transition matrix.

    This visualization is not optimized for large matrices. It is meant to be used for the visualization of small
    models with up to 10-20 states, e.g., obtained by a HMM coarse-graining. If used with large network, the automatic
    node positioning will be very slow and may still look ugly.

    .. plot::

        import numpy as np
        P = np.array([[0.8, 0.15, 0.05, 0.0, 0.0],
                      [0.1, 0.75, 0.05, 0.05, 0.05],
                      [0.05, 0.1, 0.8, 0.0, 0.05],
                      [0.0, 0.2, 0.0, 0.8, 0.0],
                      [1e-7, 0.02 - 1e-7, 0.02, 0.0, 0.96]])

        from deeptime.plots import plot_markov_model
        ax, pos = plot_markov_model(P)
        ax.set_aspect('equal')

    Parameters
    ----------
    msm : MarkovStateModel or ndarray
        The Markov state model to plot. Can also be the transition matrix.
    pos : ndarray(n,2) or dict, optional, default=None
        User-defined positions to draw the states on. If not given, will try to place them automatically.
        The output of networkx layouts can be used for this argument.
    state_sizes : ndarray(n), optional, default=None
        User-defined areas of the discs drawn for each state. If not given, the
        stationary probability of P will be used.
    state_colors : string, ndarray(n), or list, optional, default='#ff5500' (orange)
        string :
            a Hex code for a single color used for all states
        array :
            n values in [0,1] which will result in a grayscale plot
        list :
            of len = nstates, with a color for each state. The list can mix strings, RGB values and
            hex codes, e.g. :py:obj:`state_colors` = ['g', 'red', [.23, .34, .35], '#ff5500'] is
            possible.
    state_labels : list of strings, optional, default is 'auto'
        A list with a label for each state, to be displayed at the center
        of each node/state. If left to 'auto', the labels are automatically set to the state
        indices.
    minflux : float, optional, default=1e-6
        The minimal flux (p_i * p_ij) for a transition to be drawn
    edge_scale : float, optional, default=1.0
        Relative arrow scale. Set to a value different from 1 to increase or decrease the arrow width.
    edge_curvature : float, optional, default=1.0
        Relative arrow curvature. Set to a value different from 1 to make arrows more or less curved.
    edge_labels : 'weights', None or a ndarray(n,n) with label strings. Optional, default='weights'
        Strings to be placed upon arrows. If None, no labels will be used.
        If 'weights', the elements of P will be used. If a matrix of strings is given by the user these will be used.
    edge_label_format : str, optional, default='{2.e}'
        The numeric format to print the arrow labels.
    ax : matplotlib Axes object, optional, default=None
        The axes to plot to. When set to None a new Axes (and Figure) object will be used.
    textkwargs : optional argument for the text of the state and arrow labels.
        See https://matplotlib.org/stable/api/text_api.html for more info. The
        parameter 'size' refers to the size of the state and arrow labels and overwrites the
        matplotlib default. The parameter 'arrow_label_size' is only used for the arrow labels;
        please note that 'arrow_label_size' is not part of matplotlib.text.Text's set of parameters
        and will raise an exception when passed to matplotlib.text.Text directly.

    Returns
    -------
    ax, pos : matplotlib.axes.Axes, ndarray(n,2)
        An axes object containing the plot and the positions of states.
        Can be used later to plot a different network representation (e.g. the flux)
    """
    if not isinstance(msm, MarkovStateModel):
        msm = MarkovStateModel(msm)
    P = msm.transition_matrix.copy()
    if state_sizes is None:
        state_sizes = msm.stationary_distribution
    if minflux > 0:
        if msm.sparse:
            sddiag = scipy.sparse.diags(msm.stationary_distribution)
        else:
            sddiag = np.diag(msm.stationary_distribution)
        flux = sddiag.dot(msm.transition_matrix)
        if msm.sparse:
            P = P.multiply(P >= minflux)
            P.eliminate_zeros()
        else:
            P[flux < minflux] = 0.0
    if pos is None:
        import networkx as nx
        graph = nx.from_scipy_sparse_array(P) if msm.sparse else nx.from_numpy_array(P)
        pos = nx.spring_layout(graph)
    network = Network(P, pos=pos,  state_scale=state_scale, state_colors=state_colors, state_labels=state_labels,
                      state_sizes=state_sizes, edge_scale=edge_scale, edge_curvature=edge_curvature,
                      edge_labels=edge_labels, edge_label_format=edge_label_format)
    return network.plot(ax=ax, **textkwargs), pos


def plot_flux(flux: ReactiveFlux, state_sizes=None, flux_scale=1.0, state_scale=1.0, state_colors=None,
              state_labels='auto', minflux=1e-9, edge_scale=1.0, edge_curvature=1.0, edge_labels='weights',
              edge_label_format='{:.2e}', attribute_to_plot='net_flux', show_committor=True, cmap='coolwarm',
              ax=None, **textkwargs):
    r""" Network representation of a reactive flux.

    This visualization is not optimized for large fluxes. It is meant to be used for the visualization of small models
    with up to 10-20 states, e.g. obtained by a PCCA-based coarse-graining of the full flux. If used with large
    network, the automatic node positioning will be very slow and may still look ugly.

    .. plot::

        import numpy as np
        P = np.array([[0.8, 0.15, 0.05, 0.0, 0.0],
                     [0.1, 0.75, 0.05, 0.05, 0.05],
                     [0.05, 0.1, 0.8, 0.0, 0.05],
                     [0.0, 0.2, 0.0, 0.8, 0.0],
                     [0.0, 0.02, 0.02, 0.0, 0.96]])

        from deeptime.markov.msm import MarkovStateModel
        flux = MarkovStateModel(P).reactive_flux([2], [3])

        from deeptime.plots import plot_flux
        ax, pos = plot_flux(flux, flux_scale=100)
        ax.set_aspect('equal')

    Parameters
    ----------
    flux : ReactiveFlux
        The flux to plot.
    state_sizes : ndarray(n), optional, default=None
        User-defined areas of the discs drawn for each state. If not given, the stationary probability will be used.
    flux_scale : float, optional, default=1.0
        Uniform scaling of the flux values.
    state_scale : float, optional, default=1.0
        Uniform scaling of state discs.
    state_colors : string, ndarray(n), or list, optional, default=None
        Per default colors according to `cmap` with respect to committor probability. Otherwise if given as:
            string :
                a Hex code for a single color used for all states
            array :
                n values in [0,1] which will result in a grayscale plot
            list :
                of len = nstates, with a color for each state. The list can mix strings, RGB values and
                hex codes, e.g. :py:obj:`state_colors` = ['g', 'red', [.23, .34, .35], '#ff5500'] is
                possible.
    state_labels : list of strings, optional, default is 'auto'
        A list with a label for each state, to be displayed at the center of each node/state. If left to 'auto',
        the labels are automatically set to the state indices.
    minflux : float, optional, default=1e-9
        The minimal flux for a transition to be drawn
    edge_scale : float, optional, default=1.0
        Relative arrow scale. Set to a value different from 1 to increase or decrease the arrow width.
    edge_curvature : float, optional, default=1.0
        Relative arrow curvature. Set to a value different from 1 to make arrows more or less curved.
    edge_labels : 'weights', None or a ndarray(n,n) with label strings. Optional, default='weights'
        Strings to be placed upon arrows. If None, no labels will be used. If 'weights', the elements of P will be
        used. If a matrix of strings is given by the user these will be used.
    edge_label_format : str, optional, default='{:.2e}'
        The numeric format to print the arrow labels.
    attribute_to_plot : str, optional, default='net_flux'
        specify the attribute of the flux object to plot.
    show_committor: boolean (default=False)
        Print the committor value on the x-axis.
    ax : matplotlib.axes.Axes, optional, default=None
        The axes to plot to. When set to None, `gca()` is used.
    textkwargs : optional argument for the text of the state and arrow labels.
        See https://matplotlib.org/stable/api/text_api.html for more info. The
        parameter 'size' refers to the size of the state and arrow labels and overwrites the
        matplotlib default. The parameter 'arrow_label_size' is only used for the arrow labels;
        please note that 'arrow_label_size' is not part of matplotlib.text.Text's set of parameters
        and will raise an exception when passed to matplotlib.text.Text directly.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object on which the flux was plotted.
    pos : ndarray
        The positions.
    """
    from . import _plots_bindings
    F = flux_scale * getattr(flux, attribute_to_plot)
    if state_sizes is None:
        state_sizes = flux.stationary_distribution
    if state_colors is None:
        state_colors = flux.forward_committor

    # initial positions
    pos = np.stack((flux.forward_committor,
                    np.random.uniform(0, 1, size=len(flux.forward_committor)))).T
    pos = _plots_bindings.fruchterman_reingold(flux.net_flux, pos, update_dims=[1], iterations=150)
    # rescale so that y positions are between 0 and 1
    pos[:, 1] -= np.min(pos[:, 1])
    pos[:, 1] /= np.max(pos[:, 1])

    if minflux > 0:
        F[F < minflux] = 0.0

    if isinstance(state_labels, str) and state_labels == 'auto':
        # the first and last element correspond to A and B in ReactiveFlux
        state_labels = np.array([str(i) for i in range(flux.n_states)])
        state_labels[np.array(flux.source_states)] = "A"
        state_labels[np.array(flux.target_states)] = "B"

    plot = Network(F, pos=pos, state_labels=state_labels, state_sizes=state_sizes, state_colors=state_colors,
                   state_scale=state_scale, edge_scale=edge_scale, edge_labels=edge_labels,
                   edge_label_format=edge_label_format, edge_curvature=edge_curvature, cmap=cmap)

    ax = plot.plot(ax=ax, **textkwargs)
    if show_committor:
        ax.set_xlabel('Committor probability')
    return ax, pos
