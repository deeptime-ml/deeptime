import math
from typing import Optional

import numpy as np
from scipy.sparse import issparse

from deeptime.util.decorators import plotting_function


def _draw_arrow(
        ax, x1, y1, x2, y2, Dx, Dy, label="", width=1.0, arrow_curvature=1.0, color="grey",
        patchA=None, patchB=None, shrinkA=0, shrinkB=0, arrow_label_size=None):
    """
    Draws a slightly curved arrow from (x1,y1) to (x2,y2).
    Will allow the given patches at start end end.

    """
    # set arrow properties
    dist = math.sqrt(((x2 - x1) / float(Dx)) ** 2 + ((y2 - y1) / float(Dy)) ** 2)
    arrow_curvature *= 0.075  # standard scale
    rad = arrow_curvature / dist
    tail_width = width
    head_width = max(0.5, 2 * width)
    head_length = head_width
    ax.annotate(
        "", xy=(x2, y2), xycoords='data', xytext=(x1, y1), textcoords='data',
        arrowprops=dict(
            arrowstyle='simple,head_length=%f,head_width=%f,tail_width=%f' % (
                head_length, head_width, tail_width),
            color=color, shrinkA=shrinkA, shrinkB=shrinkB, patchA=patchA, patchB=patchB,
            connectionstyle="arc3,rad=%f" % -rad),
        zorder=0)
    # weighted center position
    center = np.array([0.55 * x1 + 0.45 * x2, 0.55 * y1 + 0.45 * y2])
    v = np.array([x2 - x1, y2 - y1])  # 1->2 vector
    vabs = np.abs(v)
    vnorm = np.array([v[1], -v[0]])  # orthogonal vector
    vnorm = np.divide(vnorm, np.linalg.norm(vnorm))  # normalize
    # cross product to determine the direction into which vnorm points
    z = np.cross(v, vnorm)
    if z < 0:
        vnorm *= -1
    offset = 0.5 * arrow_curvature * \
             ((vabs[0] / (vabs[0] + vabs[1]))
              * Dx + (vabs[1] / (vabs[0] + vabs[1])) * Dy)
    ptext = center + offset * vnorm
    ax.text(
        ptext[0], ptext[1], label, size=arrow_label_size,
        horizontalalignment='center', verticalalignment='center', zorder=1)


@plotting_function(requires_networkx=True)
def plot_adjacency(adjacency_matrix, positions: Optional[np.ndarray] = None, layout=None, ax=None, node_size=None,
                   self_loops=False, curved=True):
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

    if not self_loops:
        graph.remove_edges_from(nx.selfloop_edges(graph))
    nx.draw_networkx_nodes(graph, pos, node_size=node_size, ax=ax)
    if curved:
        Dx = max(x[0] for x in pos.values()) - min(x[0] for x in pos.values())
        Dy = max(x[1] for x in pos.values()) - min(x[1] for x in pos.values())
        edges = graph.edges()
        for (e1, e2) in edges:
            _draw_arrow(ax, *pos[e1].T, *pos[e2].T, Dx, Dy)
    else:
        nx.draw_networkx_edges(graph, pos, ax=ax)
    return ax, graph
