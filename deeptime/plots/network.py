from scipy.sparse import issparse

from deeptime.util.decorators import plotting_function


def _explicit_layout(graph, positions):
    return dict(zip(graph, positions))


@plotting_function(requires_networkx=True)
def plot_adjacency(adjacency_matrix, positions=None):
    import networkx as nx
    if issparse(adjacency_matrix):
        graph = nx.from_scipy_sparse_matrix(adjacency_matrix)
    else:
        graph = nx.from_numpy_matrix(adjacency_matrix)
    if positions is not None:
        layout = ...
