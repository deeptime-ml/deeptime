import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pytest
from numpy.testing import assert_raises

from deeptime.plots.network import Network, plot_markov_model
from deeptime.markov.msm import MarkovStateModel

matplotlib.use('Agg')


@pytest.mark.parametrize("cmap", ["nipy_spectral", plt.cm.nipy_spectral, None])
@pytest.mark.parametrize("labels", [None, np.array([["juhu"] * 5] * 5), 'weights'], ids=['None', 'strs', 'weights'])
def test_network(labels, cmap):
    P = np.array([[0.8, 0.15, 0.05, 0.0, 0.0],
                  [0.1, 0.75, 0.05, 0.05, 0.05],
                  [0.05, 0.1, 0.8, 0.0, 0.05],
                  [0.0, 0.2, 0.0, 0.8, 0.0],
                  [0.0, 0.02, 0.02, 0.0, 0.96]])
    from scipy import sparse
    Psparse = sparse.csr_matrix(P)

    flux = MarkovStateModel(Psparse).reactive_flux([2], [3])

    positions = nx.planar_layout(nx.from_scipy_sparse_matrix(flux.gross_flux))
    pl = Network(flux.gross_flux, positions, edge_curvature=2., edge_labels=labels,
                 state_colors=np.linspace(0, 1, num=flux.n_states), cmap=cmap)
    ax = pl.plot()
    ax.set_aspect('equal')


def test_network_invalid_args():
    msm = MarkovStateModel(np.eye(3))
    with assert_raises(ValueError):
        Network(msm.transition_matrix, pos=np.zeros((2, 2)))  # not enough positions
    network = Network(msm.transition_matrix, pos=np.zeros((3, 2)))
    with assert_raises(ValueError):
        network.edge_labels = np.array([["hi"]*2]*2)  # not enough labels
    with assert_raises(ValueError):
        network.edge_labels = 'bogus'  # may be 'weights'
    network.state_sizes = [1., 2., 3.]
    with assert_raises(ValueError):
        network.state_sizes = [1., 2.]
    network.state_labels = ["1", "2", "3"]
    with assert_raises(ValueError):
        network.state_labels = ["1", "2"]
    network.state_colors = np.random.uniform(size=(3, 3))
    network.state_colors = np.random.uniform(size=(3, 4))
    with assert_raises(ValueError):
        network.state_colors = np.random.uniform(size=(3, 5))
    with assert_raises(ValueError):
        network.state_colors = np.random.uniform(size=(2, 4))
    network.plot()


def test_msm():
    P = np.array([[0.8, 0.15, 0.05, 0.0, 0.0],
                  [0.1, 0.75, 0.05, 0.05, 0.05],
                  [0.05, 0.1, 0.8, 0.0, 0.05],
                  [0.0, 0.2, 0.0, 0.8, 0.0],
                  [1e-7, 0.02 - 1e-7, 0.02, 0.0, 0.96]])
    from scipy import sparse
    Psparse = sparse.csr_matrix(P)

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')

    plot_markov_model(Psparse, ax=ax1)
    plot_markov_model(P, ax=ax2)
