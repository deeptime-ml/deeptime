import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from deeptime.plots.network import Network, plot_markov_model
from deeptime.markov.msm import MarkovStateModel

matplotlib.use('Agg')


def test_network():
    P = np.array([[0.8, 0.15, 0.05, 0.0, 0.0],
                  [0.1, 0.75, 0.05, 0.05, 0.05],
                  [0.05, 0.1, 0.8, 0.0, 0.05],
                  [0.0, 0.2, 0.0, 0.8, 0.0],
                  [0.0, 0.02, 0.02, 0.0, 0.96]])
    from scipy import sparse
    Psparse = sparse.csr_matrix(P)

    flux = MarkovStateModel(Psparse).reactive_flux([2], [3])

    positions = nx.planar_layout(nx.from_scipy_sparse_matrix(flux.gross_flux))
    labels = np.array([["juhu"] * flux.n_states] * flux.n_states)
    pl = Network(flux.gross_flux, positions, edge_curvature=2., edge_labels=labels,
                 state_colors=np.linspace(0, 1, num=flux.n_states))

    f, (ax1, ax2) = plt.subplots(1, 2)

    ax = pl.plot(ax=ax1)
    ax.set_aspect('equal')

    from pyemma.plots import plot_network

    flux = MarkovStateModel(P).reactive_flux([2], [3])
    plot_network(flux.gross_flux, pos=np.array([positions[i] for i in range(flux.n_states)]), ax=ax2)

    # test against seaborn cmap


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
