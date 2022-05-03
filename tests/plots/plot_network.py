import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from deeptime.plots.network import Network
from deeptime.markov.msm import MarkovStateModel

def test_sanity():
    P = np.array([[0.8,  0.15, 0.05,  0.0,  0.0],
                  [0.1,  0.75, 0.05, 0.05, 0.05],
                  [0.05,  0.1,  0.8,  0.0,  0.05],
                  [0.0,  0.2, 0.0,  0.8,  0.0],
                  [0.0,  0.02, 0.02, 0.0,  0.96]])

    flux = MarkovStateModel(P).reactive_flux([2], [3])

    positions = nx.planar_layout(nx.from_numpy_array(flux.gross_flux))
    pl = Network(flux.gross_flux, positions)

    f, (ax1, ax2) = plt.subplots(1, 2)

    ax = pl.plot(ax=ax1, state_colors=np.linspace(0, 1, num=flux.n_states), edge_curvature=2.)
    ax.set_aspect('equal')

    from pyemma.plots import plot_network

    plot_network(flux.gross_flux, pos=np.array([positions[i] for i in range(flux.n_states)]), ax=ax2)
    plt.show()

