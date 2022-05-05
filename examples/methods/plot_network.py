"""
Network plots
=============

We demonstrate different kinds of network plots based on :meth:`plots.Network <deeptime.plots.Network>`.
In particular:

    * plotting a Markov state model where the state sizes depend on the stationary distribution and edges are scaled
      according to jump probabilities (:meth:`deeptime.plots.plot_markov_model`)
    * plotting the gross flux, in accordance to edge widths and colored according to the forward committor
      (:meth:`deeptime.plots.Network`).
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

from mpl_toolkits.axes_grid1 import make_axes_locatable

from deeptime.markov.msm import MarkovStateModel
from deeptime.plots import plot_markov_model, Network, plot_flux

P = np.array([[0.8, 0.15, 0.05, 0.0, 0.0],
              [0.1, 0.75, 0.05, 0.05, 0.05],
              [0.05, 0.1, 0.8, 0.0, 0.05],
              [0.0, 0.2, 0.0, 0.8, 0.0],
              [1e-7, 0.02 - 1e-7, 0.02, 0.0, 0.96]])

f, axes = plt.subplots(2, 2, figsize=(16, 12))
for ax in axes.flatten():
    ax.set_aspect('equal')

ax = axes[0][0]
ax.set_title('Plotting the Markov model')
plot_markov_model(P, ax=ax)

ax = axes[0][1]
ax.set_title('Plotting the gross flux')
flux = MarkovStateModel(P).reactive_flux(source_states=[2], target_states=[3])
positions = nx.planar_layout(nx.from_numpy_array(flux.gross_flux))
cmap = mpl.cm.get_cmap('coolwarm')
network = Network(flux.gross_flux, positions, edge_curvature=2.,
                  state_colors=flux.forward_committor, cmap=cmap)
network.plot(ax=ax)
norm = mpl.colors.Normalize(vmin=np.min(flux.forward_committor), vmax=np.max(flux.forward_committor))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
f.colorbar(mpl.cm.ScalarMappable(norm, cmap), cax=cax)

ax = axes[1][0]
ax.set_title('Plotting the net flux')
ax.get_yaxis().set_visible(False)
plot_flux(flux, attribute_to_plot='net_flux', ax=ax)

f.delaxes(axes[1][1])
