"""
Gross and net flux on the Drunkard's walk example
=================================================

This example shows how to compute and visualize gross and net reactive flux (see :class:`deeptime.markov.ReactiveFlux`)
using the :meth:`deeptime.data.drunkards_walk`.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

from deeptime.data import drunkards_walk

sim = drunkards_walk(grid_size=(10, 10),
                     bar_location=[(0, 0), (0, 1), (1, 0), (1, 1)],
                     home_location=[(8, 8), (8, 9), (9, 8), (9, 9)])
sim.add_barrier((5, 1), (5, 5))
sim.add_barrier((0, 9), (5, 8))
sim.add_barrier((9, 2), (7, 6))
sim.add_barrier((2, 6), (5, 6))

sim.add_barrier((7, 9), (7, 7), weight=5.)
sim.add_barrier((8, 7), (9, 7), weight=5.)

sim.add_barrier((0, 2), (2, 2), weight=5.)
sim.add_barrier((2, 0), (2, 1), weight=5.)

flux = sim.msm.reactive_flux(sim.home_state, sim.bar_state)

fig, axes = plt.subplots(1, 2, figsize=(18, 10))
dividers = [make_axes_locatable(axes[i]) for i in range(len(axes))]
caxes = [divider.append_axes("right", size="5%", pad=0.05) for divider in dividers]

titles = ["Gross flux", "Net flux"]
fluxes = [flux.gross_flux, flux.net_flux]

cmap = plt.cm.copper_r
thresh = [0, 1e-12]

for i in range(len(axes)):
    ax = axes[i]
    F = fluxes[i]
    ax.set_title(titles[i])

    vmin = np.min(F[np.nonzero(F)])
    vmax = np.max(F)

    sim.plot_2d_map(ax)
    sim.plot_network(ax, F, cmap=cmap, connection_threshold=thresh[i])
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=caxes[i])
