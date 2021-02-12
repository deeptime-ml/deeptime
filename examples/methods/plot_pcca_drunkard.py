"""
PCCA+ on the Drunkard's walk example
====================================

This example shows a decomposition into metastable sets (see :meth:`deeptime.markov.pcca`) of states
in the :meth:`deeptime.data.drunkards_walk` example.
The state assignments are shown via their probability distributions over the micro states.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

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

pcca = sim.msm.pcca(6)

fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    ax.set_title(f"Memberships for metastable set {i + 1}")
    handles, labels = sim.plot_2d_map(ax, barrier_mode='hollow')

    Q = pcca.memberships[:, i].reshape(sim.grid_size)
    cb = ax.imshow(Q, interpolation='nearest', origin='lower', cmap=plt.cm.Blues)
norm = mpl.colors.Normalize(vmin=0, vmax=1)
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Blues), ax=axes, shrink=.8)
