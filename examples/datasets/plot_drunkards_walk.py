r"""
Drunkard's walk
===============

The :meth:`deeptime.data.drunkards_walk` model, a markov state model on a 2-dimensional grid.
"""

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

start = (7, 2)
walk = sim.walk(start=start, n_steps=250, seed=40)
print("Number of steps in the walk:", len(walk))

fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(*start, marker='*', label='Start', c='cyan', s=150, zorder=5)
sim.plot_path(ax, walk)
handles, labels = sim.plot_2d_map(ax)
ax.legend(handles=handles, labels=labels)
