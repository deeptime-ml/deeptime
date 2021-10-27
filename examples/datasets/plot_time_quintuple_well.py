r"""
Time-dependent quintuple well
=============================

Demonstrates :meth:`deeptime.data.time_dependent_quintuple_well`. The potential wells slowly oscillate around the
origin.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from deeptime.data import time_dependent_quintuple_well

random_state = np.random.RandomState(33)

cmap = plt.cm.viridis

system = time_dependent_quintuple_well(h=1e-4, n_steps=100, beta=5)
x = np.arange(-2.5, 2.5, 0.1)
y = np.arange(-2.5, 2.5, 0.1)
xy = np.meshgrid(x, y)

x0 = random_state.uniform(-2.5, 2.5, size=(100, 2))
trajs = system.trajectory(0., x0, length=500)

l = []
for t in np.arange(0., 20., 0.01):
    V = system.potential(t, np.dstack(xy).reshape(-1, 2)).reshape(xy[0].shape)
    l.append(V)
l = np.stack(l)

vmin = np.min(l)
vmax = np.max(l)

fig, ax = plt.subplots()
ax.set_xlim([np.min(xy[0]), np.max(xy[0])])
ax.set_ylim([np.min(xy[1]), np.max(xy[1])])
handle = ax.contourf(*xy, l[0], vmin=vmin, vmax=vmax, cmap=cmap, levels=1000)
scatter_handle = ax.scatter(*trajs[:, 0].T, color='red', zorder=100)
handles = [scatter_handle, handle]


def update(i):
    out = [scatter_handle]
    handles[0].set_offsets(trajs[:, i])
    for tp in handles[1].collections:
        tp.remove()
    handles[1] = ax.contourf(*xy, l[i], vmin=vmin, vmax=vmax, cmap=cmap)
    out += handles[1].collections
    return out


ani = animation.FuncAnimation(fig, update, interval=50, blit=True, repeat=True, frames=trajs.shape[1])
plt.show()