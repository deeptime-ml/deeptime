r"""
Quadruple-well
==============

Example for the :meth:`deeptime.data.quadruple_well` dataset.
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from deeptime.data import quadruple_well

system = quadruple_well(n_steps=1000)
traj = system.trajectory(np.array([[1, -1]]), 100, seed=46)

xy = np.arange(-2, 2, 0.1)
coords = np.dstack(np.meshgrid(xy, xy)).reshape(-1, 2)
V = system.potential(coords).reshape((xy.shape[0], xy.shape[0]))

fig, ax = plt.subplots(1, 1)
ax.set_title("Example of a trajectory in the potential landscape")

cb = ax.contourf(xy, xy, V, levels=np.linspace(0.0, 3.0, 20), cmap='coolwarm')

x = np.r_[traj[:, 0]]
y = np.r_[traj[:, 1]]
f, u = scipy.interpolate.splprep([x, y], s=0, per=False)
xint, yint = scipy.interpolate.splev(np.linspace(0, 1, 50000), f)

points = np.stack([xint, yint]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
coll = LineCollection(segments, cmap='bwr')
coll.set_array(np.linspace(0, 1, num=len(points), endpoint=True))
coll.set_linewidth(1)
ax.add_collection(coll)

fig.colorbar(cb)
