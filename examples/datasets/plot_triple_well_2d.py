r"""
Triple-well 2D
==============

Example for the :meth:`deeptime.data.triple_well_2d` dataset.

"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from deeptime.data import triple_well_2d

system = triple_well_2d(n_steps=10000)
traj = system.trajectory(np.array([[-1, 0]]), 20, seed=42)

x = np.arange(-2, 2, 0.01)
y = np.arange(-1, 2, 0.01)
xy = np.meshgrid(x, y)
V = system.potential(np.dstack(xy).reshape(-1, 2)).reshape(xy[0].shape)

fig, ax = plt.subplots(1, 1)
ax.set_title("Example of a trajectory in the potential landscape")

cb = ax.contourf(x, y, V, levels=20, cmap='coolwarm')

x = np.r_[traj[:, 0]]
y = np.r_[traj[:, 1]]
f, u = scipy.interpolate.splprep([x, y], s=0, per=False)
xint, yint = scipy.interpolate.splev(np.linspace(0, 1, 50000), f)

points = np.stack([xint, yint]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
coll = LineCollection(segments, cmap='jet')
coll.set_array(np.linspace(0, 1, num=len(points), endpoint=True))
coll.set_linewidth(1)
ax.add_collection(coll)

fig.colorbar(cb)
