r"""
Lorenz system
=============

The Lorenz system. See :meth:`deeptime.data.lorenz_system`.
"""

import deeptime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

system = deeptime.data.lorenz_system()
x0 = np.array([[8, 7, 15]])
traj = system.trajectory(x0, 3500)

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(*x0.T, color='blue', label=r"$t_\mathrm{start}$")
ax.scatter(*traj[-1].T, color='red', label=r"$t_\mathrm{final}$")

points = traj.reshape((-1, 1, 3))
segments = np.concatenate([points[:-1], points[1:]], axis=1)
coll = Line3DCollection(segments, cmap='coolwarm')
coll.set_array(np.linspace(0, 1, num=len(points), endpoint=True))
coll.set_linewidth(2)
ax.add_collection(coll)
ax.set_xlim3d((-19, 19))
ax.set_ylim3d((-26, 26))
ax.set_zlim3d((0, 45))
ax.set_box_aspect(np.ptp(traj, axis=0))
ax.legend()
