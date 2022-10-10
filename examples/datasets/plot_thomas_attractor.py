r"""
Thomas attractor
================

The Thomas attractor. See :meth:`deeptime.data.thomas_attractor`.
"""

import deeptime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

system = deeptime.data.thomas_attractor(.15)
x0 = np.array([[0, 0, -3]])
traj = system.trajectory(x0, 500)

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(*x0.T, color='blue', label=r"$t_\mathrm{start}$")
ax.scatter(*traj[-1].T, color='red', label=r"$t_\mathrm{final}$")

points = traj.reshape((-1, 1, 3))
segments = np.concatenate([points[:-1], points[1:]], axis=1)
coll = Line3DCollection(segments, cmap='jet')
coll.set_array(np.linspace(0, 1, num=len(points), endpoint=True))
coll.set_linewidth(1)
ax.add_collection(coll)

ax.plot(*traj.T, alpha=0)
ax.set_box_aspect(np.ptp(traj, axis=0))
ax.legend()
