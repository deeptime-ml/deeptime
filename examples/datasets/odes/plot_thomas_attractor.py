r"""
Thomas attractor
================

The Thomas attractor. See :meth:`deeptime.data.thomas_attractor`.
"""

import deeptime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

system = deeptime.data.thomas_attractor(.1)

x0 = np.random.uniform(-3, 3, size=(50, 3))
traj = system.trajectory(x0, 500)
tstart = 0
tfinish = system.h * system.n_steps * len(traj)

ax = plt.figure().add_subplot(projection='3d')

for t in traj:
    points = t.reshape((-1, 1, 3))
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    coll = Line3DCollection(segments, cmap='twilight', alpha=.3)
    coll.set_array(t[:, -1])
    coll.set_linewidth(1)
    ax.add_collection(coll)

ax.plot(*traj[0].T, alpha=0)
ax.set_box_aspect(np.ptp(traj[0], axis=0))
