r"""
Asymmetric Quadruple-well
=========================

Example for the :meth:`deeptime.data.quadruple_well_asymmetric` dataset.
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from deeptime.data import quadruple_well_asymmetric

system = quadruple_well_asymmetric(n_steps=10000)
traj = system.trajectory(np.array([[-1, 1]]), 70, seed=36)

xy = np.arange(-1.8, 1.8, 0.1)
coords = np.dstack(np.meshgrid(xy, xy)).reshape(-1, 2)
V = system.potential(coords).reshape((xy.shape[0], xy.shape[0]))

fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal')
ax.set_title("Example of a trajectory")

cb = ax.contourf(xy, xy, V, levels=50, cmap='coolwarm')

ax.annotate(f'V(x) = {system.potential([[1., 1.]]).squeeze():.2f}', xy=(1., 1.), xycoords='data',
            xytext=(-30, 40), textcoords='offset points',
            bbox=dict(boxstyle="round"), arrowprops=dict(arrowstyle='simple'))
ax.annotate(f'V(x) = {system.potential([[-1., -1.]]).squeeze():.2f}', xy=(-1., -1.), xycoords='data',
            xytext=(-30, -40), textcoords='offset points',
            bbox=dict(boxstyle="round"), arrowprops=dict(arrowstyle='simple'))
ax.annotate(f'V(x) = {system.potential([[-1., 1.]]).squeeze():.2f}', xy=(-1., 1.), xycoords='data',
            xytext=(-30, 40), textcoords='offset points',
            bbox=dict(boxstyle="round"), arrowprops=dict(arrowstyle='simple'))
ax.annotate(f'V(x) = {system.potential([[1., -1.]]).squeeze():.2f}', xy=(1., -1.), xycoords='data',
            xytext=(-30, -40), textcoords='offset points',
            bbox=dict(boxstyle="round"), arrowprops=dict(arrowstyle='simple'))

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
