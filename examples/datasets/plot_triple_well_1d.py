r"""
Triple-well 1D
==============

Example for the :meth:`deeptime.data.triple_well_1d` dataset.
"""

import matplotlib.pyplot as plt
import numpy as np

from deeptime.data import triple_well_1d

system = triple_well_1d(h=1e-3, n_steps=500)
xs = np.linspace(0, 6., num=500)
ys = system.potential(xs.reshape(-1, 1))

trajectory = system.trajectory(x0=0.5, length=20000, seed=53)

f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
ax1.plot(xs, ys.reshape(-1))
ax1.set_xlabel('x')
ax1.set_ylabel('V(x)')
ax1.set_title('Potential landscape')

ax2.hist(trajectory.reshape(-1), bins=50, density=True)
ax2.set_title('Histogram of trajectory')

plt.show()
