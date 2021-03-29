r"""
One-dimensional Prinz potential
===============================

Example for the :meth:`deeptime.data.prinz_potential` dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
from deeptime.data import prinz_potential

system = prinz_potential(n_steps=500, h=1e-5)
xs = np.linspace(-1, 1, 1000)
energy = system.potential(xs.reshape((-1, 1)))
traj = system.trajectory([[0.]], 3000)
dtraj = np.digitize(traj, bins=xs[:-1], right=False).squeeze()
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(energy, xs)
ax1.set_ylabel('x coordinate')
ax1.set_xlabel('energy(x)')
ax2.plot(xs[dtraj])
ax2.set_xlabel('time (a.u.)')
