r"""
One-dimensional Ornstein-Uhlenbeck process
==========================================

Example for the :meth:`deeptime.data.ornstein_uhlenbeck` dataset.
"""

import matplotlib.pyplot as plt
import deeptime as dt

traj = dt.data.ornstein_uhlenbeck().trajectory([[-0.]], 250)
plt.plot(traj.squeeze())
plt.xlabel('t')
plt.xlabel('x(t)')
plt.show()
