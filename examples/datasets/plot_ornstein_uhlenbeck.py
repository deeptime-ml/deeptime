r"""
One-dimensional Ornstein-Uhlenbeck process
==========================================

Example for the :meth:`deeptime.data.ornstein_uhlenbeck` dataset.
"""

import matplotlib.pyplot as plt

from deeptime.data import ornstein_uhlenbeck

traj = ornstein_uhlenbeck().trajectory([[-0.]], 250)
plt.plot(traj.squeeze())
plt.xlabel('t')
plt.xlabel('x(t)')
plt.show()
