r"""
Custom ODEs
===========

Demonstrating usage of :meth:`deeptime.data.custom_ode`.

One is able to define ODEs of the form

.. math::

    \mathrm{d}X_t = F(X_t)

for :math:`X_t\in\mathbb{R}^d`, :math:`d\in\{1,2,3,4,5\}`.
"""
import matplotlib.pyplot as plt
import numpy as np

from deeptime.data import custom_ode

h = 1e-1
n_steps = 2
n_evals = 50

final_time = h * n_steps * (n_evals-1)

ode = custom_ode(dim=1, rhs=lambda x: [-.5 * x[0]], h=h, n_steps=n_steps)
traj = ode.trajectory(x0=1., length=n_evals)

xs = np.linspace(0, final_time, num=n_evals)
plt.plot(xs, traj, 'x')

plt.plot(xs, np.exp(-.5 * xs))
