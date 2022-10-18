r"""
Custom SDEs
===========

Demonstrating usage of :meth:`deeptime.data.custom_sde`.

One is able to define SDEs of the form

.. math::

    \mathrm{d}X_t = F(X_t) \mathrm{d}t + \sigma\mathrm{d}W_t

for :math:`X_t\in\mathbb{R}^d`, :math:`d\in\{1,2,3,4,5\}`.
"""

import matplotlib.pyplot as plt
import numpy as np

from deeptime.data import custom_sde


def harmonic_sphere_energy(x, radius=.5, k=1.):
    dist_to_origin = np.linalg.norm(x, axis=-1)
    dist_to_sphere = dist_to_origin - radius
    energy = np.zeros((len(x),))
    ixs = np.argwhere(dist_to_sphere > 0)[:, 0]
    energy[ixs] = 0.5 * k * dist_to_sphere[ixs] ** 2
    return energy


def harmonic_sphere_force(x, radius=.5, k=1.):
    dist_to_origin = np.linalg.norm(x)
    dist_to_sphere = dist_to_origin - radius
    if dist_to_sphere > 0:
        return -k * dist_to_sphere * np.array(x) / dist_to_origin
    else:
        return [0., 0.]


sde = custom_sde(dim=2, rhs=lambda x: harmonic_sphere_force(x, radius=.5, k=1),
                 sigma=np.diag([1., 1.]), h=1e-3, n_steps=100)
traj = sde.trajectory([[0., 0.]], 500)

xy = np.arange(-3.5, 3.5, 0.1)
coords = np.dstack(np.meshgrid(xy, xy)).reshape(-1, 2)
potential_landscape = harmonic_sphere_energy(coords).reshape((xy.shape[0], xy.shape[0]))
cb = plt.contourf(xy, xy, potential_landscape, levels=50, cmap='coolwarm')
plt.colorbar(cb)

plt.plot(*traj.T, color='black')
