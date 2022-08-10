"""
Plotting two-dimensional contours from xyz
==========================================

This example demonstrates how to plot unordered xyz data - in this case, particle positions (xy) and their energy (z) -
as contour.
"""

import numpy as np

import matplotlib.pyplot as plt

from deeptime.data import triple_well_2d
from deeptime.plots import plot_contour2d_from_xyz

system = triple_well_2d(h=1e-3, n_steps=100)
trajs = system.trajectory(x0=[[-1, 0], [1, 0], [0, 0]], length=1000)
traj_concat = np.concatenate(trajs, axis=0)

energies = system.potential(traj_concat)

f, ax = plt.subplots(1, 1)
ax, mappable = plot_contour2d_from_xyz(*traj_concat.T, energies, contourf_kws=dict(cmap='coolwarm'), ax=ax)
f.colorbar(mappable)
