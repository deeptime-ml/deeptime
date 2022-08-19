"""
Plotting two-dimensional densities from xy
==========================================

This example demonstrates how to plot unordered xy data - in this case, particle positions (xy) - as contour of their
density in both linear and log scale. See :meth:`deeptime.plots.plot_density`.
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import ticker

from deeptime.data import quadruple_well
from deeptime.plots import plot_density

system = quadruple_well(h=1e-3, n_steps=100)
trajs = system.trajectory(x0=[[-1, 0], [1, 0], [0, 0]], length=5000)
traj_concat = np.concatenate(trajs, axis=0)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

ax1.set_title('Density')
ax1, mappable1 = plot_density(*traj_concat.T, n_bins=20, contourf_kws=dict(), ax=ax1)
f.colorbar(mappable1, ax=ax1, format=ticker.FuncFormatter(lambda x, pos: f"{x:.3f}"))

ax2.set_title('Log Density')
contourf_kws = dict(locator=ticker.LogLocator(base=10, subs=range(1, 10)))
ax2, mappable2 = plot_density(*traj_concat.T, n_bins=20, avoid_zero_counts=True, contourf_kws=contourf_kws, ax=ax2)
f.colorbar(mappable2, ax=ax2, format=ticker.FuncFormatter(lambda x, pos: f"{x:.0e}"))
