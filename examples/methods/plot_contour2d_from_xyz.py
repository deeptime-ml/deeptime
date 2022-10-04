"""
2D contours from xyz
====================

This example demonstrates how to plot unordered xyz data - in this case, particle positions (xy) and their energy (z) -
as contour as well as a state map on the right-hand side depicting a decomposition into three coarse metastable states.
See :meth:`deeptime.plots.plot_contour2d_from_xyz`.
"""

import numpy as np

import matplotlib.pyplot as plt

from deeptime.clustering import KMeans
from deeptime.markov.msm import MaximumLikelihoodMSM

from deeptime.data import triple_well_2d
from deeptime.plots import plot_contour2d_from_xyz

system = triple_well_2d(h=1e-3, n_steps=100)
trajs = system.trajectory(x0=[[-1, 0], [1, 0], [0, 0]], length=5000)
traj_concat = np.concatenate(trajs, axis=0)

energies = system.potential(traj_concat)

clustering = KMeans(n_clusters=20).fit_fetch(traj_concat)
dtrajs = [clustering.transform(traj) for traj in trajs]

msm = MaximumLikelihoodMSM(lagtime=1).fit(dtrajs).fetch_model()
pcca = msm.pcca(n_metastable_sets=3)
coarse_states = pcca.assignments[np.concatenate(dtrajs)]

f, (ax1, ax2) = plt.subplots(1, 2)
ax1, mappable = plot_contour2d_from_xyz(*traj_concat.T, energies, contourf_kws=dict(cmap='coolwarm'), ax=ax1)
ax2, _ = plot_contour2d_from_xyz(*traj_concat.T, coarse_states, n_bins=200, ax=ax2)
f.colorbar(mappable, ax=ax1)
