"""
Energy surface
==============

We show how to plot a two-dimensional energy surface based on :meth:`util.energy2d <deeptime.util.energy2d>`
and :meth:`plots.plot_energy2d <deeptime.plots.plot_energy2d>`.
"""
import numpy as np

from deeptime.data import triple_well_2d
from deeptime.clustering import KMeans
from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.util import energy2d
from deeptime.plots import plot_energy2d

trajs = triple_well_2d(h=1e-3, n_steps=100).trajectory(x0=[[-1, 0], [1, 0], [0, 0]], length=5000)
traj_concat = np.concatenate(trajs, axis=0)
clustering = KMeans(n_clusters=20).fit_fetch(traj_concat)
dtrajs = [clustering.transform(traj) for traj in trajs]
msm = MaximumLikelihoodMSM(lagtime=1).fit_fetch(dtrajs)
weights = msm.compute_trajectory_weights(np.concatenate(dtrajs))[0]

energies = energy2d(*traj_concat.T, bins=(80, 20), kbt=1, weights=weights, shift_energy=True)
plot = plot_energy2d(energies, contourf_kws=dict(cmap='nipy_spectral'))
plot.colorbar.set_label('energy / kJ')
