"""
Implied timescales
==================

This example demonstrates how to obtain an implied timescales (ITS) plot for a Bayesian Markov state model.
"""

import matplotlib.pyplot as plt
import numpy as np

from deeptime.clustering import KMeans
from deeptime.data import double_well_2d
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import BayesianMSM
from deeptime.plots import ImpliedTimescalesData, plot_implied_timescales

system = double_well_2d()
data = system.trajectory(x0=np.random.normal(scale=.2, size=(10, 2)), length=1000)
clustering = KMeans(n_clusters=50).fit_fetch(np.concatenate(data))

xs, ys = np.meshgrid(np.linspace(-1.5, 1.5, 100), np.linspace(-1, 1, 100))

x = np.arange(-2, 2, 0.01)
y = np.arange(-3, 3, 0.01)
xy = np.meshgrid(x, y)
V = system.potential(np.dstack(xy).reshape(-1, 2)).reshape(xy[0].shape)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title("Potential landscape")

cb = ax1.contourf(x, y, V, levels=40, cmap='coolwarm')
ax1.scatter(*clustering.cluster_centers.T, marker='x', color='black')


dtrajs = [clustering.transform(traj) for traj in data]

models = []
lagtimes = np.arange(1, 10)
for lagtime in lagtimes:
    counts = TransitionCountEstimator(lagtime=lagtime, count_mode='effective').fit_fetch(dtrajs)
    models.append(BayesianMSM(n_samples=50).fit_fetch(counts))

its_data = ImpliedTimescalesData.from_models(models)
plot_implied_timescales(ax2, its_data, n_its=2)
ax2.set_yscale('log')
ax2.set_title('Implied timescales')
