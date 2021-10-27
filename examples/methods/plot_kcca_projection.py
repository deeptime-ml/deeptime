"""
Kernel CCA on the sqrt-Model to transform data
==============================================

This example shows an application of :class:`KernelCCA <deeptime.decomposition.KernelCCA>` on the
:meth:`sqrt model <deeptime.data.sqrt_model>` dataset. We transform the data by evaluating the estimated eigenfunctions
into a (quasi) linearly separable space. Crisp assignments are obtained by :class:`KMeans <deeptime.cluster.KMeans>`
clustering.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from deeptime.clustering import KMeans
from deeptime.data import sqrt_model
from deeptime.decomposition import KernelCCA
from deeptime.kernels import GaussianKernel

dtraj, obs = sqrt_model(1500)
dtraj_test, obs_test = sqrt_model(5000)

kernel = GaussianKernel(2.)
est = KernelCCA(kernel, n_eigs=2)
model = est.fit((obs[1:], obs[:-1])).fetch_model()
evals = model.transform(obs_test)
clustering = KMeans(2).fit(np.real(model.transform(obs))).fetch_model()
assignments = clustering.transform(np.real(evals))

n_mismatch = np.sum(np.abs(assignments - dtraj_test))
assignments_perm = np.where((assignments == 0) | (assignments == 1), assignments ^ 1, assignments)
n_mismatch_perm = np.sum(np.abs(assignments_perm - dtraj_test))

if n_mismatch_perm < n_mismatch:
    assignments = assignments_perm
    n_mismatch = n_mismatch_perm

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 14))
ax1.set_title(f"Discrete states vs. estimated discrete states,\n"
              f"{(len(dtraj_test) - n_mismatch) / len(dtraj_test):.3f}% correctly assigned")
ax1.plot(assignments[:150], label="Estimated assignments")
ax1.plot(dtraj_test[:150], 'x', label="Ground truth")
ax1.set_xlabel("time")
ax1.set_ylabel("state")
ax1.legend()


def plot_scatter(ax, states, observations, obs_ref=None):
    mask = np.zeros(states.shape, dtype=bool)
    mask[np.where(states == 0)] = True
    if obs_ref is None:
        ax.scatter(*observations[mask].T, color='green', label='State 1')
        ax.scatter(*observations[~mask].T, color='blue', label='State 2')
        ax.legend()
    else:
        scatter1 = ax.scatter(*observations[mask].T, cmap=mpl.cm.get_cmap('Greens'), c=obs_ref[mask][:, 1])
        scatter2 = ax.scatter(*observations[~mask].T, cmap=mpl.cm.get_cmap('Blues'), c=obs_ref[~mask][:, 1])
        h1, l1 = scatter1.legend_elements(num=1)
        h2, l2 = scatter2.legend_elements(num=1)
        ax.add_artist(ax.legend(handles=h1 + h2, labels=["State 1", "State 2"]))


ax2.set_title("Observed test data colored by estimated state assignment")
plot_scatter(ax2, assignments, obs_test)

ax3.set_title("Test data, colored by ground truth")
plot_scatter(ax3, dtraj_test, obs_test, obs_test)

ax4.set_title("Transformation of test data, colored by ground truth")
plot_scatter(ax4, dtraj_test, evals, obs_test)
