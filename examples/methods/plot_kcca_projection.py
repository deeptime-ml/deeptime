"""
Kernel CCA on the sqrt-Model to transform data
==============================================

This example shows an application of :class:`KernelCCA <deeptime.decomposition.KernelCCA>` on the
:meth:`sqrt model <deeptime.data.sqrt_model>` dataset. We transform the data by evaluating the estimated eigenfunctions
into a (quasi) linearly separable space. Crisp assignments are obtained by :class:`Kmeans <deeptime.cluster.Kmeans>`
clustering.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import deeptime as dt

dtraj, obs = dt.data.sqrt_model(1500)
dtraj_test, obs_test = dt.data.sqrt_model(5000)

kernel = dt.kernels.GaussianKernel(2.)
est = dt.decomposition.KernelCCA(kernel, 2)
model = est.fit((obs[1:], obs[:-1])).fetch_model()
evals = model.transform(obs_test)
assignments = dt.clustering.Kmeans(2).fit(np.real(evals)).transform(np.real(evals))

n_mismatch = np.sum(np.abs(assignments - dtraj_test))
assignments_perm = np.where((assignments == 0) | (assignments == 1), assignments ^ 1, assignments)
n_mismatch_perm = np.sum(np.abs(assignments_perm - dtraj_test))

if n_mismatch_perm < n_mismatch:
    assignments = assignments_perm
    n_mismatch = n_mismatch_perm

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 10))
ax1.set_title(f"Discrete states vs. estimated discrete states,\n"
              f"{(len(dtraj_test) - n_mismatch) / len(dtraj_test):.3f}% correctly assigned")
ax1.plot(assignments[:150], label="Estimated assignments")
ax1.plot(dtraj_test[:150], 'x', label="Ground truth")
ax1.set_xlabel("time")
ax1.set_ylabel("state")
ax1.legend()


def plot_scatter(ax, states, observations):
    mask = np.zeros(states.shape, dtype=bool)
    mask[np.where(states == 0)] = True
    ax.scatter(*observations[mask].T, color=mpl.cm.viridis(0), label='State 1')
    ax.scatter(*observations[~mask].T, color=mpl.cm.viridis(.999), label='State 2')
    ax.legend()


ax2.set_title("Transformation of test data, colored by ground truth")
plot_scatter(ax2, dtraj_test, evals)

ax3.set_title("Observed test data colored by estimated state assignment")
plot_scatter(ax3, assignments, obs_test)
