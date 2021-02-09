"""
Kernel CCA on the sqrt-Model to transform data
==============================================

This example shows an application of :class:`KernelCCA <deeptime.decomposition.KernelCCA>` on the
:meth:`sqrt model <deeptime.data.sqrt_model>` dataset. We transform the data by evaluating the estimated eigenfunctions
into a (quasi) linearly separable space. Crisp assignments are obtained by :class:`Kmeans <deeptime.cluster.Kmeans>`
clustering.
"""

import numpy as np
import matplotlib.pyplot as plt
import deeptime as dt

dtraj, obs = dt.data.sqrt_model(1500)

kernel = dt.kernels.GaussianKernel(2.)
est = dt.decomposition.KernelCCA(kernel, 2)
model = est.fit((obs[1:], obs[:-1])).fetch_model()
evals = model.transform(obs)
assignments = dt.clustering.Kmeans(2).fit(np.real(evals)).transform(np.real(evals))

n_mismatch = np.sum(np.abs(assignments - dtraj))
assignments_perm = np.where((assignments == 0) | (assignments == 1), assignments ^ 1, assignments)
n_mismatch_perm = np.sum(np.abs(assignments_perm - dtraj))

if n_mismatch_perm < n_mismatch:
    assignments = assignments_perm
    n_mismatch = n_mismatch_perm

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 10))
ax1.set_title(f"Discrete states vs. estimated discrete states,\n"
              f"{(len(dtraj) - n_mismatch) / len(dtraj):.3f}% correctly assigned")
ax1.plot(assignments[:150], label="Estimated assignments")
ax1.plot(dtraj[:150], 'x', label="Ground truth")
ax1.set_xlabel("time")
ax1.set_ylabel("state")
ax1.legend()

ax2.set_title("Transformation of original data, colored by ground truth")
ax2.scatter(*evals.T, c=dtraj)

ax3.set_title("Observed data colored by estimated state assignment")
ax3.scatter(*obs.T, c=assignments)
