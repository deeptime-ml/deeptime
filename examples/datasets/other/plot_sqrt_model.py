r"""
Sqrt model
==========

Sample a hidden state and an sqrt-transformed emission trajectory. Demonstrates :meth:`deeptime.data.sqrt_model`.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from deeptime.data import sqrt_model

n_samples = 10000
dtraj, traj = sqrt_model(n_samples)

X, Y = np.meshgrid(
    np.linspace(np.min(traj[:, 0]), np.max(traj[:, 0]), 100),
    np.linspace(np.min(traj[:, 1]), np.max(traj[:, 1]), 100),
)
kde_input = np.dstack((X, Y)).reshape(-1, 2)

kernel = stats.gaussian_kde(traj.T, bw_method=.1)
Z = kernel(kde_input.T).reshape(X.shape)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.plot(dtraj[:500])
ax1.set_title('Discrete trajectory')
ax1.set_xlabel('time (a.u.)')
ax1.set_ylabel('state')

cm = ax2.contourf(X, Y, Z)
plt.colorbar(cm, ax=ax2)
ax2.set_title('Heatmap of observations')
