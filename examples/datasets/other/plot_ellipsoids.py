r"""
Ellipsoids dataset
==================

The :meth:`deeptime.data.ellipsoids` dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

from deeptime.data import ellipsoids

data_source = ellipsoids(seed=17)
x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)
X, Y = np.meshgrid(x, y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
rv1 = multivariate_normal(data_source.state_0_mean, data_source.covariance_matrix)
rv2 = multivariate_normal(data_source.state_1_mean, data_source.covariance_matrix)

fig = plt.figure()
ax = fig.gca()

ax.contourf(X, Y, (rv1.pdf(pos) + rv2.pdf(pos)).reshape(len(x), len(y)))
ax.autoscale(False)
ax.set_aspect('equal')
ax.scatter(*data_source.observations(100).T, color='cyan', marker='x', label='samples')
plt.grid()
plt.title(r'Ellipsoids dataset observations with laziness of $0.97$.')
plt.legend()
