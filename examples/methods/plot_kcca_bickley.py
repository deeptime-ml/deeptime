"""
Kernel CCA on the Bickley jet to find coherent sets
===================================================

This example shows an application of :class:`KernelCCA <deeptime.decomposition.KernelCCA>` on the
:meth:`bickley jet <deeptime.data.bickley_jet>` dataset. One can cluster in the singular function space
to find coherent structures.
"""
import matplotlib.pyplot as plt
import numpy as np

from deeptime.clustering import KMeans
from deeptime.data import bickley_jet
from deeptime.decomposition import KernelCCA
from deeptime.kernels import GaussianKernel

dataset = bickley_jet(n_particles=1000, n_jobs=8).endpoints_dataset()
kernel = GaussianKernel(.7)

estimator = KernelCCA(kernel, n_eigs=5, epsilon=1e-3)
model = estimator.fit((dataset.data, dataset.data_lagged)).fetch_model()

ev_real = np.real(model.eigenvectors)
kmeans = KMeans(n_clusters=7, n_jobs=8).fit(ev_real)
kmeans = kmeans.fetch_model()

fig = plt.figure()
gs = fig.add_gridspec(ncols=2, nrows=3)

ax = fig.add_subplot(gs[0, 0])
ax.scatter(*dataset.data.T, c=ev_real[:, 0])
ax.set_title('1st Eigenfunction')

ax = fig.add_subplot(gs[0, 1])
ax.scatter(*dataset.data.T, c=ev_real[:, 1])
ax.set_title('2nd Eigenfunction')

ax = fig.add_subplot(gs[1, 0])
ax.scatter(*dataset.data.T, c=ev_real[:, 2])
ax.set_title('3rd Eigenfunction')

ax = fig.add_subplot(gs[1, 1])
ax.scatter(*dataset.data.T, c=ev_real[:, 3])
ax.set_title('4th Eigenfunction')

ax = fig.add_subplot(gs[2, 0])
ax.scatter(*dataset.data.T, c=ev_real[:, 4])
ax.set_title('5th Eigenfunction')

ax = fig.add_subplot(gs[2, 1])
ax.scatter(*dataset.data.T, c=kmeans.transform(ev_real))
ax.set_title('Clustering of the eigenfunctions')
