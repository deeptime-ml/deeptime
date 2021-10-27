"""
VAMP vs. TICA vs. PCA
=====================

This example directly reflects the example used in the
`TICA documentation <../notebooks/tica.ipynb>`__ plus a VAMP projection.
Since this data stems from an in-equilibrium distribution, TICA and VAMP should not show qualitative differences.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from deeptime.data import ellipsoids
from deeptime.decomposition import VAMP, TICA


def plot_dominant_component(ax, dxy, title):
    x, y = np.meshgrid(
        np.linspace(np.min(feature_trajectory[:, 0]), np.max(feature_trajectory[:, 0]), 4),
        np.linspace(np.min(feature_trajectory[:, 1]), np.max(feature_trajectory[:, 1]), 4)
    )
    ax.scatter(*feature_trajectory.T, marker='.')
    ax.quiver(x, y, dxy[0], dxy[1])
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')


data = ellipsoids(seed=17)
discrete_trajectory = data.discrete_trajectory(n_steps=1000)
feature_trajectory = data.map_discrete_to_observations(discrete_trajectory)

vamp = VAMP(dim=1, lagtime=1)
vamp = vamp.fit(feature_trajectory).fetch_model()
vamp_projection = vamp.transform(feature_trajectory)
dxy_vamp = vamp.singular_vectors_left[:, 0]  # dominant vamp component

tica = TICA(dim=1, lagtime=1)
tica = tica.fit(feature_trajectory).fetch_model()
tica_projection = tica.transform(feature_trajectory)
dxy_tica = tica.singular_vectors_left[:, 0]  # dominant tica component

pca = PCA(n_components=1)
pca.fit(feature_trajectory)
pca_projection = pca.transform(feature_trajectory)
dxy_pca = pca.components_[0]  # dominant pca component

f = plt.figure(constrained_layout=False, figsize=(14, 14))
gs = f.add_gridspec(nrows=2, ncols=3)
ax_projections = f.add_subplot(gs[0, :])
ax_tica = f.add_subplot(gs[1, 0])
ax_vamp = f.add_subplot(gs[1, 1])
ax_pca = f.add_subplot(gs[1, 2])

ax_projections.set_title("Projections of two-dimensional trajectory")
ax_projections.set_xlabel('x')
ax_projections.set_ylabel('t')

ax_projections.plot(pca_projection, label='PCA', alpha=.5)
ax_projections.plot(tica_projection, label='TICA')
ax_projections.plot(vamp_projection, label='VAMP', linestyle='dashed')
ax_projections.legend()

plot_dominant_component(ax_pca, dxy_pca, 'Samples with dominant PCA component')
plot_dominant_component(ax_tica, dxy_tica, 'Samples with dominant TICA component')
plot_dominant_component(ax_vamp, dxy_vamp, 'Samples with dominant VAMP component')
