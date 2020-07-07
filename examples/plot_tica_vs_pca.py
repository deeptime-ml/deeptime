"""
TICA vs. PCA
============

This example directly reflects the example used in the TICA documentation.
"""

import sktime
import matplotlib.pyplot as plt
import numpy as np

ellipsoids = sktime.data.ellipsoids(seed=17)
discrete_trajectory = ellipsoids.discrete_trajectory(n_steps=1000)
feature_trajectory = ellipsoids.map_discrete_to_observations(discrete_trajectory)

tica = sktime.decomposition.TICA(dim=1)
tica = tica.fit(feature_trajectory, lagtime=1).fetch_model()
tica_projection = tica.transform(feature_trajectory)

from sklearn.decomposition import PCA

pca = PCA(n_components=1)
pca.fit(feature_trajectory)
pca_projection = pca.transform(feature_trajectory)

f, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0][0].plot(tica_projection)
axes[0][0].set_title('TICA projection of 2-dimensional trajectory')
axes[0][0].set_xlabel('x')
axes[0][0].set_ylabel('t')

dxy = tica.singular_vectors_left[:, 0]  # dominant tica component

axes[0][1].scatter(*feature_trajectory.T, marker='.')
x, y = np.meshgrid(
    np.linspace(np.min(feature_trajectory[:, 0]), np.max(feature_trajectory[:, 0]), 4),
    np.linspace(np.min(feature_trajectory[:, 1]), np.max(feature_trajectory[:, 1]), 4)
)
axes[0][1].quiver(x, y, dxy[0], dxy[1])

axes[0][1].set_aspect('equal')
axes[0][1].set_xlabel('x')
axes[0][1].set_ylabel('y')
axes[0][1].set_title('Example data samples with dominant TICA component')

axes[1][0].plot(pca_projection)
axes[1][0].set_title('PCA projection of 2-dimensional trajectory')
axes[1][0].set_xlabel('x')
axes[1][0].set_ylabel('t')

dxy = pca.components_[0]  # dominant pca component

axes[1][1].scatter(*feature_trajectory.T, marker='.')
axes[1][1].quiver(x, y, dxy[0], dxy[1])

axes[1][1].set_aspect('equal')
axes[1][1].set_xlabel('x')
axes[1][1].set_ylabel('y')
axes[1][1].set_title('Example data samples with dominant PCA component')
