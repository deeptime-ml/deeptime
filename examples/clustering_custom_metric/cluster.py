import deeptime as dt
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import bindings

# register maxnorm
dt.clustering.metrics.register('max', bindings.maxnorm)

rnd = np.random.RandomState(seed=17)

# create a GMM object
gmm = GaussianMixture(n_components=1, random_state=rnd, covariance_type='diag')
gmm.weights_ = np.array([1])
gmm.means_ = rnd.uniform(low=-20., high=20., size=(1, 2))
gmm.covariances_ = rnd.uniform(low=15., high=18., size=(1, 2))

samples, labels = gmm.sample(5000)

kmeans_euclidean = dt.clustering.KMeans(
    n_clusters=5,
    fixed_seed=13,
    n_jobs=8,
    metric='euclidean'
).fit_fetch(samples)

kmeans_maxnorm = dt.clustering.KMeans(
    n_clusters=5,
    fixed_seed=13,
    n_jobs=8,
    metric='max'
).fit_fetch(samples)

f, (ax1, ax2) = plt.subplots(1, 2)

assignments = kmeans_euclidean.transform(samples)
ax1.scatter(*samples.T, c=assignments, s=3)
ax1.scatter(*kmeans_euclidean.cluster_centers.T, marker='*', c='black')
ax1.set_title('Euclidean metric')

assignments = kmeans_maxnorm.transform(samples)
ax2.scatter(*samples.T, c=assignments, s=3)
ax2.scatter(*kmeans_maxnorm.cluster_centers.T, marker='*', c='black')
ax2.set_title('Maxnorm induced metric')

plt.show()
