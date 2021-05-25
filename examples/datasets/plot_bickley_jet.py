r"""
Bickley Jet
===========

The :meth:`deeptime.data.bickley_jet` dataset.
"""

# sphinx_gallery_thumbnail_number = 1

import matplotlib.pyplot as plt
import numpy as np

from deeptime.data import bickley_jet

n_particles = 250
dataset = bickley_jet(n_particles, n_jobs=8)
c = np.copy(dataset[0, :, 0])
c /= c.max()

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ani = dataset.make_animation(c=c, agg_backend=False, interval=75, fig=fig, ax=ax, max_frame=100)
