"""
VAMP on Position Based Fluids
=============================

Projection of position based fluids simulation timeseries on dominant component. For more details,
see the `VAMP tutorial <../notebooks/vamp.ipynb#Example-with-position-based-fluids>`__.
"""

import matplotlib.pyplot as plt
import numpy as np

from deeptime.data import position_based_fluids
from deeptime.decomposition import VAMP

pbf_simulator = position_based_fluids(n_burn_in=500, n_jobs=8)
trajectory = pbf_simulator.simulate_oscillatory_force(n_oscillations=3, n_steps=400)
n_grid_x = 20
n_grid_y = 10
kde_trajectory = pbf_simulator.transform_to_density(
    trajectory, n_grid_x=n_grid_x, n_grid_y=n_grid_y, n_jobs=8
)
tau = 100
model = VAMP(lagtime=100).fit(kde_trajectory).fetch_model()
projection_left = model.forward(kde_trajectory, propagate=False)
projection_right = model.backward(kde_trajectory, propagate=False)

f, ax = plt.subplots(1, 1, figsize=(5, 5))
start = 400
stop = len(kde_trajectory) - tau  # 5000
left = projection_left[:-tau][start:stop, 0]
right = projection_right[tau:][start:stop, 0]
lw = 4
ax.plot(np.arange(start, stop), left, label="left", linewidth=lw)
ax.plot(np.arange(start, stop)[::50], right[::50], '--', label="right", linewidth=3, markersize=12)
ax.vlines([start + i * 400 for i in range(1, (stop - start) // 400)], np.min(left), np.max(left),
          linestyles='dotted')
ax.legend()
