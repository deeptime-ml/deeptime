r"""
Position-based fluids
=====================

Example of the :meth:`deeptime.data.position_based_fluids` simulation engine.
"""

import numpy as np
import matplotlib.pyplot as plt

from deeptime.data import position_based_fluids

init_pos_x = np.arange(-12, 12, 1.4).astype(np.float32)
init_pos_y = np.arange(-12, 0, 1.4).astype(np.float32)
initial_positions = np.dstack(np.meshgrid(init_pos_x, init_pos_y)).reshape(-1, 2)

pbf = position_based_fluids(n_burn_in=0, n_jobs=8, initial_positions=initial_positions)
ftraj = pbf.simulate_oscillatory_force(2, n_steps=400)
ani = pbf.make_animation(ftraj, agg_backend=False, figsize=(6, 4), stride=5)
plt.show()
