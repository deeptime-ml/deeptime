r"""
Arnold-Beltrami-Childress flow
==============================

Example for the :meth:`deeptime.data.abc_flow` dataset.
"""

import numpy as np
from matplotlib import animation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from deeptime.data import abc_flow


def update(num):
    data = scatters[num]
    graph.set_data(data[:, 0], data[:, 1])
    graph.set_3d_properties(data[:, 2])
    return graph,


system = abc_flow(n_steps=25)
scatters = [np.random.uniform(np.pi-.5, np.pi+.5, size=(500, 3))]
for _ in range(50):
    scatters.append(system(scatters[-1], n_jobs=8))

scatters = np.array(scatters)

f = plt.figure(figsize=(18, 18))

ax = f.add_subplot(1, 1, 1, projection='3d')
ax.set_title('Evolution of test points in the ABC flowfield')
graph, = ax.plot(*scatters[0].T, linestyle="", marker="o")
ax.set_xlim([0, 2*np.pi])
ax.set_ylim([0, 2*np.pi])
ax.set_zlim([0, 2*np.pi])

ani = animation.FuncAnimation(f, update, 50, interval=50, blit=True)
