r"""
Swissroll model
===============

Sample a hidden state and an swissroll-transformed emission trajectory.
Demonstrates :meth:`deeptime.data.swissroll_model`.
"""

import matplotlib.pyplot as plt
from matplotlib import animation

from deeptime.data import swissroll_model

n_samples = 500
dtraj, traj = swissroll_model(n_samples)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def init():
    ax.scatter(*traj.T, marker='o', s=20, c=dtraj, alpha=0.6)
    return fig,


def animate(i):
    ax.view_init(elev=10., azim=3*i)
    return fig,


# Animate
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=120, interval=40, blit=False)
