import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from tqdm import tqdm

from deeptime.data import time_dependent_quintuple_well

random_state = np.random.RandomState(33)

cmap = plt.cm.viridis

system = time_dependent_quintuple_well(h=1e-3, n_steps=10, beta=100)
x = np.arange(-2.5, 2.5, 0.1)
y = np.arange(-2.5, 2.5, 0.1)
xy = np.meshgrid(x, y)

gradxs = []
gradys = []
gradn = []
xxyy = np.dstack(xy)
for pt in xxyy.reshape(-1, 2):
    gradx, grady = system.rhs(0., [pt[0], pt[1]])
    gradxs.append(gradx)
    gradys.append(grady)
    gradn.append(np.linalg.norm([gradx, grady]))

cb = plt.contourf(*xy, np.array(gradn).reshape(xy[0].shape), levels=np.linspace(-5, 5, 100))
plt.colorbar(cb)
plt.show()

trajs = []
for _ in range(10):
    x0 = random_state.uniform(-2.5, 2.5, size=(1, 2))
    traj = system.trajectory(0., [[0.1, 0.1]], n_evaluations=int(10 / (1e-3 * 10)))
    trajs.append(traj)
trajs = np.stack(trajs)

l = []
for t in tqdm(np.arange(0., 10., 1e-2)):
    V = system.potential(t, np.dstack(xy).reshape(-1, 2)).reshape(xy[0].shape)
    l.append(V)
l = np.stack(l)

vmin = np.min(l)
vmax = np.max(l)

fig, ax = plt.subplots()
handle = ax.contourf(*xy, l[0], vmin=vmin, vmax=vmax, cmap=cmap)
scatter_handle = ax.scatter(*trajs[:, 0, :].T, color='red', zorder=100)
handles = [scatter_handle, handle]


def update(i):
    out = [scatter_handle]
    handles[0].set_offsets(trajs[:, i, :])
    for tp in handles[1].collections:
        tp.remove()
    handles[1] = ax.contourf(*xy, l[i], vmin=vmin, vmax=vmax, cmap=cmap)
    out += handles[1].collections
    return out


interval = .01
ani = animation.FuncAnimation(fig, update, interval=50, blit=True, repeat=False, frames=len(l))

# ani.save('bleh.mp4')
plt.show()
