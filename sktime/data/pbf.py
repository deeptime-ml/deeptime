import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sktime.data._data_bindings as bd

class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, numpoints=50):
        self.numpoints = numpoints
        self.stream = self.data_stream()

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5,
                                           init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, s, c = next(self.stream).T
        self.scat = self.ax.scatter(x, y, c=c, s=s, vmin=0, vmax=1,
                                    cmap="jet", edgecolor="k")
        l = 25
        self.ax.axis([-l, l, -l, l])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        n_steps = 5000
        interaction_distance = 1.5
        init_pos_x = np.arange(-24, 24, interaction_distance*.9).astype(np.float32)
        init_pos_y = np.arange(12, 24, interaction_distance*.9).astype(np.float32)
        init_pos = np.dstack(np.meshgrid(init_pos_x, init_pos_y)).reshape(-1, 2)
        n_particles = len(init_pos)
        # init_pos[n_particles//2 : ] += .01
        pbf = bd.PBF(init_pos, np.array([50, 50], dtype=np.float32), interaction_distance, 1)
        pbf.n_solver_iterations = 5
        # pbf.gravity = 2.
        pbf.epsilon = 100
        # pbf.timestep = 1e-8
        pbf.equilibrium_density = 1.

        print("n solve", pbf.n_solver_iterations)
        print("gravity", pbf.gravity)
        print("dt", pbf.timestep)
        print("eps", pbf.epsilon)
        print("rho0", pbf.equilibrium_density)
        print("tis", pbf.tensile_instability_scale)
        print("tik", pbf.tensile_instability_k)

        traj = pbf.run(n_steps)
        traj = traj.reshape((n_steps+1, n_particles, 2))
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""
        # xy = (np.random.random((self.numpoints, 2))-0.5)*10
        s = np.empty((n_particles,))
        s.fill(0.5)
        c = np.empty((n_particles,))
        c.fill(0.5)
        # s, c = np.random.random((self.numpoints, 2)).T
        for t in range(len(traj)):
            yield np.c_[traj[t, :, 0], traj[t, :, 1], s, c]
        # while True:
        #     # xy += 0.03 * (np.random.random((self.numpoints, 2)) - 0.5)
        #     # s += 0.05 * (np.random.random(self.numpoints) - 0.5)
        #     # c += 0.02 * (np.random.random(self.numpoints) - 0.5)
        #     yield np.c_[xy[:,0], xy[:,1], s, c]

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])
        # Set sizes...
        self.scat.set_sizes(300 * abs(data[:, 2])**1.5 + 100)
        # Set colors..
        self.scat.set_array(data[:, 3])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,


if __name__ == '__main__':
    a = AnimatedScatter()
    plt.show()
