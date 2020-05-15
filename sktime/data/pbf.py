import numpy as np
import sktime.data._data_bindings as bd

from ..util import handle_n_jobs

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation


    class AnimatedScatter(object):
        """An animated scatter plot using matplotlib.animations.FuncAnimation."""

        def __init__(self, numpoints=50):
            self.numpoints = numpoints
            self.stream = self.data_stream()

            # Setup the figure and axes...
            self.fig, self.ax = plt.subplots()
            # Then setup FuncAnimation.
            self.ani = animation.FuncAnimation(self.fig, self.update, interval=50,
                                               init_func=self.setup_plot, blit=True, repeat=False, save_count=5000)

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
            n_steps = 500
            interaction_distance = 1.5
            init_pos_x = np.arange(-24, 24, interaction_distance * .9).astype(np.float32)
            init_pos_y = np.arange(12, 24, interaction_distance * .9).astype(np.float32)
            init_pos = np.dstack(np.meshgrid(init_pos_x, init_pos_y)).reshape(-1, 2)

            init_pos_x = np.arange(-24, 24, interaction_distance * .9).astype(np.float32)
            init_pos_y = np.arange(-24, -18, interaction_distance * .9).astype(np.float32)
            init_pos_lower = np.dstack(np.meshgrid(init_pos_x, init_pos_y)).reshape(-1, 2)

            init_pos = np.concatenate((init_pos, init_pos_lower), axis=0)

            n_particles = len(init_pos)
            # init_pos[n_particles//2 : ] += .01
            pbf = bd.PBF(init_pos, np.array([50, 50], dtype=np.float32), interaction_distance, 8)
            pbf.n_solver_iterations = 5
            # pbf.gravity = 2.
            pbf.epsilon = 10
            # pbf.timestep = 1e-8
            pbf.equilibrium_density = 1.

            print("n solve", pbf.n_solver_iterations)
            print("gravity", pbf.gravity)
            print("dt", pbf.timestep)
            print("eps", pbf.epsilon)
            print("rho0", pbf.equilibrium_density)
            print("tis", pbf.tensile_instability_scale)
            print("tik", pbf.tensile_instability_k)

            import tqdm
            traj_total = None
            for i in tqdm.tqdm(range(10)):
                traj = pbf.run(n_steps, 1.5 if i % 2 != 0 else 0)
                if traj_total is None:
                    traj_total = traj
                else:
                    traj_total = np.concatenate((traj_total, traj), axis=0)
            traj = traj_total.reshape((-1, n_particles, 2))
            traj = traj[::10]
            """Generate a random walk (brownian motion). Data is scaled to produce
            a soft "flickering" effect."""
            # xy = (np.random.random((self.numpoints, 2))-0.5)*10
            s = np.empty((n_particles,))
            s.fill(0.5)
            c = np.empty((n_particles,))
            c.fill(0.5)
            # s, c = np.random.random((self.numpoints, 2)).T

            for t in tqdm.tqdm(range(len(traj))):
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
            self.scat.set_sizes(300 * abs(data[:, 2]) ** 1.5 + 100)
            # Set colors..
            self.scat.set_array(data[:, 3])

            # We need to return the updated artist for FuncAnimation to draw..
            # Note that it expects a sequence of artists, thus the trailing comma.
            return self.scat,
except ImportError:
    pass


class PBF(object):

    def __init__(self, domain_size: np.ndarray, initial_positions: np.ndarray,
                 interaction_distance: float, n_jobs=None, n_solver_iterations: int = 5,
                 gravity: float = 10., epsilon: float = 10., timestep: float = 0.016, rest_density: float = 1.,
                 tensile_instability_distance: float = .2, tensile_instability_k: float = 0.1):
        if np.atleast_1d(domain_size).ndim != 1 or domain_size.shape[0] != 2 or np.any(domain_size <= 0):
            raise ValueError("Invalid domain size: must be positive and 1-dimensional of length two.")
        if initial_positions.ndim != 2 or initial_positions.shape[1] != 2:
            raise ValueError("initial positions must be a 2-dimensional numpy array of shape (N, 2), where N is"
                             "the number of particles.")
        if interaction_distance <= 0:
            raise ValueError("Interaction distance must be positive.")
        domain_size = domain_size.astype(np.float32, subok=False, copy=False)
        initial_positions = initial_positions.astype(np.float32, subok=False, copy=False)

        self._engine = bd.PBF(initial_positions, domain_size, interaction_distance, handle_n_jobs(n_jobs))
        self._engine.n_solver_iterations = n_solver_iterations
        self._engine.gravity = gravity
        self._engine.epsilon = epsilon
        self._engine.timestep = timestep
        self._engine.rest_density = rest_density
        self._engine.tensile_instability_distance = tensile_instability_distance
        self._engine.tensile_instability_k = tensile_instability_k

    def run(self, n_steps: int, drift: float):
        return self._engine.run(n_steps, drift)


if __name__ == '__main__':
    a = AnimatedScatter()
    a.ani.save("out.mp4", codec="h264")
    print("done")
    # plt.show()
