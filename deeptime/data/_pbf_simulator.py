import numpy as np

from threadpoolctl import threadpool_limits

from . import _data_bindings as bd
from ..util.decorators import plotting_function
from ..util.parallel import handle_n_jobs, joining


class PBFSimulator:
    r""" A position based fluids simulator for two-dimensional systems. :footcite:`macklin2013position`

    Its underlying principle is by definition of a rest density :math:`\rho_0`, which the particles in the system
    try to reach by a smoothed particle hydrodynamics style simulation :footcite:`gingold1977smoothed`
    :footcite:`lucy1977numerical`. Up to numerics the simulation is deterministic given a set of parameters.

    Parameters
    ----------
    domain_size : (2,) ndarray
        A 1-dimensional ndarray with two elements describing the extension of the simulation box.
    initial_positions : (N, 2) ndarray
        A 2-dimensional ndarray describing the :math:`N` particles' initial positions. This also fixes the number
        of particles. It is preserved during runs of the simulation.
    interaction_distance : float
        Interaction distance for particles, influences the cell size for the neighbor list.
    n_jobs : int or None, default=None
        Number threads to use for simulation.
    n_solver_iterations : int, default=5
        The number of solver iterations for particle position updates. The more, the slower the simulation but also
        the more accurate.
    gravity : float, default=10
        Gravity parameter which acts on particles' velocities in each time step as constant force.
    epsilon : float, default=10
        Damping parameter. The higher, the more damping.
    timestep : float, default=0.016
        The timestep used for propagation.
    rest_density : float, defaul=1.
        The rest density :math:`\rho_0`.
    tensile_instability_distance : float, default=0.2
        Parameter responsible for surface-tension-like effects.
    tensile_instability_k : float, default=0.1
        Also controls surface-tension effects.

    Notes
    -----

    Each particle has positions :math:`p_1,\ldots,p_n \in\mathbb{R}^d` and
    velocity :math:`v_1,\ldots,v_n\in\mathbb{R^d}`. For the local density a standard SPH estimator is used

    .. math ::
        \rho_i = \sum_i m_i W(p_i - p_j, h),

    where :math:`m_i` refers to the particle's mass (here :math:`m_i \equiv 1` for simplicity), :math:`W` is a
    convolution kernel with interaction distance :math:`h`, i.e., particles with distance larger than :math:`h`
    get assigned a weight of zero in the convolution operation.

    The whole system has an underlying family of density constraints

    .. math ::
        C_i(p_1,\ldots,p_n) = \frac{\rho_i}{\rho_0} - 1.

    The simulation algorithm tries to fulfill this constraint by updating each particle's position, i.e., it tries to
    find :math:`\Delta p_k` so that

    .. math ::
        C_k(p_1,\ldots,p_k + \Delta p_k,\ldots,p_n) = 0.

    This is done by performing Newton's optimization method along the constraint's gradient. Because of potentially
    vanishing gradients, the optimization is regularized / dampened with a parameter :math:`\varepsilon`. In this
    implementation, larger values of :math:`\varepsilon` lead to more regularization.

    When a particle has only very few neighbors, it might not be possible to reach the rest density. This can lead
    to particle clumping (i.e., areas of negative pressures). One remedy is an artificial internal pressure term,
    here referred to as tensile instability. It modifies the weighting inside the convolution kernel itself.
    The default parameters `tensile_instability_distance` and `tensile_instability_k` are the suggested ones in the
    PBF publication.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, domain_size: np.ndarray, initial_positions: np.ndarray,
                 interaction_distance: float, n_jobs=None, n_solver_iterations: int = 5,
                 gravity: float = 10., epsilon: float = 10., timestep: float = 0.016, rest_density: float = 1.,
                 tensile_instability_distance: float = .2, tensile_instability_k: float = 0.1):
        n_jobs = handle_n_jobs(n_jobs)
        if np.atleast_1d(domain_size).ndim != 1 or np.atleast_1d(domain_size).shape[0] != 2 or np.any(domain_size <= 0):
            raise ValueError("Invalid domain size: must be positive and 1-dimensional of length two.")
        if initial_positions.ndim != 2 or initial_positions.shape[1] != 2:
            raise ValueError("initial positions must be a 2-dimensional numpy array of shape (N, 2), where N is"
                             "the number of particles.")
        if interaction_distance <= 0:
            raise ValueError("Interaction distance must be positive.")
        domain_size = domain_size.astype(np.float32, subok=False, copy=False)
        initial_positions = initial_positions.astype(np.float32, subok=False, copy=False)

        self._engine = bd.PBF(initial_positions, domain_size, interaction_distance, n_jobs)
        self._engine.n_solver_iterations = n_solver_iterations
        self._engine.gravity = gravity
        self._engine.epsilon = epsilon
        self._engine.timestep = timestep
        self._engine.rest_density = rest_density
        self._engine.tensile_instability_distance = tensile_instability_distance
        self._engine.tensile_instability_k = tensile_instability_k

    def run(self, n_steps: int, drift: float = 0.):
        r""" Performs `n_steps` many simulation steps and returns a trajectory of recorded particle positions.

        Parameters
        ----------
        n_steps : int
            Number of steps to run.
        drift : float, default=0
            Drift that is added to particles' velocity in `x` direction.

        Returns
        -------
        trajectory : (n_steps, 2) ndarray
            Output trajectory.
        """
        return self._engine.run(n_steps, drift)

    def simulate_oscillatory_force(self, n_oscillations, n_steps, drift=0.3):
        r""" Runs `2*n_oscillations*n_steps` many simulation steps in total by applying the drift alternatingly
        in positive and  negative direction for `n_steps` each, `n_oscillation` times.

        Parameters
        ----------
        n_oscillations : int
            Number of oscillations.
        n_steps : int
            Number of steps per run with a certain drift setting. Is then run again with the negated drift.
        drift : float
            The magnitude of the drift force into x direction.

        Returns
        -------
        trajectory : (T, n) ndarray
            Output trajectory.
        """
        n_runs = 2 * n_oscillations
        traj_total = np.empty((n_runs * n_steps, self.n_particles * 2))
        for i in range(n_runs):
            traj = self.run(n_steps, drift if i % 2 == 0 else -drift)
            traj_total[i * n_steps:(i + 1) * n_steps] = traj
        return traj_total

    def transform_to_density(self, trajectory, n_grid_x=20, n_grid_y=10, n_jobs=None):
        r"""Transforms a two-dimensional PBF particle trajectory to a trajectory of densities by performing KDEs.
        Since we have the prior knowledge that no particles get lost on the way, the densities are
        normalized frame-wise.

        Parameters
        ----------
        trajectory : (T, n, 2) ndarray
            The input trajectory for n particles.
        n_grid_x : int, default=20
            Number of evaluation points of simulation box in x direction.
        n_grid_y : int, default=10
            Number of evaluation points of simulation box in y direction.
        n_jobs : int or None, default=None
            Number of jobs to use when transforming to densities.

        Returns
        -------
        trajectory : (T, n_grid_x * n_grid_y) ndarray
            Output trajectory
        """
        n_jobs = handle_n_jobs(n_jobs)
        return _transform_to_density_impl(self.domain_size, trajectory, n_grid_x, n_grid_y, n_jobs)

    @plotting_function
    def make_animation(self, trajectories, stride=1, mode="scatter", **kw):  # pragma: no cover
        r""" Creates a matplotlib animation object consisting of either scatter plots of contour plots for plotting
        particles and densities, respectively.

        Parameters
        ----------
        trajectories : list of ndarray
            Input trajectories. Must all have same shape.
        stride : int, default=1
            Apply stride to frames so that fewer data is presented.
        mode : str, default="scatter"
            Aside from "scatter" this also supports "contourf" for densities.
        **kw
            Optional keyword arguments. The following are supported:

            * "n_grid_x", required when in mode "contourf" (see :meth:`transform_to_density`)
            * "n_grid_y", required when in mode "contourf" (see :meth:`transform_to_density`)
            * "figsize" controls figure size, defaults to (n_traj * 8, 6)
            * "ncols" controls number of columns, defaults to n_traj
            * "nrows" controls number of rows, defaults to 1
            * "titles" optional list of strings for titles

        Returns
        -------
        animation : FuncAnimation
            Matplotlib animation object.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        if not isinstance(trajectories, (list, tuple)):
            trajectories = [trajectories]
        trajectories = [traj.reshape((len(traj), -1, 2))[::stride] for traj in trajectories]

        n_particles = self.n_particles

        domain_size = self.domain_size

        agg_backend = kw.get('agg_backend', True)

        if agg_backend:
            backend_ = mpl.get_backend()
            mpl.use("Agg")  # Prevent showing stuff

        figsize = kw.get("figsize", (len(trajectories)*8, 6))
        ncols = kw.get("ncols", len(trajectories))
        nrows = kw.get("nrows", 1)
        titles = kw.get("titles", [None] * len(trajectories))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if not isinstance(axes, np.ndarray):
            axes = np.asanyarray(axes)
        for title, ax in zip(titles, axes.flat):
            ax.set_xlim((-domain_size[0] / 2, domain_size[0] / 2))
            ax.set_ylim((-domain_size[1] / 2, domain_size[1] / 2))
            if title is not None:
                ax.set_title(title)

        s = np.empty((n_particles,))
        s.fill(300)
        c = np.empty((n_particles,))
        c.fill(0.5)
        plot_handles = []

        # needed for contourf
        grid = None
        gridx = None
        gridy = None

        if mode == "scatter":
            for traj, ax in zip(trajectories, axes.flat):
                handle = ax.scatter(traj[0, :, 0], traj[0, :, 1], s=s, c=c, vmin=0, vmax=1,
                                    cmap="jet", edgecolor="k")
                plot_handles.append(handle)
        elif mode == "contourf":
            n_grid_x = kw['n_grid_x']
            n_grid_y = kw['n_grid_y']

            gridx = np.linspace(-domain_size[0] / 2, domain_size[0] / 2, num=n_grid_x).astype(np.float32)
            gridy = np.linspace(-domain_size[1] / 2, domain_size[1] / 2, num=n_grid_y).astype(np.float32)
            grid = np.meshgrid(gridx, gridy)

            for k, ax in enumerate(axes.flat):
                handle = ax.contourf(grid[0], grid[1], trajectories[k][0].reshape((len(gridy), len(gridx))))
                plot_handles.append(handle)
        else:
            raise ValueError("Unknown mode {}".format(mode))

        def update_scatter(i):
            for traj, handle in zip(trajectories, plot_handles):
                X = traj[i]
                handle.set_offsets(X)
            return plot_handles

        def update_contourf(i):
            out = []
            for k, ax in enumerate(axes.flat):
                X = trajectories[k][i]
                for tp in plot_handles[k].collections:
                    tp.remove()
                plot_handles[k] = ax.contourf(grid[0], grid[1], X.reshape((len(gridy), len(gridx))))
                out += plot_handles[k].collections
            return out

        update = update_scatter if mode == "scatter" else update_contourf
        ani = animation.FuncAnimation(fig, update, interval=50, blit=True, repeat=False,
                                      frames=len(trajectories[0]))

        if agg_backend:
            mpl.use(backend_)  # Reset backend
        return ani

    @property
    def domain_size(self):
        r""" The size of the domain. """
        return self._engine.domain_size

    @property
    def n_particles(self):
        r""" Number of particles in the simulation. """
        return self._engine.n_particles


def _transform_to_density_impl_worker(args):
    with threadpool_limits(limits=1, user_api='blas'):
        t, frame, kde_input = args[0], args[1], args[2]
        from scipy.stats import gaussian_kde
        out = gaussian_kde(frame.T, bw_method=0.2).evaluate(kde_input.T)
        out /= out.sum()
        return t, out


def _transform_to_density_impl(domain_size, trajectory, n_grid_x=20, n_grid_y=10, n_jobs: int = 1):
    trajectory = trajectory.reshape((len(trajectory), -1, 2))

    gridx = np.linspace(-domain_size[0] / 2, domain_size[0] / 2, num=n_grid_x).astype(np.float32)
    gridy = np.linspace(-domain_size[1] / 2, domain_size[1] / 2, num=n_grid_y).astype(np.float32)
    grid = np.meshgrid(gridx, gridy)
    kde_input = np.dstack(grid).reshape(-1, 2)
    traj_kde = np.empty((len(trajectory), len(kde_input)))

    import multiprocessing as mp
    with joining(mp.Pool(processes=n_jobs)) as pool:
        args = [(t, trajectory[t], kde_input) for t in range(len(trajectory))]
        for result in pool.imap_unordered(_transform_to_density_impl_worker, args):
            traj_kde[result[0]] = result[1]

    return traj_kde
