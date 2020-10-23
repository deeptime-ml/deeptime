import numpy as np
from scipy.integrate import solve_ivp

from . import TimeSeriesDataset, TimeLaggedDataset
from ..util.decorators import plotting_function
from ..util.parallel import handle_n_jobs


class BickleyJet(object):
    r"""Implementation of the Bickley jet.
    Based on :cite:`bickley-simulator-hadjighasem2016spectral`.

    References
    ----------
    .. bibliography:: /references.bib
        :style: unsrt
        :filter: docname in docnames
        :keyprefix: bickley-simulator-
    """

    def __init__(self):
        r""" Creates a new instance of the simulator and sets a few parameters. In particular,

        .. math::

            \begin{aligned}
                U_0 &= 5.4138 \times \frac{10^6\mathrm{m}}{\mathrm{day}},\\
                L_0 &= 1.77 \times 10^6\,\mathrm{m},\\
                r_0 &= 6.371 \times 10^6\,\mathrm{m},\\
                c &= (0.1446, 0.205, 0.461)^\top U_0,\\
                \mathrm{eps} &= (0.075, 0.15, 0.3)^\top,\\
                k &= (2,4,6)^\top \frac{1}{r_0}.
            \end{aligned}

        """
        # set parameters
        self.U0 = 5.4138  # units changed to 10^6 m per day
        self.L0 = 1.77  # in 10^6 m
        self.r0 = 6.371  # in 10^6 m
        self.c = np.array([0.1446, 0.205, 0.461]) * self.U0
        self.eps = np.array([0.075, 0.15, 0.3])
        self.k = np.array([2, 4, 6]) / self.r0

    def generate(self, n_particles, n_jobs=None) -> np.ndarray:
        """Generates a trajectory with a fixed number of particles / test points.

        Parameters
        ----------
        n_particles : int
            Number of particles.
        n_jobs : int, optional, default=None
            Number of jobs.

        Returns
        -------
        Z : np.ndarray (2, 401, m)
            Trajectories for m uniformly distributed test points in Omega = [0, 20] x [-3, 3].
        """
        return _generate_impl(n_particles, self.L0, self.U0, self.c, self.eps, self.k, n_jobs=n_jobs)

    @staticmethod
    def to_3d(data: np.ndarray, radius: float = 1.) -> np.ndarray:
        r"""Maps a generated trajectory into 3d space by transforming it through

        .. math::

            \begin{pmatrix} x \\ y \end{pmatrix} \mapsto \begin{pmatrix}
                r\cdot \cos\left( 2\pi \frac{x}{20} \right) \\
                r\cdot \sin\left( 2\pi \frac{x}{20} \right) \\
                \frac{y}{3}
            \end{pmatrix}

        which means that the particles are now on the surface of a cylinder with a fixed radius, i.e.,
        periodic boundary conditions are directly encoded in the space.

        Parameters
        ----------
        data : (T, 2) ndarray
            The generated trajectory.
        radius : float, default=1
            The radius of the cylinder.
        Returns
        -------
        xyz : (T, 3) np.ndarray
            Trajectory on the cylinder.
        """
        t = data[..., 0] * 2. * np.pi / 20.
        z = data[..., 1]
        x_3d = radius * np.cos(t)
        y_3d = radius * np.sin(t)
        xyz = np.stack((x_3d, y_3d, z / 3))
        return xyz.T


class BickleyJetDataset(TimeSeriesDataset):

    def __init__(self, trajectory):
        super(BickleyJetDataset, self).__init__(trajectory)

    @plotting_function
    def make_animation(self, **kw):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        n_particles = self.data.shape[1]

        backend_ = mpl.get_backend()
        mpl.use("Agg")  # Prevent showing stuff

        figsize = kw.get("figsize", (8, 6))
        title = kw.get("title", None)
        s = kw.get("s", None)
        c = kw.get("c", None)
        stride = kw.get("stride", 1)
        cmap = kw.get('cmap', 'jet')
        edgecolor = kw.get('edgecolor', 'k')

        if s is None:
            s = np.empty((n_particles,))
            s.fill(300)
        if c is None:
            c = np.empty((n_particles,))
            c.fill(0.5)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_title(title)

        plot_handles = []
        s_handle = ax.scatter(self.data[0, :, 0], self.data[0, :, 1], s=s, c=c, vmin=0, vmax=1,
                              cmap=cmap, edgecolor=edgecolor)
        plot_handles.append(s_handle)

        def update(i):
            handle = plot_handles[0]
            handle.set_offsets(self.data[::stride][i])
            return plot_handles

        ani = animation.FuncAnimation(fig, update, interval=50, blit=True, repeat=False,
                                      frames=self.data[::stride].shape[0])

        mpl.use(backend_)  # Reset backend

        return ani

    def endpoints_dataset(self):
        return BickleyJetEndpointsDataset(self.data[0], self.data[-1])


class BickleyJetEndpointsDataset(TimeLaggedDataset):

    def __init__(self, data_0, data_t):
        super().__init__(data_0, data_t)

    def to_3d(self, radius: float = 1.):
        r""" See :meth:`BickleyJet.to_3d`. """
        return BickleyJetEndpointsDataset3D(
            BickleyJet.to_3d(self.data, radius), BickleyJet.to_3d(self.data_lagged, radius)
        )


class BickleyJetEndpointsDataset3D(TimeLaggedDataset):

    def __init__(self, data: np.ndarray, data_lagged: np.ndarray):
        super().__init__(data, data_lagged)
        assert data.shape[1] == 3
        assert data_lagged.shape[1] == 3

    def cluster(self, n_bins):
        from deeptime.clustering import ClusterModel

        minval = min(np.min(self.data), np.min(self.data_lagged))
        maxval = max(np.max(self.data), np.max(self.data_lagged))

        grid = np.linspace(minval, maxval, num=n_bins, endpoint=True)
        mesh = np.vstack(np.meshgrid(grid, grid, grid)).reshape(3, -1).T
        cm = ClusterModel(len(mesh), mesh)

        dtraj1 = cm.transform(self.data.astype(np.float64))
        traj1 = np.zeros((len(self.data), mesh.shape[0]))
        traj1[np.arange(len(self.data)), dtraj1] = 1.

        dtraj2 = cm.transform(self.data_lagged.astype(np.float64))
        traj2 = np.zeros((len(self.data_lagged), mesh.shape[0]))
        traj2[np.arange(len(self.data_lagged)), dtraj2] = 1.

        return BickleyJetEndpointsDataset3DClustered(traj1, traj2)


class BickleyJetEndpointsDataset3DClustered(TimeLaggedDataset):

    def __init__(self, data, data_lagged):
        super().__init__(data, data_lagged)


def _generate_impl_worker(args):
    i, Xi, L0, U0, c, eps, k = args

    T0 = 0
    T1 = 40
    nT = 401  # number of time points, TODO: add as parameter?
    T = np.linspace(T0, T1, nT)  # time points

    sol = solve_ivp(_rhs, [0, 40], Xi, t_eval=T, args=(L0, U0, c, eps, k))
    # periodic in x-direction
    sol.y[0, :] = np.mod(sol.y[0, :], 20)

    return i, sol.y


def _generate_impl(n_particles, L0, U0, c, eps, k, n_jobs=None) -> np.ndarray:
    X = np.vstack((20 * np.random.rand(n_particles), 6 * np.random.rand(n_particles) - 3))
    nT = 401
    Z = np.zeros((2, nT, n_particles))

    import multiprocessing as mp
    with mp.Pool(processes=handle_n_jobs(n_jobs)) as pool:
        args = [(i, X[:, i], L0, U0, c, eps, k) for i in range(n_particles)]
        for result in pool.imap_unordered(_generate_impl_worker, args):
            Z[:, :, result[0]] = result[1]
    return Z


def _sech(x):
    """
    Hyperbolic secant.
    """
    return 1 / np.cosh(x)


def _rhs(t, x, L0, U0, c, eps, k):
    f = np.real(eps[0] * np.exp(-1j * k[0] * c[0] * t) * np.exp(1j * k[0] * x[0])
                + eps[1] * np.exp(-1j * k[1] * c[1] * t) * np.exp(1j * k[1] * x[0])
                + eps[2] * np.exp(-1j * k[2] * c[2] * t) * np.exp(1j * k[2] * x[0]))
    df_dx = np.real(eps[0] * np.exp(-1j * k[0] * c[0] * t) * 1j * k[0] * np.exp(1j * k[0] * x[0])
                    + eps[1] * np.exp(-1j * k[1] * c[1] * t) * 1j * k[1] * np.exp(1j * k[1] * x[0])
                    + eps[2] * np.exp(-1j * k[2] * c[2] * t) * 1j * k[2] * np.exp(1j * k[2] * x[0]))

    sech_sq = _sech(x[1] / L0) ** 2

    return np.array([U0 * sech_sq + 2 * U0 * np.tanh(x[1] / L0) * sech_sq * f,
                     U0 * L0 * sech_sq * df_dx])
