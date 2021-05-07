import numpy as np

from ._systems import TimeDependentSystem
from ..util.data import TimeLaggedDataset
from ..util.decorators import plotting_function
from ._data_bindings import BickleyJet as BickleyJetImpl


class BickleyJet(TimeDependentSystem):
    r"""Implementation of the Bickley jet. Based on :footcite:`hadjighasem2016spectral`.

    The parameters are set to

    .. math::

        \begin{aligned}
            U_0 &= 5.4138 \times \frac{10^6\mathrm{m}}{\mathrm{day}},\\
            L_0 &= 1.77 \times 10^6\,\mathrm{m},\\
            r_0 &= 6.371 \times 10^6\,\mathrm{m},\\
            c &= (0.1446, 0.205, 0.461)^\top U_0,\\
            \mathrm{eps} &= (0.075, 0.15, 0.3)^\top,\\
            k &= (2,4,6)^\top \frac{1}{r_0}.
        \end{aligned}

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, h: float, n_steps: int):
        # set parameters
        super().__init__(BickleyJetImpl(), h, n_steps)

    @property
    def U0(self):
        r""" The characteristic velocity scale :math:`U_0 = 5.4138` in :math:`10^6 \frac{\mathrm{m}}{\mathrm{day}}`. """
        return self._impl.U0
    
    @property
    def L0(self):
        r""" The characteristic length scale :math:`L_0 = 1.77` in :math:`10^6\;\mathrm{m}`. """
        return self._impl.L0

    @property
    def r0(self):
        r""" The mean radius of the earth :math:`r_0 = 6.371` in :math:`10^6\;\mathrm{m}`. """
        return self._impl.r0

    @property
    def c(self):
        r""" Traveling speeds :math:`c = (0.1446, 0.205, 0.461)^\top U_0`. """
        return self._impl.c

    @property
    def eps(self):
        r""" Wave amplitudes :math:`\varepsilon = (0.075, 0.15, 0.3)^\top`. """
        return self._impl.eps

    @property
    def k(self):
        r""" Wave numbers :math:`k = (2,4,6)^\top \frac{1}{r_0}`. """
        return self._impl.k

    def trajectory(self, t0, x0, length, seed=-1, n_jobs=None):
        traj = super().trajectory(t0, x0, length, seed, n_jobs)
        traj[..., 0] = np.mod(traj[..., 0], 20)  # periodicity in x direction
        return traj

    def generate(self, n_particles, n_jobs=None, seed=None) -> np.ndarray:
        """Generates a trajectory with a fixed number of particles / test points for 401 evaluation steps, i.e.,
        `401 * self.n_steps * self.h` integration steps.

        Parameters
        ----------
        n_particles : int
            Number of particles.
        n_jobs : int, optional, default=None
            Number of jobs.
        seed : int or None, optional, default=None
            Random seed used for initialization of particle positions at :math:`t=0`.

        Returns
        -------
        Z : np.ndarray (m, 401, 2)
            Trajectories for m uniformly distributed test points in Omega = [0, 20] x [-3, 3].
        """
        state = np.random.RandomState(seed)
        X = np.vstack((state.uniform(0, 20, (n_particles,)), state.uniform(-3, 3, (n_particles, ))))
        return self.trajectory(0, X.T, 401, n_jobs=n_jobs)

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


class BickleyJetDataset:

    def __init__(self, trajectory):
        self.data = trajectory

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @plotting_function
    def make_animation(self, **kw):  # pragma: no cover
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        agg_backend = kw.get('agg_backend', True)
        figsize = kw.get("figsize", (8, 6))
        title = kw.get("title", None)
        s = kw.get("s", None)
        c = kw.get("c", None)
        stride = kw.get("stride", 1)
        cmap = kw.get('cmap', 'jet')
        edgecolor = kw.get('edgecolor', 'k')
        repeat = kw.get('repeat', False)
        interval = kw.get('interval', 50)
        fig = kw.get('fig', None)
        ax = kw.get('ax', None)
        max_frame = kw.get('max_frame', None)

        data = self.data[:max_frame:stride]
        n_particles = data.shape[1]

        if agg_backend:
            backend_ = mpl.get_backend()
            mpl.use("Agg")  # Prevent showing stuff

        if s is None:
            s = np.empty((n_particles,))
            s.fill(300)
        if c is None:
            c = np.empty((n_particles,))
            c.fill(0.5)

        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_title(title)
        ax.set_xlim([0, 20])
        ax.set_ylim([-3, 3])

        plot_handles = []
        s_handle = ax.scatter(*data[0].T, s=s, c=c, vmin=0, vmax=1,
                              cmap=cmap, edgecolor=edgecolor)
        plot_handles.append(s_handle)

        def update(i):
            handle = plot_handles[0]
            handle.set_offsets(data[::stride][i])
            return plot_handles

        ani = animation.FuncAnimation(fig, update, interval=interval, blit=True, repeat=repeat,
                                      frames=data[::stride].shape[0])

        if agg_backend:
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
        from deeptime.clustering import BoxDiscretization

        disc = BoxDiscretization(3, n_bins).fit(np.concatenate((self.data, self.data_lagged))).fetch_model()
        traj1 = disc.transform_onehot(self.data)
        traj2 = disc.transform_onehot(self.data_lagged)

        return BickleyJetEndpointsDataset3DClustered(traj1, traj2)


class BickleyJetEndpointsDataset3DClustered(TimeLaggedDataset):

    def __init__(self, data, data_lagged):
        super().__init__(data, data_lagged)
