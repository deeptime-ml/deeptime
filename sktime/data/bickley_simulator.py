import numpy as np
from scipy.integrate import solve_ivp

class BickleyJet(object):
    """
    Implementation of the Bickley jet based on "A Spectral Clustering Approach to Lagrangian Vortex Detection"
    by A. Hadjighasem, D. Karrasch, H. Teramoto, and G. Haller.
    """

    def __init__(self):
        # set parameters
        self.U0 = 5.4138  # units changed to 10^6 m per day
        self.L0 = 1.77  # in 10^6 m
        self.r0 = 6.371  # in 10^6 m
        self.c = np.array([0.1446, 0.205, 0.461]) * self.U0
        self.eps = np.array([0.075, 0.15, 0.3])
        self.k = np.array([2, 4, 6]) / self.r0

    def generate(self, n_particles, n_jobs=None) -> np.ndarray:
        """

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


def _generate_impl_worker(args):
    i, Xi, L0, U0, c, eps, k = args

    T0 = 0
    T1 = 40
    nT = 401  # number of time points, TODO: add as parameter?
    T = np.linspace(T0, T1, nT)  # time points

    sol = solve_ivp(_rhs, [0, 40], Xi, t_eval=T, args=(L0, U0, c, eps, k))
    # periodic in x-direction
    for i in range(sol.y.shape[1]):
        while sol.y[0, i] > 20:
            sol.y[0, i] -= 20
        while sol.y[0, i] < 0:
            sol.y[0, i] += 20
    # sol.y[0, :] = np.mod(sol.y[0, :], 20)  # periodic in x-direction

    return i, sol.y


def _generate_impl(n_particles, L0, U0, c, eps, k, n_jobs=None) -> np.ndarray:
    from sktime.util import handle_n_jobs

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


if __name__ == "__main__":
    m = 5000
    sys = BickleyJet()

    traj = sys.generate(m, 4)
    print(traj.shape)

    import matplotlib.pyplot as plt
    for i in range(0, 401, 5):
        plt.figure(1);
        plt.clf()
        plt.scatter(traj[0, i, :], traj[1, i, :])
        plt.xlim(0, 20);
        plt.ylim(-3, 3)
        plt.pause(0.01)


    # from scipy.cluster.vq import kmeans2
    # import matplotlib.pyplot as plt
    #
    # import d3s.kernels as kernels
    # import d3s.algorithms as algorithms
    #
    # m = 5000
    # sys = BickleyJet()
    # Z = sys.generate(m)
    # nT = Z.shape[1]
    #
    # # %% plot particles
    # for i in range(0, nT, 5):
    #     plt.figure(1);
    #     plt.clf()
    #     plt.scatter(Z[0, i, :], Z[1, i, :])
    #     plt.xlim(0, 20);
    #     plt.ylim(-3, 3)
    #     plt.pause(0.01)
    #
    # # %% apply kernel CCA to detect coherent sets
    # X = Z[:, 0, :]  # particles at time T0
    # Y = Z[:, -1, :]  # particles at time T1
    #
    # sigma = 1
    # k = kernels.gaussianKernel(sigma)
    #
    # evs = 9  # number of eigenfunctions to be computed
    # d, V = algorithms.kcca(X, Y, k, evs, epsilon=1e-3)
    #
    # # %% plot eigenfunctions
    # for i in range(evs):
    #     plt.figure()
    #     plt.scatter(X[0, :], X[1, :], c=V[:, i])
    # plt.show()
    #
    # # %% k-means of eigenfunctions
    # c, l = kmeans2(np.real(V), 7)
    # plt.figure()
    # plt.scatter(X[0, :], X[1, :], c=l)
    # plt.show()
