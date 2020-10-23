from typing import Tuple, List, Union

import numpy as np


def double_well_discrete():
    r"""MCMC process in a symmetric double well potential, spatially discretized to 100 bins.
    The discrete trajectory contains 100000 steps, discrete time step dt=10. The result object allows access to
    discretizations of varying quality as well as gives opportunity to synthetically generate more data.

    Returns
    -------
    dataset : deeptime.data.double_well_dataset.DoubleWellDiscrete
        an object that contains a markov state model corresponding to the process and the discrete trajectory
    """
    from deeptime.data.double_well_dataset import DoubleWellDiscrete
    return DoubleWellDiscrete()


def ellipsoids(laziness: float = 0.97, seed=None):
    r""" Example data of a two-state markov chain which can be featurized into two parallel ellipsoids and optionally
    rotated into higher-dimensional space.

    See :class:`Ellipsoids <deeptime.data.ellipsoids_dataset.Ellipsoids>` for more details.

    .. plot::

       import matplotlib.pyplot as plt
       import deeptime

       ftraj = deeptime.data.ellipsoids(seed=17).observations(1000)
       plt.scatter(*(ftraj.T))
       plt.grid()
       plt.title(r'Ellipsoids dataset observations with laziness of $0.97$.')
       plt.show()

    Parameters
    ----------
    laziness : float in half-open interval (0.5, 1.], default=0.97
        The probability to stay in either state rather than transitioning.
    seed : int, optional, default=None
        Optional random seed for reproducibility.

    Returns
    -------
    dataset : deeptime.data.ellipsoids_dataset.Ellipsoids
        an object that contains methods to create discrete and continuous observations
    """
    from deeptime.data.ellipsoids_dataset import Ellipsoids
    return Ellipsoids(laziness=laziness, seed=seed)


def position_based_fluids(n_burn_in=5000, n_jobs=None):
    r""" Creates a position based fluids (PBF) simulator. It was introduced in :cite:`data-api-macklin2013position`.
    Up to numerics the simulation is deterministic.

    The simulation box has dimensions :math:`[-40, 40]\times [-25, 25]` and the initial positions of the particles are
    around the top boundary of the box. For simplicity of use, the initial positions are fixed in this method and yield
    972 particles. For custom positioning, please use the simulator directly.

    The interaction distance is set to :math:`d = 1.5` and `n_burn_in` steps are
    performed to equilibrate the system before returning the simulator.

    For more details see :class:`PBFSimulator <deeptime.data.pbf_simulator.PBFSimulator>`.

    .. plot::

        import matplotlib.pyplot as plt
        import deeptime

        ftraj = deeptime.data.position_based_fluids(n_burn_in=150).run(300)
        f, axes = plt.subplots(3, 2, figsize=(15, 10))
        for i, ax in enumerate(axes.flat):
            ax.scatter(*(ftraj[i*50].reshape(-1, 2).T))
            ax.set_title("t = {}".format(i*50))
            ax.grid()
        f.suptitle(r'PBF dataset observations.')
        plt.show()

    Parameters
    ----------
    n_burn_in : int, default=5000
        Number of steps without any drift force to equilibrate the system.
    n_jobs : int or None, default=None
        Number of threads to use for simulation.

    Returns
    -------
    simulator : deeptime.data.pbf_simulator.PBFSimulator
        The PBF simulator.

    References
    ----------
    .. bibliography:: /references.bib
        :style: unsrt
        :filter: docname in docnames
        :keyprefix: data-api-
    """
    from deeptime.data.pbf_simulator import PBFSimulator
    interaction_distance = 1.5
    init_pos_x = np.arange(-24, 24, interaction_distance * .9).astype(np.float32)
    init_pos_y = np.arange(-12, 24, interaction_distance * .9).astype(np.float32)
    init_pos = np.dstack(np.meshgrid(init_pos_x, init_pos_y)).reshape(-1, 2)
    domain = np.array([80, 50])
    pbf = PBFSimulator(domain_size=domain, initial_positions=init_pos, interaction_distance=interaction_distance,
                       n_jobs=n_jobs)
    # equilibrate
    pbf.run(n_burn_in, 0)

    return pbf


def drunkards_walk(grid_size: Tuple[int, int] = (10, 10),
                   bar_location: Union[Tuple[int, int], List[Tuple[int, int]]] = (9, 9),
                   home_location: Union[Tuple[int, int], List[Tuple[int, int]]] = (0, 0)):
    r"""This example dataset simulates the steps a drunkard living in a two-dimensional plane takes finding
    either the bar or the home as two absorbing states.

    The drunkard can take steps in a 3x3 grid with uniform probability (as possible, in the corners the only
    possibilities are the ones that do not lead out of the grid). The transition matrix
    :math:`P\in\mathbb{R}^{nm\times nm}`  possesses one absorbing state for home and bar, respectively,
    and uniform two-dimensional jump probabilities in between. The grid is of size :math:`n\times m` and a point
    :math:`(i,j)` is identified with state :math:`i+nj` in the transition matrix.

    .. plot::

        import numpy as np
        import deeptime

        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        import scipy
        from scipy.interpolate import CubicSpline

        sim = deeptime.data.drunkards_walk(bar_location=(0, 0), home_location=(9, 9))
        walk = sim.walk(start=(7, 2), n_steps=250, seed=17)

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.scatter(*sim.home_location.T, marker='*', label='Home', c='red', s=150, zorder=5)
        ax.scatter(*sim.bar_location.T, marker='*', label='Bar', c='orange', s=150, zorder=5)
        ax.scatter(7, 2, marker='*', label='Start', c='black', s=150, zorder=5)

        x = np.r_[walk[:, 0]]
        y = np.r_[walk[:, 1]]
        f, u = scipy.interpolate.splprep([x, y], s=0, per=False)
        xint, yint = scipy.interpolate.splev(np.linspace(0, 1, 50000), f)
        ax.scatter(x, y, label='Visited intermediates')

        points = np.stack([xint, yint]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        coll = LineCollection(segments, cmap='cool', linestyle='dotted')
        coll.set_array(np.linspace(0, 1, num=len(points), endpoint=True))
        coll.set_linewidth(2)
        ax.add_collection(coll)

        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))
        ax.set_xlabel('coordinate x')
        ax.set_ylabel('coordinate y')
        ax.grid()
        ax.legend()

    Parameters
    ----------
    grid_size : tuple
        The grid size, must be tuple of length two.
    bar_location : tuple
        The bar location, must be valid coordinate and tuple of length two.
    home_location : tuple
        The home location, must be valid coordinate and tuple of length two.

    Returns
    -------
    simulator : deeptime.data.drunkards_walk_simulator.DrunkardsWalk
        Simulator instance.
    """
    from deeptime.data.drunkards_walk_simulator import DrunkardsWalk
    return DrunkardsWalk(grid_size, bar_location=bar_location, home_location=home_location)


def bickley_jet(n_particles: int, n_jobs=None):
    r"""Simulates the Bickley jet for a number of particles.
    The implementation is based on :cite:`bickley-api-hadjighasem2016spectral` with parameters

    .. math::

            \begin{aligned}
                U_0 &= 5.4138 \times \frac{10^6\mathrm{m}}{\mathrm{day}},\\
                L_0 &= 1.77 \times 10^6\,\mathrm{m},\\
                r_0 &= 6.371 \times 10^6\,\mathrm{m},\\
                c &= (0.1446, 0.205, 0.461)^\top U_0,\\
                \mathrm{eps} &= (0.075, 0.15, 0.3)^\top,\\
                k &= (2,4,6)^\top \frac{1}{r_0},
            \end{aligned}

    in a domain :math:`\Omega = [0, 20] \times [-3, 3]`. The resulting dataset describes the temporal evolution
    of :code:`n_particles` over 401 timesteps in :math:`\Omega`. The domain is periodic in x-direction.
    The dataset offers methods to wrap the domain into three-dimensional
    space onto the surface of a cylinder

    .. math::

        \begin{pmatrix} x \\ y \end{pmatrix} \mapsto \begin{pmatrix}
            r\cdot \cos\left( 2\pi \frac{x}{20} \right) \\
            r\cdot \sin\left( 2\pi \frac{x}{20} \right) \\
            \frac{y}{3}
        \end{pmatrix},

    with the option to further discretize the three-dimensional dataspace via binning. This way the
    discontinuity introduced by 2D periodicity is treated.

    .. plot::

        import matplotlib.pyplot as plt
        import deeptime

        n_particles = 1000
        dataset = deeptime.data.bickley_jet(n_particles, n_jobs=8)

        fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(16, 10))

        for t, ax in zip([0, 1, 2, 200, 300, 400], axes.flatten()):
            ax.scatter(*dataset[t].T, c=dataset[0, :, 0], s=50)
            ax.set_title(f"Particles at t={t}")

    Parameters
    ----------
    n_particles : int
        Number of particles which are propagated.
    n_jobs : n_jobs : int or None, default=None
        Number of threads to use for simulation.

    Returns
    -------
    dataset : deeptime.data.bickley_simulator.BickleyJetDataset
        Dataset over all the generated frames.

    Examples
    --------

    >>> import deeptime
    >>> dataset = deeptime.data.bickley_jet(n_particles=5, n_jobs=1)
    >>> # shape is 401 frames for 5 particles in two dimensions
    >>> print(dataset.data.shape)
    (401, 5, 2)

    >>> # returns a timelagged dataset for first and last frame
    >>> endpoints = dataset.endpoints_dataset()
    >>> endpoints.data.shape
    (5, 2)

    >>> # maps the endpoints dataset onto a cylinder of radius 5
    >>> endpoints_3d = endpoints.to_3d(radius=5.)
    >>> endpoints_3d.data.shape
    (5, 3)

    >>> # bins the data uniformly with 10 bins per axis
    >>> endpoints_3d_clustered = endpoints_3d.cluster(n_bins=10)
    >>> # 5 particles and 10*10*10 bins
    >>> endpoints_3d_clustered.data.shape
    (5, 1000)

    References
    ----------
    .. bibliography:: /references.bib
        :style: unsrt
        :filter: docname in docnames
        :keyprefix: bickley-api-
    """
    from deeptime.data.bickley_simulator import BickleyJet, BickleyJetDataset
    simulator = BickleyJet()
    traj = simulator.generate(n_particles=n_particles, n_jobs=n_jobs)
    traj_reshaped = traj.transpose(1, 2, 0)
    return BickleyJetDataset(traj_reshaped)


def birth_death_chain(q, p):
    r""" Generates a birth and death chain simulator from annihilation and creation probabilities `q` and `p`.

    A general birth and death chain on a d-dimensional state space has the transition matrix

    .. math::

        p_{ij} = \begin{cases}
            q_i &\text{, if } j=i-1 \text{ and } i>0,\\
            r_i &\text{, if } j=i,\\
            p_i &\text{, if } j=i+1 \text{ and } i<d-1.
        \end{cases}

    The annihilation probability of state :math:`i=1` must not be zero, same for the creation probability
    of the last state :math:`i=n`. The sum of the probabilities must be bounded component-wise, i.e.,
    :math:`q_i + p_i \leq 1\;\forall i=1,\ldots ,n`.

    Parameters
    ----------
    q : array_like
        Annihilation probabilities for transition from i to i-1.
    p : array_like
        Creation probabilities for transition from i to i+1.

    Returns
    -------
    chain : deeptime.data.birth_death_chain_dataset.BirthDeathChain
        The chain.
    """
    from deeptime.data.birth_death_chain_dataset import BirthDeathChain
    return BirthDeathChain(q, p)


def tmatrix_metropolis1d(energies, d=1.0):
    r"""Transition matrix describing the Metropolis chain jumping
    between neighbors in a discrete 1D energy landscape.

    Parameters
    ----------
    energies : (M,) ndarray
        Energies in units of kT
    d : float (optional)
        Diffusivity of the chain, d in (0, 1]

    Returns
    -------
    P : (M, M) ndarray
        Transition matrix of the Markov chain

    Notes
    -----
    Transition probabilities are computed as

    .. math::

        \begin{aligned}
        p_{i,i-1} &= 0.5 d \min \left\{ 1.0, \mathrm{e}^{-(E_{i-1} - E_i)} \right\}, \\
        p_{i,i+1} &= 0.5 d \min \left\{ 1.0, \mathrm{e}^{-(E_{i+1} - E_i)} \right\}, \\
        p_{i,i}   &= 1.0 - p_{i,i-1} - p_{i,i+1}.
        \end{aligned}

    """
    import math
    # check input
    if d <= 0 or d > 1:
        raise ValueError('Diffusivity must be in (0,1]. Trying to set the invalid value', str(d))
    # init
    n = len(energies)
    transition_matrix = np.zeros((n, n))
    # set off diagonals
    transition_matrix[0, 1] = 0.5 * d * min(1.0, math.exp(-(energies[1] - energies[0])))
    for i in range(1, n - 1):
        transition_matrix[i, i - 1] = 0.5 * d * min(1.0, math.exp(-(energies[i - 1] - energies[i])))
        transition_matrix[i, i + 1] = 0.5 * d * min(1.0, math.exp(-(energies[i + 1] - energies[i])))
    transition_matrix[n - 1, n - 2] = 0.5 * d * min(1.0, math.exp(-(energies[n - 2] - energies[n - 1])))
    # normalize
    transition_matrix += np.diag(1.0 - np.sum(transition_matrix, axis=1))
    # done
    return transition_matrix
