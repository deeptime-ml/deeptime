from typing import Tuple

import numpy as np


def double_well_discrete():
    r"""MCMC process in a symmetric double well potential, spatially discretized to 100 bins.
    The discrete trajectory contains 100000 steps, discrete time step dt=10. The result object allows access to
    discretizations of varying quality as well as gives opportunity to synthetically generate more data.

    Returns
    -------
    dataset : sktime.data.double_well_dataset.DoubleWellDiscrete
        an object that contains a markov state model corresponding to the process and the discrete trajectory
    """
    from sktime.data.double_well_dataset import DoubleWellDiscrete
    return DoubleWellDiscrete()


def ellipsoids(laziness: float = 0.97, seed=None):
    r""" Example data of a two-state markov chain which can be featurized into two parallel ellipsoids and optionally
    rotated into higher-dimensional space.

    See :class:`Ellipsoids <sktime.data.ellipsoids_dataset.Ellipsoids>` for more details.

    .. plot::

       import matplotlib.pyplot as plt
       import sktime

       ftraj = sktime.data.ellipsoids(seed=17).observations(1000)
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
    dataset : sktime.data.ellipsoids_dataset.Ellipsoids
        an object that contains methods to create discrete and continuous observations
    """
    from sktime.data.ellipsoids_dataset import Ellipsoids
    return Ellipsoids(laziness=laziness, seed=seed)


def position_based_fluids(n_burn_in=5000, n_jobs=None):
    r""" Creates a position based fluids (PBF) simulator. It was introduced in :cite:`data-api-macklin2013position`.
    Up to numerics the simulation is deterministic.

    The simulation box has dimensions :math:`[-40, 40]\times [-25, 25]` and the initial positions of the particles are
    around the top boundary of the box. For simplicity of use, the initial positions are fixed in this method and yield
    972 particles. For custom positioning, please use the simulator directly.

    The interaction distance is set to :math:`d = 1.5` and `n_burn_in` steps are
    performed to equilibrate the system before returning the simulator.

    For more details see :class:`PBFSimulator <sktime.data.pbf_simulator.PBFSimulator>`.

    .. plot::

        import matplotlib.pyplot as plt
        import sktime

        ftraj = sktime.data.position_based_fluids(n_burn_in=150).run(300)
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
    simulator : sktime.data.pbf_simulator.PBFSimulator
        The PBF simulator.

    References
    ----------
    .. bibliography:: /references.bib
        :style: unsrt
        :filter: docname in docnames
        :keyprefix: data-api-
    """
    from sktime.data.pbf_simulator import PBFSimulator
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
                   bar_location: Tuple[int, int] = (9, 9),
                   home_location: Tuple[int, int] = (0, 0)):
    r"""This example dataset simulates the steps a drunkard living in a two-dimensional plane takes finding
    either the bar or the home as two absorbing states.

    The drunkard can take steps in a 3x3 grid with uniform probability (as possible, in the corners the only
    possibilities are the ones that do not lead out of the grid). The transition matrix
    :math:`P\in\mathbb{R}^{nm\times nm}`  possesses one absorbing state for home and bar, respectively,
    and uniform two-dimensional jump probabilities in between. The grid is of size :math:`n\times m` and a point
    :math:`(i,j)` is identified with state :math:`i+nj` in the transition matrix.

    .. plot::

        import numpy as np
        import sktime

        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        import scipy
        from scipy.interpolate import CubicSpline

        sim = sktime.data.drunkards_walk(bar_location=(0, 0), home_location=(9, 9))
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
    simulator : sktime.data.drunkards_walk_simulator.DrunkardsWalk
        Simulator instance.
    """
    from sktime.data.drunkards_walk_simulator import DrunkardsWalk
    return DrunkardsWalk(grid_size, bar_location=bar_location, home_location=home_location)
