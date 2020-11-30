from typing import Tuple, List, Union, Optional

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

    In particular, a synthetic trajectory of observations of states :math:`S = \{0, 1\}` can be generated from
    a :class:`MSM <deeptime.markov.msm.MarkovStateModel>`. The transition probabilities have to be chosen so that
    the chain is lazy, i.e., it is more likely to stay in one state than to transition to another.

    Optionally, a continuous observation chain can be generated with two parallel ellipsoidal multivariate normal
    distributions. In this case, the MSM acts as hidden markov state model with a Gaussian output model. For
    benchmark and demonstration purposes, this observation chain can be rotated into a higher dimensional space
    and equipped with additional noise.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import multivariate_normal

        import deeptime as dt

        data_source = dt.data.ellipsoids(seed=17)
        x = np.linspace(-10, 10, 1000)
        y = np.linspace(-10, 10, 1000)
        X, Y = np.meshgrid(x, y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        rv1 = multivariate_normal(data_source.state_0_mean, data_source.covariance_matrix)
        rv2 = multivariate_normal(data_source.state_1_mean, data_source.covariance_matrix)

        fig = plt.figure()
        ax = fig.gca()

        ax.contourf(X, Y, (rv1.pdf(pos) + rv2.pdf(pos)).reshape(len(x), len(y)))
        ax.autoscale(False)
        ax.set_aspect('equal')
        ax.scatter(*data_source.observations(100).T, color='cyan', marker='x', label='samples')
        plt.grid()
        plt.title(r'Ellipsoids dataset observations with laziness of $0.97$.')
        plt.legend()
        plt.show()

    Parameters
    ----------
    laziness : float in half-open interval (0.5, 1.], default=0.97
        The probability to stay in either state rather than transitioning. This yields a transition matrix of

        .. math:: P = \begin{pmatrix} \lambda & 1-\lambda \\ 1-\lambda & \lambda \end{pmatrix},

        where :math:`\lambda` is the selected laziness parameter.
    seed : int, optional, default=None
        Optional random seed for reproducibility.

    Returns
    -------
    dataset : deeptime.data.ellipsoids_dataset.Ellipsoids
        an object that contains methods to create discrete and continuous observations

    Examples
    --------
    >>> import deeptime as dt
    >>> feature_trajectory = dt.data.ellipsoids(seed=17).observations(n_steps=500)
    >>> assert feature_trajectory.shape == (500, 2)
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


def sqrt_model(n_samples, seed=None):
    r""" Sample a hidden state and an sqrt-transformed emission trajectory.
    We sample a hidden state trajectory and sqrt-masked emissions in two
    dimensions such that the two metastable states are not linearly separable.

    .. plot::

        import matplotlib.pyplot as plt

        import deeptime as dt

        n_samples = 30000
        dtraj, traj = dt.data.sqrt_model(n_samples)

        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.hexbin(*traj.T, bins=10, cmap='coolwarm')


    Parameters
    ----------
    n_samples : int
        Number of samples to produce.
    seed : int, optional, default=None
        Random seed to use. Defaults to None, which means that the random device will be default-initialized.

    Returns
    -------
    sequence : (n_samples, ) ndarray
        The discrete states.
    trajectory : (n_samples, ) ndarray
        The observable.

    Notes
    -----
    First, the hidden discrete-state trajectory is simulated. Its transition matrix is given by

    .. math::
        P = \begin{pmatrix}0.95 & 0.05 \\ 0.05 & 0.95 \end{pmatrix}.

    The observations are generated via the means are :math:`\mu_0 = (0, 1)^\top` and :math:`\mu_1= (0, -1)`,
    respectively, as well as the covariance matrix

    .. math::
        C = \begin{pmatrix} 30 & 0 \\ 0 & 0.015 \end{pmatrix}.

    Afterwards, the trajectory is transformed via

    .. math::
        (x, y) \mapsto (x, y + \sqrt{| x |}).
    """
    from deeptime.markov.msm import MarkovStateModel

    state = np.random.RandomState(seed)
    cov = sqrt_model.cov
    states = sqrt_model.states
    msm = MarkovStateModel(sqrt_model.transition_matrix)
    dtraj = msm.simulate(n_samples, seed=seed)
    traj = states[dtraj, :] + state.multivariate_normal(np.zeros(len(cov)), cov, size=len(dtraj),
                                                        check_valid='ignore')
    traj[:, 1] += np.sqrt(np.abs(traj[:, 0]))
    return dtraj, traj


sqrt_model.cov = np.array([[30.0, 0.0], [0.0, 0.015]])
sqrt_model.states = np.array([[0.0, 1.0], [0.0, -1.0]])
sqrt_model.transition_matrix = np.array([[0.95, 0.05], [0.05, 0.95]])


def quadruple_well(h: float = 1e-3, n_steps: int = 10000):
    r""" This dataset generates trajectories of a two-dimensional particle living in a quadruple-well potential
    landscape. It is subject to the stochastic differential equation

    .. math::

        \mathrm{d}X_t = \nabla V(X_t) \mathrm{d}t + \sqrt{2\beta^{-1}}\mathrm{d}W_t

    with :math:`W_t` being a Wiener process and the potential :math:`V` being given by

    .. math::

        V(x) = (x_1 - 1)^2 + (x_2 - 1)^2.

    The inverse temperature is set to be :math:`\beta = 4`.

    .. plot::

        import numpy as np
        import deeptime as dt
        import scipy
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        traj = dt.data.quadruple_well(n_steps=1000, seed=46).trajectory(np.array([[1, -1]]), 100)

        xy = np.arange(-2, 2, 0.1)
        XX, YY = np.meshgrid(xy, xy)
        V = (XX**2 - 1)**2 + (YY**2 - 1)**2

        fig, ax = plt.subplots(1, 1)
        ax.set_title("Example of a trajectory in the potential landscape")

        cb = ax.contourf(xy, xy, V, levels=np.linspace(0.0, 3.0, 20), cmap='coolwarm')

        x = np.r_[traj[:, 0]]
        y = np.r_[traj[:, 1]]
        f, u = scipy.interpolate.splprep([x, y], s=0, per=False)
        xint, yint = scipy.interpolate.splev(np.linspace(0, 1, 50000), f)

        points = np.stack([xint, yint]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        coll = LineCollection(segments, cmap='bwr')
        coll.set_array(np.linspace(0, 1, num=len(points), endpoint=True))
        coll.set_linewidth(1)
        ax.add_collection(coll)

        fig.colorbar(cb)

    Parameters
    ----------
    h : float, default = 1e-3
        Integration step size. The implementation uses an Euler-Maruyama integrator.
    n_steps : int, default = 10000
        Number of integration steps between each evaluation. That means the default lag time is :code:`h*n_steps=10`.

    Returns
    -------
    model : QuadrupleWell2D
        The model.

    Examples
    --------
    The model possesses the capability to simulate trajectories as well as be evaluated at test points:

    >>> import numpy as np
    >>> import deeptime as dt

    First, set up the model (which internally already creates the integrator).

    >>> model = dt.data.quadruple_well(h=1e-3, n_steps=100)  # create model instance

    Now, a trajectory can be generated:

    >>> traj = model.trajectory(np.array([[0., 0.]]), 1000, seed=42)  # simulate trajectory
    >>> assert traj.shape == (1000, 2)  # 1000 evaluations from initial condition [0, 0]

    Or, alternatively the model can be evaluated at test points (mapping forward using the dynamical system):

    >>> test_points = np.random.uniform(-2, 2, (100, 2))  # 100 test point in [-2, 2] x [-2, 2]
    >>> evaluations = model(test_points, seed=53, n_jobs=1)
    >>> assert evaluations.shape == (100, 2)
    """
    from ._data_bindings import QuadrupleWell2D
    system = QuadrupleWell2D()
    system.h = h
    system.n_steps = n_steps
    return system


def triple_well_2d(h=1e-5, n_steps=10000):
    r""" This dataset generates trajectories of a two-dimensional particle living in a triple-well potential
    landscape. The example can be found in :cite:`data-api-schutte2013metastability`.

    The particle is subject to the stochastic differential equation

    .. math::

        \mathrm{d}X_t = \nabla V(X_t) \mathrm{d}t + \sigma(t, X_t)\mathrm{d}W_t

    with :math:`W_t` being a Wiener process, :math:`\sigma = 1.09`, and the potential :math:`V` being given by

    .. math::

        \begin{aligned}
        V(x) &= 3e^{-x^2 - (y-\frac{1}{3})^2} - 3e^{-x^2 - (y - \frac{5}{3})^2} \\
        &\quad - 5e^{-(x-1)^2 - y^2} - 5e^{-(x+1)^2 - y^2} \\
        &\quad + \frac{2}{10} x^4 + \frac{2}{10}\left(y-\frac{1}{3}\right)^4.
        \end{aligned}

    .. plot::

        import numpy as np
        import deeptime as dt
        import scipy
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        traj = dt.data.triple_well_2d(n_steps=10000).trajectory(np.array([[-1, 0]]), 20, seed=42)

        x = np.arange(-2, 2, 0.1)
        y = np.arange(-1, 2, 0.1)
        XX, YY = np.meshgrid(x, y)
        V = 3*np.exp(-(XX**2) - (YY - 1/3)**2) - 3*np.exp(-XX**2 - (YY - 5/3)**2) \
            - 5*np.exp(-(XX-1)**2 - YY**2) - 5*np.exp(-(XX+1)**2 - YY**2) \
            + (2/10)*XX**4 + (2/10)*(YY-1/3)**4

        fig, ax = plt.subplots(1, 1)
        ax.set_title("Example of a trajectory in the potential landscape")

        cb = ax.contourf(x, y, V, levels=np.linspace(-4.5, 4.5, 20), cmap='coolwarm')

        x = np.r_[traj[:, 0]]
        y = np.r_[traj[:, 1]]
        f, u = scipy.interpolate.splprep([x, y], s=0, per=False)
        xint, yint = scipy.interpolate.splev(np.linspace(0, 1, 50000), f)

        points = np.stack([xint, yint]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        coll = LineCollection(segments, cmap='jet')
        coll.set_array(np.linspace(0, 1, num=len(points), endpoint=True))
        coll.set_linewidth(1)
        ax.add_collection(coll)

        fig.colorbar(cb)

    Parameters
    ----------
    h : float, default = 1e-5
        Integration step size. The implementation uses an Euler-Maruyama integrator.
    n_steps : int, default = 10000
        Number of integration steps between each evaluation. That means the default lag time is :code:`h*n_steps=0.1`.

    Returns
    -------
    model : TripleWell2D
        The model.

    Examples
    --------
    The model possesses the capability to simulate trajectories as well as be evaluated at test points:

    >>> import numpy as np
    >>> import deeptime as dt

    First, set up the model (which internally already creates the integrator).

    >>> model = dt.data.triple_well_2d(h=1e-3, n_steps=100)  # create model instance

    Now, a trajectory can be generated:

    >>> traj = model.trajectory(np.array([[-1., 0.]]), 1000, seed=42)  # simulate trajectory
    >>> assert traj.shape == (1000, 2)  # 1000 evaluations from initial condition [0, 0]

    Or, alternatively the model can be evaluated at test points (mapping forward using the dynamical system):

    >>> test_points = np.random.uniform(-2, 2, (100, 2))  # 100 test point in [-2, 2] x [-2, 2]
    >>> evaluations = model(test_points, seed=53, n_jobs=1)
    >>> assert evaluations.shape == (100, 2)

    References
    ----------
    .. bibliography:: /references.bib
        :style: unsrt
        :filter: docname in docnames
        :keyprefix: data-api-
    """
    from ._data_bindings import TripleWell2D
    system = TripleWell2D()
    system.h = h
    system.n_steps = n_steps
    return system


def abc_flow(h=1e-3, n_steps=10000):
    r""" The Arnold-Beltrami-Childress flow :cite:`data-api-arnold1966topology`.
    It is generated by the ODE

    .. math::

        \begin{aligned}
        \dot{x} &= A\sin(z) + C\cos(y)\\
        \dot{y} &= B\sin(x) + A\cos(z)\\
        \dot{z} &= C\sin(y) + B\cos(x)
        \end{aligned}

    on the domain :math:`\Omega=[0, 2\pi]^3` with the parameters :math:`A=\sqrt{3}`, :math:`B=\sqrt{2}`,
    and :math:`C=1`.

    .. plot::

        import numpy as np
        import deeptime as dt
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        system = dt.data.abc_flow(n_steps=100)
        scatters = [np.random.uniform(np.pi-.5, np.pi+.5, size=(200, 3))]
        for _ in range(12):
            scatters.append(system(scatters[-1], n_jobs=8))

        f = plt.figure(figsize=(18, 18))
        f.suptitle('Evolution of test points in the ABC flowfield')
        for i in range(4):
            for j in range(3):
                ix = j + i*3
                ax = f.add_subplot(4, 3, ix+1, projection='3d')
                ax.set_title(f"T={ix*system.n_steps*system.h:.2f}")
                ax.scatter(*scatters[ix].T, c=np.linspace(0, 1, num=200))
                ax.set_xlim([0, 2*np.pi])
                ax.set_ylim([0, 2*np.pi])
                ax.set_zlim([0, 2*np.pi])
        plt.show()


    Parameters
    ----------
    h : float, default = 1e-3
        Integration step size. The implementation uses an Runge-Kutta integrator.
    n_steps : int, default = 10000
        Number of integration steps between each evaluation. That means the default lag time is :code:`h*n_steps=10`.

    Returns
    -------
    system : ABCFlow
        The system.

    Examples
    --------
    The model possesses the capability to simulate trajectories as well as be evaluated at test points:

    >>> import numpy as np
    >>> import deeptime as dt

    First, set up the model (which internally already creates the integrator).

    >>> model = dt.data.abc_flow(h=1e-3, n_steps=100)  # create model instance

    Now, a trajectory can be generated:

    >>> traj = model.trajectory(np.array([[-1., 0.]]), 1000, seed=42)  # simulate trajectory
    >>> assert traj.shape == (1000, 3)  # 1000 evaluations from initial condition [0, 0]

    Or, alternatively the model can be evaluated at test points (mapping forward using the dynamical system):

    >>> test_points = np.random.uniform(.5, 1.5, (100, 3))  # 100 test points
    >>> evaluations = model(test_points, seed=53, n_jobs=1)
    >>> assert evaluations.shape == (100, 3)

    References
    ----------
    .. bibliography:: /references.bib
        :style: unsrt
        :filter: docname in docnames
        :keyprefix: data-api-
    """
    from ._data_bindings import ABCFlow
    system = ABCFlow()
    system.h = h
    system.n_steps = n_steps
    return system


def ornstein_uhlenbeck(h=1e-3, n_steps=500):
    r""" The one-dimensional Ornstein-Uhlenbeck process. It is given by the stochastic differential equation

    .. math::

        dX_t = -\alpha X_t dt + \sqrt{2\beta^{-1}}dW_t

    with parameters :math:`\alpha=1` and :math:`\beta=4`.

    Parameters
    ----------
    h : float, default = 1e-3
        Integration step size. The implementation uses an Euler-Maruyama integrator.
    n_steps : int, default = 500
        Number of integration steps between each evaluation. That means the default lag time is :code:`h*n_steps=10`.

    Returns
    -------
    system : OrnsteinUhlenbeck
        The system.

    Examples
    --------
    The model possesses the capability to simulate trajectories as well as be evaluated at test points:

    >>> import numpy as np
    >>> import deeptime as dt

    First, set up the model (which internally already creates the integrator).

    >>> model = dt.data.ornstein_uhlenbeck(h=1e-3, n_steps=100)  # create model instance

    Now, a trajectory can be generated:

    >>> traj = model.trajectory(np.array([[-1.]]), 1000, seed=42)  # simulate trajectory
    >>> assert traj.shape == (1000, 1)  # 1000 evaluations from initial condition [0, 0]

    Or, alternatively the model can be evaluated at test points (mapping forward using the dynamical system):

    >>> test_points = np.random.uniform(.5, 1.5, (100, 1))  # 100 test points
    >>> evaluations = model(test_points, seed=53, n_jobs=1)
    >>> assert evaluations.shape == (100, 1)
    """
    from ._data_bindings import OrnsteinUhlenbeck
    system = OrnsteinUhlenbeck()
    system.h = h
    system.n_steps = n_steps
    return system
