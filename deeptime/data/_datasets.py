from typing import Tuple, List, Union, Callable, Optional

import numpy as np

from ._birth_death_chain import BirthDeathChain
from ._double_well import DoubleWellDiscrete
from ._ellipsoids import Ellipsoids
from ._pbf_simulator import PBFSimulator
from ._drunkards_walk_simulator import DrunkardsWalk
from ._bickley_simulator import BickleyJet, BickleyJetDataset
from ._systems import TimeDependentSystem, TimeIndependentSystem, CustomSystem


def double_well_discrete():
    r"""MCMC process in a symmetric double well potential, spatially discretized to 100 bins.
    The discrete trajectory contains 100000 steps, discrete time step dt=10. The result object allows access to
    discretizations of varying quality as well as gives opportunity to synthetically generate more data.

    .. plot:: datasets/plot_double_well_discrete.py

    Returns
    -------
    dataset : deeptime.data.DoubleWellDiscrete
        an object that contains a markov state model corresponding to the process and the discrete trajectory
    """
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

    .. plot:: datasets/plot_ellipsoids.py

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
    dataset : deeptime.data.Ellipsoids
        an object that contains methods to create discrete and continuous observations

    Examples
    --------
    >>> import deeptime as dt
    >>> feature_trajectory = dt.data.ellipsoids(seed=17).observations(n_steps=500)
    >>> assert feature_trajectory.shape == (500, 2)
    """
    return Ellipsoids(laziness=laziness, seed=seed)


def position_based_fluids(n_burn_in=5000, initial_positions=None, n_jobs=None):
    r""" Creates a position based fluids (PBF) simulator. It was introduced in :footcite:`macklin2013position`.
    Up to numerics the simulation is deterministic.

    The simulation box has dimensions :math:`[-40, 40]\times [-25, 25]` and the initial positions of the particles are
    around the top boundary of the box. For simplicity of use, the initial positions are fixed in this method and yield
    972 particles. For custom positioning, please use the simulator directly.

    The interaction distance is set to :math:`d = 1.5` and `n_burn_in` steps are
    performed to equilibrate the system before returning the simulator.

    For more details see :class:`PBFSimulator <deeptime.data.PBFSimulator>`.

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
    initial_positions : ndarray, optional, default=None
        Explicit initial positions, optional.
    n_jobs : int or None, default=None
        Number of threads to use for simulation.

    Returns
    -------
    simulator : deeptime.data.PBFSimulator
        The PBF simulator.

    References
    ----------
    .. footbibliography::
    """
    from deeptime.util.parallel import handle_n_jobs
    n_jobs = handle_n_jobs(n_jobs)
    interaction_distance = 1.5
    if initial_positions is None:
        init_pos_x = np.arange(-24, 24, interaction_distance * .9).astype(np.float32)
        init_pos_y = np.arange(-12, 24, interaction_distance * .9).astype(np.float32)
        initial_positions = np.dstack(np.meshgrid(init_pos_x, init_pos_y)).reshape(-1, 2)
    domain = np.array([80, 50])
    pbf = PBFSimulator(domain_size=domain, initial_positions=initial_positions,
                       interaction_distance=interaction_distance,
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

    .. plot:: datasets/plot_drunkards_walk.py

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
    simulator : deeptime.data.DrunkardsWalk
        Simulator instance.
    """
    return DrunkardsWalk(grid_size, bar_location=bar_location, home_location=home_location)


def bickley_jet(n_particles: int, n_jobs: Optional[int] = None, seed: Optional[int] = None) -> BickleyJetDataset:
    r"""Simulates the Bickley jet for a number of particles.
    The implementation is based on :footcite:`hadjighasem2016spectral` with parameters

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
    n_jobs : int or None, default=None
        Number of threads to use for simulation.
    seed : int or None, optional, default=None
        Random seed used for initialization of particle positions at :math:`t=0`.

    Returns
    -------
    dataset : BickleyJetDataset
        Dataset over all the generated frames.

    See Also
    --------
    BickleyJet
        Underlying trajectory generator.

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
    .. footbibliography::
    """
    from deeptime.util.parallel import handle_n_jobs
    n_jobs = handle_n_jobs(n_jobs)
    simulator = BickleyJet(h=1e-2, n_steps=10)
    traj = simulator.generate(n_particles=n_particles, n_jobs=n_jobs, seed=seed)
    traj_reshaped = traj.transpose(1, 0, 2)
    return BickleyJetDataset(traj_reshaped)


def birth_death_chain(q, p, sparse=False):
    r""" Generates a birth and death chain simulator from annihilation and creation probabilities `q` and `p`.

    A general birth and death chain on a d-dimensional state space has the transition matrix

    .. math::

        p_{ij} = \begin{cases}
            q_i &\text{, if } j=i-1 \text{ and } i>0,\\
            r_i &\text{, if } j=i,\\
            p_i &\text{, if } j=i+1 \text{ and } i < d-1
        \end{cases}


    The annihilation probability of state :math:`i=1` must not be zero, same for the creation probability
    of the last state :math:`i=n`. The sum of the probabilities must be bounded component-wise, i.e.,
    :math:`q_i + p_i \leq 1\;\forall i=1,\ldots ,n`.

    .. plot:: datasets/plot_birth_death_chain.py

    Parameters
    ----------
    q : array_like
        Annihilation probabilities for transition from i to i-1.
    p : array_like
        Creation probabilities for transition from i to i+1.
    sparse : bool, optional, default=False
        Whether to use sparse matrices.

    Returns
    -------
    chain : deeptime.data.BirthDeathChain
        The chain.
    """
    return BirthDeathChain(q, p, sparse=sparse)


def tmatrix_metropolis1d(energies, d=1.0):
    r"""Transition matrix describing the Metropolis chain jumping
    between neighbors in a discrete 1D energy landscape.

    .. plot:: datasets/plot_tmatrix_1d.py

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
    assert 0 < d <= 1, f'Diffusivity must be in (0,1]. Trying to set the invalid value {d}.'
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

    .. plot:: datasets/plot_sqrt_model.py

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

    The observations are generated via the means are :math:`\mu_0 = (0, 1)^\top` and :math:`\mu_1= (0, -1)^\top`,
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


def swissroll_model(n_samples, seed=None):
    r""" Sample a hidden state and an swissroll-transformed emission trajectory, so that the states are not
    linearly separable.

    .. plot::

        import matplotlib.pyplot as plt
        from matplotlib import animation
        from deeptime.data import swissroll_model

        n_samples = 15000
        dtraj, traj = swissroll_model(n_samples)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*traj.T, marker='o', s=20, c=dtraj, alpha=0.6)

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
        P = \begin{pmatrix}0.95 & 0.05 &      &      \\ 0.05 & 0.90 & 0.05 &      \\
                                & 0.05 & 0.90 & 0.05 \\      &      & 0.05 & 0.95 \end{pmatrix}.

    The observations are generated via the means are :math:`\mu_0 = (7.5, 7.5)^\top`, :math:`\mu_1= (7.5, 15)^\top`,
    :math:`\mu_2 = (15, 15)^\top`, and :math:`\mu_3 = (15, 7.5)^\top`,
    respectively, as well as the covariance matrix

    .. math::
        C = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}.

    Afterwards, the trajectory is transformed via

    .. math::
        (x, y) \mapsto (x \cos (x), y, x \sin (x))^\top.
    """
    from deeptime.markov.msm import MarkovStateModel

    state = np.random.RandomState(seed)
    cov = swissroll_model.cov
    states = swissroll_model.states
    msm = MarkovStateModel(swissroll_model.transition_matrix)
    dtraj = msm.simulate(n_samples, seed=seed)
    traj = states[dtraj, :] + state.multivariate_normal(np.zeros(len(cov)), cov, size=len(dtraj),
                                                        check_valid='ignore')
    x = traj[:, 0]
    return dtraj, np.vstack([x * np.cos(x), traj[:, 1], x * np.sin(x)]).T


swissroll_model.cov = np.array([[1.0, 0.0], [0.0, 1.0]])
swissroll_model.states = np.array([[7.5, 7.5], [7.5, 15.0], [15.0, 15.0], [15.0, 7.5]])
swissroll_model.transition_matrix = np.array([[0.95, 0.05, 0.00, 0.00], [0.05, 0.90, 0.05, 0.00],
                                              [0.00, 0.05, 0.90, 0.05], [0.00, 0.00, 0.05, 0.95]])


def quadruple_well(h: float = 1e-3, n_steps: int = 10000):
    r""" This dataset generates trajectories of a two-dimensional particle living in a quadruple-well potential
    landscape. It is subject to the stochastic differential equation

    .. math::

        \mathrm{d}X_t = \nabla V(X_t) \mathrm{d}t + \sqrt{2\beta^{-1}}\mathrm{d}W_t

    with :math:`W_t` being a Wiener process and the potential :math:`V` being given by

    .. math::

        V(x) = (x_1 - 1)^2 + (x_2 - 1)^2.

    The inverse temperature is set to be :math:`\beta = 4`.

    .. plot:: datasets/plot_quadruple_well.py

    Parameters
    ----------
    h : float, default = 1e-3
        Integration step size. The implementation uses an Euler-Maruyama integrator.
    n_steps : int, default = 10000
        Number of integration steps between each evaluation. That means the default lag time is :code:`h*n_steps=10`.

    Returns
    -------
    model : TimeIndependentSystem
        The model.

    Examples
    --------
    The model possesses the capability to simulate trajectories as well as be evaluated at test points:

    >>> import numpy as np
    >>> import deeptime as dt

    First, set up the model (which internally already creates the integrator).

    >>> model = dt.data.quadruple_well(h=1e-3, n_steps=100)  # create model instance

    Now, a trajectory can be generated:

    >>> traj = model.trajectory([0., 0.], 1000, seed=42, n_jobs=1)  # simulate trajectory
    >>> assert traj.shape == (1000, 2)  # 1000 evaluations from initial condition [0, 0]

    Or, alternatively the model can be evaluated at test points (mapping forward using the dynamical system):

    >>> test_points = np.random.uniform(-2, 2, (100, 2))  # 100 test point in [-2, 2] x [-2, 2]
    >>> evaluations = model(test_points, seed=53, n_jobs=1)
    >>> assert evaluations.shape == (100, 2)
    """
    from ._data_bindings import QuadrupleWell2D
    return TimeIndependentSystem(QuadrupleWell2D(), h, n_steps)


def quadruple_well_asymmetric(h=1e-3, n_steps=10000):
    r""" This dataset generates trajectories of a two-dimensional particle living in an asymmetric quadruple-well
    potential landscape. It is subject to the stochastic differential equation

    .. math::

        \mathrm{d}X_t = \nabla V(X_t) \mathrm{d}t + \sigma\mathrm{d}W_t

    with :math:`W_t` being a Wiener process and the potential :math:`V` being given by

    .. math::

        V(x) = (x_1^4-\frac{x_1^3}{16}-2x_1^2+\frac{3x_1}{16}) + (x_2^4-\frac{x_1^3}{8}-2x_1^2+\frac{3x_1}{8}).

    The stochastic force parameter is set to :math:`\sigma = 0.6`.

    .. plot:: datasets/plot_quadruple_well_asymmetric.py

    Parameters
    ----------
    h : float, default = 1e-3
        Integration step size. The implementation uses an Euler-Maruyama integrator.
    n_steps : int, default = 10000
        Number of integration steps between each evaluation. That means the default lag time is :code:`h*n_steps=10`.

    Returns
    -------
    model : TimeIndependentSystem
        The model.

    Examples
    --------
    The model possesses the capability to simulate trajectories as well as be evaluated at test points:

    >>> import numpy as np
    >>> import deeptime as dt

    First, set up the model (which internally already creates the integrator).

    >>> model = dt.data.quadruple_well_asymmetric(h=1e-3, n_steps=100)  # create model instance

    Now, a trajectory can be generated:

    >>> traj = model.trajectory(np.array([[0., 0.]]), 1000, seed=42, n_jobs=1)  # simulate trajectory
    >>> assert traj.shape == (1000, 2)  # 1000 evaluations from initial condition [0, 0]

    Or, alternatively the model can be evaluated at test points (mapping forward using the dynamical system):

    >>> test_points = np.random.uniform(-2, 2, (100, 2))  # 100 test point in [-2, 2] x [-2, 2]
    >>> evaluations = model(test_points, seed=53, n_jobs=1)
    >>> assert evaluations.shape == (100, 2)
    """
    from ._data_bindings import QuadrupleWellAsymmetric2D
    return TimeIndependentSystem(QuadrupleWellAsymmetric2D(), h, n_steps)


def triple_well_2d(h=1e-5, n_steps=10000):
    r""" This dataset generates trajectories of a two-dimensional particle living in a triple-well potential
    landscape. The example can be found in :footcite:`schutte2013metastability`.

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

    .. plot:: datasets/plot_triple_well_2d.py

    Parameters
    ----------
    h : float, default = 1e-5
        Integration step size. The implementation uses an Euler-Maruyama integrator.
    n_steps : int, default = 10000
        Number of integration steps between each evaluation. That means the default lag time is :code:`h*n_steps=0.1`.

    Returns
    -------
    model : TimeIndependentSystem
        The model.

    Examples
    --------
    The model possesses the capability to simulate trajectories as well as be evaluated at test points:

    >>> import numpy as np
    >>> import deeptime as dt

    First, set up the model (which internally already creates the integrator).

    >>> model = dt.data.triple_well_2d(h=1e-3, n_steps=100)  # create model instance

    Now, a trajectory can be generated:

    >>> traj = model.trajectory(np.array([[-1., 0.]]), 1000, seed=42, n_jobs=1)  # simulate trajectory
    >>> assert traj.shape == (1000, 2)  # 1000 evaluations from initial condition [0, 0]

    Or, alternatively the model can be evaluated at test points (mapping forward using the dynamical system):

    >>> test_points = np.random.uniform(-2, 2, (100, 2))  # 100 test point in [-2, 2] x [-2, 2]
    >>> evaluations = model(test_points, seed=53, n_jobs=1)
    >>> assert evaluations.shape == (100, 2)

    References
    ----------
    .. footbibliography::
    """
    from ._data_bindings import TripleWell2D
    return TimeIndependentSystem(TripleWell2D(), h, n_steps)


def abc_flow(h=1e-3, n_steps=10000):
    r""" The Arnold-Beltrami-Childress flow. :footcite:`arnold1966topology`
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
    system : TimeIndependentSystem
        The system.

    Examples
    --------
    The model possesses the capability to simulate trajectories as well as be evaluated at test points:

    >>> import numpy as np
    >>> import deeptime as dt

    First, set up the model (which internally already creates the integrator).

    >>> model = dt.data.abc_flow(h=1e-3, n_steps=100)  # create model instance

    Now, a trajectory can be generated:

    >>> traj = model.trajectory(np.array([[-1., 0., 0.]]), 1000, seed=42, n_jobs=1)  # simulate trajectory
    >>> assert traj.shape == (1000, 3)  # 1000 evaluations from initial condition [0, 0]

    Or, alternatively the model can be evaluated at test points (mapping forward using the dynamical system):

    >>> test_points = np.random.uniform(.5, 1.5, (100, 3))  # 100 test points
    >>> evaluations = model(test_points, seed=53, n_jobs=1)
    >>> assert evaluations.shape == (100, 3)

    References
    ----------
    .. footbibliography::
    """
    from ._data_bindings import ABCFlow
    return TimeIndependentSystem(ABCFlow(), h, n_steps)


def ornstein_uhlenbeck(h=1e-3, n_steps=500):
    r""" The one-dimensional Ornstein-Uhlenbeck process. It is given by the stochastic differential equation

    .. math::

        dX_t = -\alpha X_t dt + \sqrt{2\beta^{-1}}dW_t

    with parameters :math:`\alpha=1` and :math:`\beta=4`.

    .. plot:: datasets/plot_ornstein_uhlenbeck.py

    Parameters
    ----------
    h : float, default = 1e-3
        Integration step size. The implementation uses an Euler-Maruyama integrator.
    n_steps : int, default = 500
        Number of integration steps between each evaluation. That means the default lag time is :code:`h*n_steps=10`.

    Returns
    -------
    system : TimeIndependentSystem
        The system.

    Examples
    --------
    The model possesses the capability to simulate trajectories as well as be evaluated at test points:

    >>> import numpy as np
    >>> import deeptime as dt

    First, set up the model (which internally already creates the integrator).

    >>> model = dt.data.ornstein_uhlenbeck(h=1e-3, n_steps=100)  # create model instance

    Now, a trajectory can be generated:

    >>> traj = model.trajectory(np.array([[-1.]]), 1000, seed=42, n_jobs=1)  # simulate trajectory
    >>> assert traj.shape == (1000, 1)  # 1000 evaluations from initial condition [0, 0]

    Or, alternatively the model can be evaluated at test points (mapping forward using the dynamical system):

    >>> test_points = np.random.uniform(.5, 1.5, (100, 1))  # 100 test points
    >>> evaluations = model(test_points, seed=53, n_jobs=1)
    >>> assert evaluations.shape == (100, 1)
    """
    from ._data_bindings import OrnsteinUhlenbeck
    return TimeIndependentSystem(OrnsteinUhlenbeck(), h, n_steps)


def prinz_potential(h=1e-5, n_steps=500, temperature_factor=1., mass=1., damping=1.):
    r""" Particle diffusing in a one-dimensional quadruple well potential landscape. :footcite:`prinz2011markov`

    The potential is defined as

    .. math::
        V(x) = 4 \left( x^8 + 0.8 e^{-80 x^2} + 0.2 e^{-80 (x-0.5)^2} + 0.5 e^{-40 (x+0.5)^2}\right).

    The integrator is an Euler-Maruyama type integrator, updating the current state :math:`x_t` via

    .. math::
        x_{t+1} =  x_t - \frac{h\nabla V(x_t)}{m\cdot d} + \sqrt{2\frac{h\cdot\mathrm{kT}}{m\cdot d}}\eta_t,

    where :math:`m` is the mass, :math:`d` the damping factor, and :math:`\eta_t \sim \mathcal{N}(0, 1)`.

    The locations of the minima can be accessed via the `minima` attribute.

    .. plot:: datasets/plot_prinz.py

    Parameters
    ----------
    h : float, default=1e-5
        The integrator step size. If the temperature is too high and the step size too large, the
        integrator may produce NaNs.
    n_steps : int, default=500
        Number of integration steps between each evaluation of the system's state.
    temperature_factor : float, default=1
        The temperature kT.
    mass : float, default=1
        The particle's mass.
    damping : float, default=1
        Damping factor.

    Returns
    -------
    system : TimeIndependentSystem
        The system.

    Examples
    --------
    The model possesses the capability to simulate trajectories as well as be evaluated at test points:

    >>> import numpy as np
    >>> from deeptime.data import prinz_potential

    First, set up the model (which internally already creates the integrator).

    >>> model = prinz_potential(h=1e-3, n_steps=100)  # create model instance

    Now, a trajectory can be generated:

    >>> traj = model.trajectory(np.array([[-0.]]), 1000, seed=42, n_jobs=1)  # simulate trajectory
    >>> assert traj.shape == (1000, 1)  # 1000 evaluations from initial condition [0]

    Or, alternatively the model can be evaluated at test points (mapping forward using the dynamical system):

    >>> test_points = np.random.uniform(.5, 1.5, (100, 1))  # 100 test points
    >>> evaluations = model(test_points, seed=53, n_jobs=1)
    >>> assert evaluations.shape == (100, 1)

    References
    ----------
    .. footbibliography::
    """
    from ._data_bindings import Prinz
    system = TimeIndependentSystem(Prinz(), h, n_steps, props={
        'kT': temperature_factor,
        'mass': mass,
        'damping': damping
    })
    system.minima = [-0.73943018, -0.22373758, 0.26914935, 0.67329636]
    return system


def triple_well_1d(h=1e-3, n_steps=500):
    r""" A simple one-dimensional triple-well potential landscape. It is given by the stochastic differential equation

    .. math::

        \mathrm{d}X_t = \nabla V(X_t) \mathrm{d}t + \sigma(t, X_t)\mathrm{d}W_t

    with :math:`W_t` being a Wiener process, :math:`\sigma = 1.09`, and the potential :math:`V` being given by

    .. math::

        V(x) = 5 - 24.82 x + 41.4251 x^2 - 27.5344 x^3 + 8.53128 x^4 - 1.24006 x^5 + 0.0684 x^6.

    .. plot:: datasets/plot_triple_well_1d.py

    Parameters
    ----------
    h : float, default=1e-3
        Integration step size.
    n_steps : int, default=500
        Default number of integration steps per evaluation.

    Returns
    -------
    system : TimeIndependentSystem
        The system.
    """
    from ._data_bindings import TripleWell1D
    return TimeIndependentSystem(TripleWell1D(), h, n_steps)


def double_well_2d(h=1e-3, n_steps=10000):
    r""" This dataset generates trajectories of a two-dimensional particle living in a double-well potential
    landscape.

    The particle is subject to the stochastic differential equation

    .. math::

        \mathrm{d}X_t = \nabla V(X_t) \mathrm{d}t + \sigma(t, X_t)\mathrm{d}W_t

    with :math:`W_t` being a Wiener process, :math:`\sigma = 0.7`, and the potential :math:`V` being given by

    .. math::

        V(x) = (x_1^2 - 1)^2 + x_2^2.

    .. plot:: datasets/plot_double_well_2d.py

    Parameters
    ----------
    h : float, default = 1e-5
        Integration step size. The implementation uses an Euler-Maruyama integrator.
    n_steps : int, default = 10000
        Number of integration steps between each evaluation. That means the default lag time is :code:`h*n_steps=0.1`.

    Returns
    -------
    model : TimeIndependentSystem
        The model.

    Examples
    --------
    The model possesses the capability to simulate trajectories as well as be evaluated at test points:

    >>> import numpy as np
    >>> import deeptime as dt

    First, set up the model (which internally already creates the integrator).

    >>> model = dt.data.double_well_2d(h=1e-3, n_steps=100)  # create model instance

    Now, a trajectory can be generated:

    >>> traj = model.trajectory(np.array([[-1., 0.]]), 1000, seed=42, n_jobs=1)  # simulate trajectory
    >>> assert traj.shape == (1000, 2)  # 1000 evaluations from initial condition [0, 0]

    Or, alternatively the model can be evaluated at test points (mapping forward using the dynamical system):

    >>> test_points = np.random.uniform(-2, 2, (100, 2))  # 100 test point in [-2, 2] x [-2, 2]
    >>> evaluations = model(test_points, seed=53, n_jobs=1)
    >>> assert evaluations.shape == (100, 2)
    """
    from ._data_bindings import DoubleWell2D
    return TimeIndependentSystem(DoubleWell2D(), h, n_steps)


def time_dependent_quintuple_well(h=1e-3, n_steps=10000, beta=5.):
    r"""This dataset generates trajectories of a two-dimensional particle living in a time-dependent quintuple-well
    potential landscape. The potential wells are slowly oscillating around the origin.

    The dynamics are described by the potential landscape

    .. math::
        V(t, x, y) = \cos \left(s\;\mathrm{arctan}(y, x) - \frac{\pi}{2}t \right) + 10 \left( \sqrt{x^2+y^2} - \frac{3}{2} - \frac{1}{2}\sin(2\pi t) \right)^2

    subject to the SDE

    .. math::
        dX_t = -\nabla V(X_t, t) dt + \sqrt{2\beta^{-1}}dW_t,

    where :math:`\beta` is the temperature and :math:`W_t` a Wiener process.

    Parameters
    ----------
    h : float, optional, default=1e-3
        Integration step size.
    n_steps : int, optional, default=10000
        Number of steps to evaluate between recording states.
    beta : float, default=5.
        The inverse temperature.

    Returns
    -------
    system : TimeDependentSystem
        The system.

    Examples
    --------
    The model possesses the capability to simulate trajectories as well as be evaluated at test points:

    >>> import numpy as np
    >>> import deeptime as dt

    First, set up the model (which internally already creates the integrator).

    >>> model = dt.data.time_dependent_quintuple_well(h=1e-3, n_steps=100, beta=5.)  # create model instance

    Now, a trajectory can be generated:

    >>> traj = model.trajectory(0., np.array([[-1., 0.]]), 1000, seed=42, n_jobs=1)  # simulate trajectory
    >>> assert traj.shape == (1000, 2)  # 1000 evaluations from initial condition [0, 0] with t=0

    Or, alternatively the model can be evaluated at test points (mapping forward using the dynamical system):

    >>> test_points = np.random.uniform(-2, 2, (100, 2))  # 100 test point in [-2, 2] x [-2, 2]
    >>> evaluations = model(0, test_points, seed=53, n_jobs=1)  # start at t=0
    >>> assert evaluations.shape == (100, 2)
    """
    from ._data_bindings import TimeDependent5Well2D
    return TimeDependentSystem(TimeDependent5Well2D(), h, n_steps, props={'beta': beta})


def custom_sde(dim: int, rhs: Callable, sigma: np.ndarray, h: float, n_steps: int):
    r""" This function allows the definition of custom stochastic differential equations (SDEs) of the form

    .. math::

        \mathrm{d}X_t = F(X_t) \mathrm{d}t + \sigma(t, X_t)\mathrm{d}W_t,

    where the right-hand side :math:`F` should map an :code:`dim`-dimensional array-like object to an
    :code:`dim`-dimensional array-like object. The prefactor in front of the Wiener process :math:`W_t` is assumed
    to be constant with respect to time and state, i.e.,

    .. math::

        \sigma(t, X_t) = \sigma \in\mathbb{R}^{\mathrm{dim}\times\mathrm{dim}}.

    .. plot:: datasets/plot_custom_sde.py

    Parameters
    ----------
    dim : int, positive and less or equal to 5
        The dimension of the SDE's state vector :math:`X_t`. Must be less or equal to 5.
    rhs : Callable
        The right-hand side function :math:`F(X_t)`. It must map a dim-dimensional array like object to a
        dim-dimensional array or list.
    sigma : (dim, dim) ndarray
        The sigma parameter.
    h : float
        Step size for the Euler-Maruyama integrator.
    n_steps : int
        Number of integration steps per evaluation / recording of the state.

    Returns
    -------
    system : CustomSystem
        The system.

    Examples
    --------
    First, some imports.

    >>> import deeptime as dt
    >>> import numpy as np

    Then, we can define the right-hand side. Here, we choose the force of an harmonic spherical inclusion potential.

    >>> def harmonic_sphere_force(x, radius=.5, k=1.):
    ...     dist_to_origin = np.linalg.norm(x)
    ...     dist_to_sphere = dist_to_origin - radius
    ...     if dist_to_sphere > 0:
    ...         return -k * dist_to_sphere * np.array(x) / dist_to_origin
    ...     else:
    ...         return [0., 0.]

    This we can use as right-hand side to define our SDE with :math:`\sigma = \mathrm{diag}(1, 1)`.

    >>> sde = dt.data.custom_sde(dim=2, rhs=lambda x: harmonic_sphere_force(x, radius=.5, k=1),
    ...                          sigma=np.diag([1., 1.]), h=1e-3, n_steps=1)

    Here, :code:`h` is the step-size of the (Euler-Maruyama) integrator and :code:`n_steps` refers to the number of
    integration steps for each evaluation.
    Given the SDE instance, we can generate trajectories via

    >>> trajectory = sde.trajectory(x0=[[0., 0.]], length=10, seed=55)
    >>> assert trajectory.shape == (10, 2)

    or propagate (in this case 300) sample points by :code:`n_steps`:

    >>> propagated_samples = sde(np.random.normal(scale=.1, size=(300, 2)))
    >>> assert propagated_samples.shape == (300, 2)
    """
    from . import _data_bindings as bindings
    if not (isinstance(dim, int) and 0 < dim <= 5):
        raise ValueError("Dimension must be positive and at most 5.")

    sigma = np.atleast_2d(np.array(sigma).squeeze())
    sigma_shape = sigma.shape
    if not sigma_shape == (dim, dim):
        raise ValueError("Sigma must be DIM x DIM matrix but had shape", sigma_shape)

    SDE = getattr(bindings, f'PySDE{dim}D')
    return CustomSystem(SDE(sigma, rhs), h, n_steps)


def custom_ode(dim: int, rhs: Callable, h: float, n_steps: int):
    r""" This function allows the definition of custom ordinary differential equations (ODEs) of the form

    .. math::

        \mathrm{d}X_t = F(X_t) \mathrm{d}t,

    where the right-hand side :math:`F` should map an :code:`dim`-dimensional array-like object to an
    :code:`dim`-dimensional array-like object.

    .. plot:: datasets/plot_custom_ode.py

    Parameters
    ----------
    dim : int, positive and less or equal to 5
        The dimension of the SDE's state vector :math:`X_t`. Must be less or equal to 5.
    rhs : Callable
        The right-hand side function :math:`F(X_t)`. It must map a dim-dimensional array like object to a
        dim-dimensional array or list.
    h : float
        Step size for the Runge-Kutta integrator.
    n_steps : int
        Number of integration steps per evaluation / recording of the state.

    Returns
    -------
    system : CustomSystem
        The system.

    Examples
    --------
    First, some imports.

    >>> import numpy as np
    >>> import deeptime as dt

    We can define the right-hand side to model an exponential decay

    >>> def rhs(x):
    ...     return [-.5 * x[0]]

    and obtain the system

    >>> system = dt.data.custom_ode(dim=1, rhs=rhs, h=1e-3, n_steps=20)

    where :code:`n_steps` is the number of (Runge-Kutta 45) integration steps per evaluation and :code:`h` the
    step-size. With the system, one can generate trajectories

    >>> traj = system.trajectory(x0=[[1.]], length=50, seed=45)
    >>> assert traj.shape == (50, 1)

    as well as propagate sample points by :code:`n_steps`:

    >>> propagated_samples = system(np.random.uniform(size=(100, 1)))
    >>> assert propagated_samples.shape == (100 ,1)
    """
    from . import _data_bindings as bindings
    if not (isinstance(dim, int) and 0 < dim <= 5):
        raise ValueError("Dimension must be positive and at most 5.")

    ODE = getattr(bindings, f'PyODE{dim}D')
    return CustomSystem(ODE(rhs), h, n_steps)
