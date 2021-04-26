import numpy as np

from ..util.parallel import handle_n_jobs


class SystemBase:

    def __init__(self, system, h: float, n_steps: int):
        r""" Creates a new wrapper. """
        self._impl = system
        self._impl.h = h
        self._impl.n_steps = n_steps

    @property
    def impl(self):
        r""" The native object. """
        return self._impl

    @property
    def dimension(self) -> int:
        r""" The dimension of the system's state.

        :type: int
        """
        return self._impl.dimension

    @property
    def time_dependent(self) -> bool:
        r""" Whether the potential (if available) and the right-hand side depend on time.

        :type: bool
        """
        return self._impl.time_dependent

    @property
    def integrator(self) -> str:
        r""" The type of integrator.

        :type: str
        """
        return self._impl.integrator

    @property
    def has_potential_function(self) -> bool:
        r""" Whether the system defines a potential.
        This means that the deterministic part of the right-hand side is the negative gradient of that potential.
        """
        return self._impl.has_potential_function


class _TimeIndependentRhsMixin:
    def trajectory(self: SystemBase, x0, length, seed=-1, n_jobs=None):
        r""" Simulates one or multiple trajectories.

        Parameters
        ----------
        x0 : array_like
            The initial condition. Must be compatible in shape to a (n_test_points, dimension)-array.
        length : int
            The length of the trajectory that is to be generated.
        seed : int, optional, default=-1
            The random seed. In case it is specified to be something else than `-1`, n_jobs must be set to `n_jobs=1`.
        n_jobs : int, optional, default=None
            Specify number of jobs according to :meth:`deeptime.util.parallel.handle_n_jobs`.

        Returns
        -------
        trajectory : np.ndarray
            The trajectory of shape (n_initial_conditions, n_evaluations, dimension).
            In case of just one initial condition the shape is squeezed to (n_evaluations, dimension).
        """
        n_jobs = handle_n_jobs(n_jobs)
        x0 = np.array(x0).reshape((-1, self.impl.dimension))
        n_initial_conditions = x0.shape[0]
        traj = self.impl.trajectory(x0, length, seed, n_jobs)
        if n_initial_conditions == 1:
            traj = traj[0]
        return traj

    def __call__(self: SystemBase, test_points, seed=-1, n_jobs=None):
        r"""Evolves the provided tests points under the dynamic for n_steps and returns.

        Parameters
        ----------
        test_points : array_like
            The test points.
        seed : int, optional, default=-1
            The seed for reproducibility. In case it is set to `seed >= 0` the number of jobs needs
            to be fixed to `n_jobs=1`.
        n_jobs : int, optional, default=None
            Specify number of jobs according to :meth:`deeptime.util.parallel.handle_n_jobs`.

        Returns
        -------
        points : np.ndarray
            The evolved test points.
        """
        n_jobs = handle_n_jobs(n_jobs)
        test_points = np.array(test_points).reshape((-1, self.impl.dimension))
        return self.impl(test_points, seed, n_jobs)

    def potential(self: SystemBase, points):
        r""" Evaluates the system's potential function at given points in state space.

        Parameters
        ----------
        points : array_like
            The points.

        Returns
        -------
        energy : np.ndarray
            The energy for each of these points.

        Raises
        ------
        AssertionError
            If the system does not have a potential function defined.
        """
        assert self.has_potential_function
        points = np.array(points).reshape((-1, self.dimension))
        return self.impl.potential(points)


class _TimeDependentRhsMixin:
    def trajectory(self: SystemBase, t0, x0, length, seed=-1, n_jobs=None):
        r""" Simulates one or multiple trajectories depending on initial state and initial time.

        Parameters
        ----------
        t0 : array_like
            The initial time. Can be picked as single float across all test points or individually.
        x0 : array_like
            The initial condition. Must be compatible in shape to a (n_test_points, dimension)-array.
        length : int
            The length of the trajectory that is to be generated.
        seed : int, optional, default=-1
            The random seed. In case it is specified to be something else than `-1`, n_jobs must be set to `n_jobs=1`.
        n_jobs : int, optional, default=None
            Specify number of jobs according to :meth:`deeptime.util.parallel.handle_n_jobs`.

        Returns
        -------
        trajectory : np.ndarray
            The trajectory of shape (n_initial_conditions, n_evaluations, dimension).
            In case of just one initial condition the shape is squeezed to (n_evaluations, dimension).
        """
        n_jobs = handle_n_jobs(n_jobs)
        x0 = np.array(x0).reshape((-1, self.impl.dimension))
        n_initial_conditions = x0.shape[0]

        t0 = np.atleast_1d(t0)
        if len(t0) == 1 and n_initial_conditions > 1:
            t0 = np.full((n_initial_conditions,), t0[0])
        traj = self.impl.trajectory(t0, x0, length, seed, n_jobs)
        if n_initial_conditions == 1:
            traj = traj[0]
        return traj

    def __call__(self: SystemBase, t0, test_points, seed=-1, n_jobs=None):
        r"""Evolves the provided tests points under the dynamic for n_steps and returns.

        Parameters
        ----------
        t0 : array_like
            The initial time. Can be picked as single float across all test points or individually.
        test_points : array_like
            The test points.
        seed : int, optional, default=-1
            The seed for reproducibility. In case it is set to `seed >= 0` the number of jobs needs
            to be fixed to `n_jobs=1`.
        n_jobs : int, optional, default=None
            Specify number of jobs according to :meth:`deeptime.util.parallel.handle_n_jobs`.

        Returns
        -------
        points : np.ndarray
            The evolved test points.
        """
        n_jobs = handle_n_jobs(n_jobs)
        test_points = np.array(test_points).reshape((-1, self.impl.dimension))
        n_test_points = len(test_points)

        t0 = np.atleast_1d(t0)
        if len(t0) == 1 and n_test_points > 1:
            t0 = np.full((n_test_points,), t0[0])

        return self.impl(t0, test_points, seed, n_jobs)

    def potential(self: SystemBase, time, points):
        r""" Evaluates the system's potential function at given points in state space at a specified time.

        Parameters
        ----------
        time : float
            The evaluation time.
        points : array_like
            The points.

        Returns
        -------
        energy : np.ndarray
            The energy for each of these points.

        Raises
        ------
        AssertionError
            If the system does not have a potential function defined.
        """
        assert self.has_potential_function
        points = np.array(points).reshape((-1, self.dimension))
        return self.impl.potential(time, points)


class TimeDependentSystem(SystemBase, _TimeDependentRhsMixin):
    r""" Wraps systems with a time-dependent right-hand side defined in extension code.

    Parameters
    ----------
    system
        The implementation.
    h : float
        Integration step size.
    n_steps : int
        Number of steps between each evaluation of the state.
    """

    def __init__(self, system, h: float, n_steps: int):
        super().__init__(system, h, n_steps)


class TimeIndependentSystem(SystemBase, _TimeIndependentRhsMixin):
    r""" Wraps systems with a time-independent right-hand side defined in extension code.

    Parameters
    ----------
    system
        The implementation.
    h : float
        Integration step size.
    n_steps : int
        Number of steps between each evaluation of the state.
    """

    def __init__(self, system, h: float, n_steps: int):
        super().__init__(system, h, n_steps)


class CustomSystem(TimeIndependentSystem):
    r""" A system as yielded by :meth:`custom_sde <deeptime.data.custom_sde>`
    or :meth:`custom_ode <deeptime.data.custom_ode>`.
    """

    def trajectory(self: SystemBase, x0, length, seed=-1, **kw):
        return super().trajectory(x0, length, seed, n_jobs=1)

    def __call__(self: SystemBase, test_points, seed=-1, **kw):
        return super().__call__(test_points, seed, n_jobs=1)
