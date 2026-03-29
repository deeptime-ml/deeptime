from typing import Optional

import numpy as np

__author__ = "clonker"


class Ellipsoids:
    r""" Example data of a two-state markov chain which can be featurized into two parallel ellipsoids and optionally
    rotated into higher-dimensional space.

    Parameters
    ----------
    laziness : float in half-open interval (0.5, 1.], default=0.97
        The probability to stay in either state rather than transitioning. This yields a transition matrix of

        .. math:: P = \begin{pmatrix} \lambda & 1-\lambda \\ 1-\lambda & \lambda \end{pmatrix},

        where :math:`\lambda` is the selected laziness parameter.
    seed : int, optional, default=None
        Optional random seed for reproducibility.
    """

    state_0_mean = np.array([0., 0.])
    state_1_mean = np.array([0., 5.])

    def __init__(self, laziness: float = 0.97, seed: Optional[int] = None):
        if laziness <= 0.5 or laziness > 1:
            raise ValueError("Laziness must be at least 0.5 and at most 1.0 but was {}".format(laziness))
        transition_matrix = np.array([[laziness, 1-laziness], [1-laziness, laziness]])
        from deeptime.markov.msm import MarkovStateModel
        self._msm = MarkovStateModel(transition_matrix)
        self._rnd = np.random.RandomState(seed=seed)
        self._seed = seed
        self._cov = np.array([[5.7, 5.65], [5.65, 5.7]])

    @property
    def msm(self):
        r""" Yields the underlying markov state model. """
        return self._msm

    @property
    def random_state(self):
        r""" The random state that is used for RNG. """
        return self._rnd

    @property
    def seed(self):
        r""" Integer value of the seed or None. """
        return self._seed

    @property
    def covariance_matrix(self):
        r""" Covariance matrix that is used to parameterize a multivariate Gaussian distribution, resulting
        in the ellipsoidal shape. """
        return self._cov

    def discrete_trajectory(self, n_steps: int):
        r""" Generates a sequence of states 0 and 1 based on the internal markov state model.

        Parameters
        ----------
        n_steps : int
            The number of steps.

        Returns
        -------
        dtraj : (n_steps,) ndarray
            Time series of states.
        """
        if self.seed is not None:
            return self.msm.simulate(n_steps, seed=self.seed)
        else:
            return self.msm.simulate(n_steps)

    def observations(self, n_steps, n_dim=2, noise=False):
        r""" Generates an observation sequence in n_dim-dimensional space.

        Parameters
        ----------
        n_steps : int
            The number of observations.
        n_dim : int, default=2
            The dimension of the observation sequence. In case it is larger than 2, the 2-dimensional sequence
            is rotated with a random rotation matrix

            .. math::
                R = \begin{pmatrix} 0 & \cos(\alpha) & -\sin(\alpha) & \cdots \\
                                    0 & \sin(\alpha) & \cos(\alpha) & \ldots \end{pmatrix}
                \in\mathbb{R}^{2\times n}

            into :math:`n`-dimensional space, where :math:`\alpha\sim\mathcal{U}(0, 2\pi)`.
        noise : bool, default=False
            Optionally equips all observations with additional Gaussian noise of variance 0.2.

        Returns
        -------
        ftraj : (n_steps, n_dim) ndarray
            Observation trajectory.
        """
        if n_dim < 2:
            raise ValueError("Dimension must be at least 2 but was {}".format(n_dim))
        dtraj = self.discrete_trajectory(n_steps)
        return self.map_discrete_to_observations(dtraj, n_dim=n_dim, noise=noise)

    def map_discrete_to_observations(self, dtraj, n_dim=2, noise=False):
        r""" Maps a discrete trajectory (see :meth:`discrete_trajectory`) to an observation trajectory
        as described in :meth:`observations`)"""
        ftraj = np.empty((len(dtraj), 2))
        state_0_indices = np.where(dtraj == 0)[0]  # indices where the state is 0
        state_1_indices = np.where(dtraj == 1)[0]  # indices where the state is 1
        # fill allocated space with samples
        ftraj[state_0_indices] = self.random_state.multivariate_normal(
            mean=self.state_0_mean, cov=self.covariance_matrix, size=len(state_0_indices)
        )
        ftraj[state_1_indices] = self.random_state.multivariate_normal(
            mean=self.state_1_mean, cov=self.covariance_matrix, size=len(state_1_indices)
        )
        if n_dim == 2:
            # feature trajectory is already 2-dimensional
            if noise:
                ftraj += self.random_state.normal(scale=0.2, size=ftraj.shape)
            return ftraj
        else:
            # random rotation into higher-dimensional space
            angle = self.random_state.uniform(0, 2. * np.pi)
            rotation_matrix = np.zeros((2, n_dim))
            rotation_matrix[0, 1] = rotation_matrix[1, 2] = np.cos(angle)
            rotation_matrix[1, 1] = np.sin(angle)
            rotation_matrix[0, 2] = -np.sin(angle)

            ftraj = ftraj @ rotation_matrix
            assert ftraj.shape == (len(dtraj), n_dim)
            if noise:
                ftraj += self.random_state.normal(scale=0.2, size=ftraj.shape)
            return ftraj
