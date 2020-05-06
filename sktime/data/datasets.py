from .double_well import DoubleWellDiscrete
from .ellipsoids import Ellipsoids


def double_well_discrete():
    r"""MCMC process in a symmetric double well potential, spatially discretized to 100 bins.
    The discrete trajectory contains 100000 steps, discrete time step dt=10. The result object allows access to
    discretizations of varying quality as well as gives opportunity to synthetically generate more data.

    Returns
    -------
    dataset : DoubleWellDiscrete
        an object that contains a markov state model corresponding to the process and the discrete trajectory
    """
    return DoubleWellDiscrete()


def ellipsoids(laziness=0.97, seed=None):
    r""" Example data of a two-state markov chain which can be featurized into two parallel ellipsoids and optionally
    rotated into higher-dimensional space.

    See :class:`Ellipsoids <sktime.data.Ellipsoids>` for more details.

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
    dataset : Ellipsoids
        an object that contains methods to create discrete and continuous observations
    """
    return Ellipsoids(laziness=laziness, seed=seed)
