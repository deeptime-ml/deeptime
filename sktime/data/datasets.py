from .double_well import DoubleWellDiscrete


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
