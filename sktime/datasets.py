import sktime.data.double_well as _double_well


def double_well_discrete():
    """
    MCMC process in a symmetric double well potential, spatially discretized to 100 bins.
    The discrete trajectory contains 100000 steps, discrete time step dt=10.
    Returns
    -------
    an object that contains a markov state model corresponding to the process and the discrete trajectory
    """
    return _double_well.DoubleWellDiscrete()
