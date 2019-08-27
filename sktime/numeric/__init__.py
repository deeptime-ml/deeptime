import numpy as np


def mdot(*args):
    """Computes a matrix product of multiple ndarrays

    This is a convenience function to avoid constructs such as np.dot(A, np.dot(B, np.dot(C, D))) and instead
    use mdot(A, B, C, D).

    Parameters
    ----------
    *args : an arbitrarily long list of ndarrays that must be compatible for multiplication,
        i.e. args[i].shape[1] = args[i+1].shape[0].
    """
    if len(args) < 1:
        raise ValueError('need at least one argument')
    elif len(args) == 1:
        return args[0]
    elif len(args) == 2:
        return np.dot(args[0], args[1])
    else:
        return np.dot(args[0], mdot(*args[1:]))
