import numpy as _np

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
    args = list(args)[::-1]
    x = args.pop()
    i = 0
    while len(args):
        y = args.pop()
        try:
            x = _np.dot(x, y)
            i += 1
        except ValueError as ve:
            raise ValueError(f'argument {i} and {i + 1} are not shape compatible:\n{ve}')
    return x
