from typing import Optional

import numpy as np


def timeshifted_split(inputs, lagtime: int, chunksize: int = 1000, stride: int = 1, n_splits: Optional[int] = None):
    r""" Utility function which splits input trajectories into pairs of timeshifted data :math:`(X_t, X_{t+\tau})`.
    In case multiple trajectories are provided, the timeshifted pairs are always within the same trajectory.

    Parameters
    ----------
    inputs : (T, n) ndarray or list of (T_i, n) ndarrays
        Input trajectory or trajectories. In case multiple trajectories are provided, they must have the same dimension
        in the second axis but may be of variable length.
    lagtime : int
        The lag time :math:`\tau` used to produce timeshifted blocks.
    chunksize : int, default=1000
        The chunk size, i.e., the maximal length of the blocks.
    stride: int, default=1
        Optional stride which is applied *after* creating a tau-shifted version of the dataset.
    n_splits : int, optional, default=None
        Alternative to chunksize - this determines the number of timeshifted blocks that is drawn from each provided
        trajectory. Supersedes whatever was provided as chunksize.
    Returns
    -------
    iterable : Generator
        A Python generator which can be iterated.

    Examples
    --------
    Using chunksize:

    >>> data = np.array([0, 1, 2, 3, 4, 5, 6])
    >>> for X, Y in timeshifted_split(data, lagtime=1, chunksize=4):
    ...     print(X, Y)
    [0 1 2 3] [1 2 3 4]
    [4 5] [5 6]

    Using n_splits:

    >>> data = np.array([0, 1, 2, 3, 4, 5, 6])
    >>> for X, Y in timeshifted_split(data, lagtime=1, n_splits=2):
    ...     print(X, Y)
    [0 1 2] [1 2 3]
    [3 4 5] [4 5 6]
    """
    if lagtime < 0:
        raise ValueError('lagtime has to be positive')
    if int(chunksize) < 0:
        raise ValueError('chunksize has to be positive')

    if not isinstance(inputs, list):
        if isinstance(inputs, tuple):
            inputs = list(inputs)
        inputs = [inputs]

    if not all(len(data) > lagtime for data in inputs):
        too_short_inputs = [i for i, x in enumerate(inputs) if len(x) < lagtime]
        raise ValueError(f'Input contained to short (smaller than lagtime({lagtime}) at following '
                         f'indices: {too_short_inputs}')

    for data in inputs:
        data = np.asarray_chkfinite(data)
        data_lagged = data[lagtime:][::stride]
        if lagtime > 0:
            data = data[:-lagtime][::stride]
        else:
            data = data[0::stride]  # otherwise data is empty as slice over `data[:-0]`

        if n_splits is not None:
            assert n_splits >= 1
            for x, x_lagged in zip(np.array_split(data, n_splits),
                                   np.array_split(data_lagged, n_splits)):
                if len(x) > 0:
                    assert len(x) == len(x_lagged)
                    if lagtime > 0:
                        yield x, x_lagged
                    else:
                        yield x
                else:
                    break
        else:
            t = 0
            while t < len(data):
                if t == len(data_lagged):
                    break
                if lagtime > 0:
                    yield data[t:min(t+chunksize, len(data))], data_lagged[t:min(t+chunksize, len(data_lagged))]
                else:
                    yield data[t:min(t+chunksize, len(data))]
                t += chunksize
