import numpy as np


def timeshifted_split(inputs, lagtime: int, chunksize=None, n_splits=None):
    assert lagtime > 0
    #assert (chunksize is not None) ^ (n_splits is not None)
    if chunksize is None and n_splits is None:
        chunksize = 500

    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    assert all(len(data) > lagtime for data in inputs)

    for data in inputs:
        data = np.asarray_chkfinite(data)

        n_splits = np.ceil(len(data) // min(len(data), chunksize)) if n_splits is None else n_splits
        assert n_splits >= 1, n_splits

        data_lagged = data[lagtime:]
        data = data[:-lagtime]

        for x, x_lagged in zip(np.array_split(data, n_splits),
                               np.array_split(data_lagged, n_splits)):
            if len(x) > 0:
                assert len(x) == len(x_lagged)
                yield x, x_lagged
            else:
                break
