import numpy as np


def timeshifted_split(inputs, lagtime: int, chunksize=1000, n_splits=None):
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
