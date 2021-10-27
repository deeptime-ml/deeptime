from typing import Optional, List

import numpy as np

from ..base import Dataset


def timeshifted_split(inputs, lagtime: int, chunksize: int = 1000, stride: int = 1, n_splits: Optional[int] = None,
                      shuffle: bool = False, random_state: Optional[np.random.RandomState] = None):
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
    shuffle : bool, default=False
        Whether to shuffle the data prior to splitting it.
    random_state : np.random.RandomState, default=None
        When shuffling this can be used to set a specific random state.

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
        raise ValueError('lagtime has to be non-negative')
    if int(chunksize) < 0:
        raise ValueError('chunksize has to be positive')

    if shuffle and random_state is None:
        random_state = np.random.RandomState()

    if not isinstance(inputs, list):
        inputs = [inputs]

    if not all(len(data) > lagtime for data in inputs):
        too_short_inputs = [i for i, x in enumerate(inputs) if len(x) <= lagtime]
        raise ValueError(f'Input contained to short (smaller than lagtime({lagtime}) at following '
                         f'indices: {too_short_inputs}')

    for data in inputs:
        data = np.asarray_chkfinite(data)
        data_lagged = data[lagtime:][::stride]
        if lagtime > 0:
            data = data[:-lagtime][::stride]
        else:
            data = data[0::stride]  # otherwise data is empty as slice over `data[:-0]`
        ix = np.arange(len(data))  # iota range over data
        if shuffle:
            random_state.shuffle(ix)

        if n_splits is not None:
            assert n_splits >= 1
            for ix_split in np.array_split(ix, n_splits):
                if len(ix_split) > 0:
                    x = data[ix_split]
                    if lagtime > 0:
                        x_lagged = data_lagged[ix_split]
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
                x = data[ix[t:min(t + chunksize, len(data))]]
                if lagtime > 0:
                    x_lagged = data_lagged[ix[t:min(t + chunksize, len(data_lagged))]]
                    yield x, x_lagged
                else:
                    yield x
                t += chunksize


class ConcatDataset(Dataset):
    r""" Concatenates existing datasets.

    Parameters
    ----------
    datasets : list of dataset
        Datasets to concatenate.
    """

    def __init__(self, datasets: List[Dataset]):
        self._lengths = [len(ds) for ds in datasets]
        self._cumlen = np.cumsum(self._lengths)
        self._datasets = datasets

    def setflags(self, write=True):
        for ds in self._datasets:
            ds.setflags(write=write)

    @property
    def subsets(self):
        r""" Returns the list of datasets this concat dataset is composed of.

        :type: list of dataset
        """
        return self._datasets

    def _dataset_index(self, ix):
        from bisect import bisect_right
        ds_index = bisect_right(self._cumlen, ix)
        item_index = ix if ds_index == 0 else ix - self._cumlen[ds_index - 1]
        return ds_index, item_index

    def __getitem__(self, ix):
        ds_index, item_index = self._dataset_index(ix)
        return self._datasets[ds_index][item_index]

    def __len__(self):
        return self._cumlen[-1]


class TimeLaggedDataset(Dataset):
    r""" High-level container for time-lagged time-series data.
    This can be used together with pytorch data tools, i.e., data loaders and other utilities.

    Parameters
    ----------
    data : (T, n) ndarray
        The data which is wrapped into a dataset
    data_lagged : (T, m) ndarray
        Corresponding timelagged data. Must be of same length.

    See Also
    --------
    TimeLaggedConcatDataset, TrajectoryDataset, TrajectoriesDataset
    """

    def __init__(self, data, data_lagged):
        assert len(data) == len(data_lagged), \
            f"Length of trajectory for data and data_lagged does not match ({len(data)} != {len(data_lagged)})"
        self._data = data
        self._data_lagged = data_lagged

    def setflags(self, write=True):
        self._data.setflags(write=write)
        self._data_lagged.setflags(write=write)

    def astype(self, dtype):
        r""" Sets the datatype of contained arrays and returns a new instance of TimeLaggedDataset.

        Parameters
        ----------
        dtype
            The new dtype.

        Returns
        -------
        converted_ds : TimeLaggedDataset
            The dataset with converted dtype.
        """
        return TimeLaggedDataset(self._data.astype(dtype), self._data_lagged.astype(dtype))

    @property
    def data(self) -> np.ndarray:
        r""" Instantaneous data. """
        return self._data

    @property
    def data_lagged(self) -> np.ndarray:
        r""" Time-lagged data. """
        return self._data_lagged

    def __getitem__(self, item):
        return self._data[item], self._data_lagged[item]

    def __len__(self):
        return len(self._data)


class TimeLaggedConcatDataset(ConcatDataset):
    r""" Specialization of the :class:`ConcatDataset` which uses that all subsets are time lagged datasets, offering
    fancy and more efficient slicing / getting items.

    Parameters
    ----------
    datasets : list of TimeLaggedDataset
        The input datasets
    """

    def __init__(self, datasets: List[TimeLaggedDataset]):
        assert all(isinstance(x, TimeLaggedDataset) for x in datasets)
        super().__init__(datasets)

    @staticmethod
    def _compute_overlap(stride, traj_len, skip):
        r""" Given two trajectories :math:`T_1` and :math:`T_2`, this function calculates for the first trajectory
        an overlap, i.e., a skip parameter for :math:`T_2` such that the trajectory fragments
        :math:`T_1` and :math:`T_2` appear as one under the given stride.

        :param stride: the (global) stride parameter
        :param traj_len: length of T_1
        :param skip: skip of T_1
        :return: skip of T_2

        Notes
        -----
        Idea for deriving the formula: It is

        .. code::
            K = ((traj_len - skip - 1) // stride + 1) = #(data points in trajectory of length (traj_len - skip)).

        Therefore, the first point's position that is not contained in :math:`T_1` anymore is given by

        .. code::
            pos = skip + s * K.

        Thus the needed skip of :math:`T_2` such that the same stride parameter makes :math:`T_1` and :math:`T_2`
        "look as one" is

        .. code::
            overlap = pos - traj_len.
        """
        return stride * ((traj_len - skip - 1) // stride + 1) - traj_len + skip

    def __getitem__(self, ix):
        if isinstance(ix, slice):
            xs, ys = [], []
            end_ds, end_ix = self._dataset_index(ix.stop if ix.stop is not None else len(self))
            start_ds, start = self._dataset_index(ix.start if ix.start is not None else 0)
            stride = ix.step if ix.step is not None else 1
            for ds in range(start_ds, end_ds + 1):
                stop_ix = self._lengths[ds] if ds != end_ds else end_ix

                if stop_ix > start and ds < len(self._lengths):
                    local_slice = slice(start, stop_ix, stride)
                    xs.append(self._datasets[ds].data[local_slice])
                    ys.append(self._datasets[ds].data_lagged[local_slice])
                    start = self._compute_overlap(stride, self._lengths[ds], start)
            return np.concatenate(xs), np.concatenate(ys)
        else:
            return super().__getitem__(ix)


class TrajectoryDataset(TimeLaggedDataset):
    r"""Creates a trajectory dataset from a single trajectory by applying a lagtime.

    Parameters
    ----------
    lagtime : int
        Lagtime, must be positive. The effective size of the dataset reduces by the selected lagtime.
    trajectory : (T, d) ndarray
        Trajectory with T frames in d dimensions.

    Raises
    ------
    AssertionError
        If lagtime is not positive or trajectory is too short for lagtime.
    """

    def __init__(self, lagtime, trajectory):
        assert lagtime > 0, "Lagtime must be positive"
        assert len(trajectory) > lagtime, "Not enough data to satisfy lagtime"
        super().__init__(trajectory[:-lagtime], trajectory[lagtime:])
        self._trajectory = trajectory
        self._lagtime = lagtime

    @property
    def lagtime(self):
        return self._lagtime

    @property
    def trajectory(self):
        return self._trajectory

    @staticmethod
    def from_trajectories(lagtime, data: List[np.ndarray]):
        r""" Creates a time series dataset from multiples trajectories by applying a lagtime.

        Parameters
        ----------
        lagtime : int
            Lagtime, must be positive. The effective size of the dataset reduces by the selected lagtime.
        data : list of ndarray
            List of trajectories.

        Returns
        -------
        dataset : TrajectoriesDataset
            Concatenation of timeseries datasets.

        Raises
        ------
        AssertionError
            If data is empty, lagtime is not positive,
            the shapes do not match, or lagtime is too long for any of the trajectories.
        """
        return TrajectoriesDataset.from_numpy(lagtime, data)


class TrajectoriesDataset(TimeLaggedConcatDataset):
    r""" Dataset composed of multiple trajectories.

    Parameters
    ----------
    data : list of TrajectoryDataset
        The trajectories in form of trajectory datasets.

    See Also
    --------
    TrajectoryDataset.from_trajectories
        Method to create a TrajectoriesDataset from multiple raw data trajectories.
    """

    def __init__(self, data: List[TrajectoryDataset]):
        assert len(data) > 0, "List of data should not be empty."
        assert all(x.lagtime == data[0].lagtime for x in data), "Lagtime must agree"
        super().__init__(data)

    @staticmethod
    def from_numpy(lagtime, data: List[np.ndarray]):
        r""" Creates a time series dataset from multiples trajectories by applying a lagtime.

        Parameters
        ----------
        lagtime : int
            Lagtime, must be positive. The effective size of the dataset reduces by the selected lagtime.
        data : list of ndarray
            List of trajectories.

        Returns
        -------
        dataset : TrajectoriesDataset
            Concatenation of timeseries datasets.

        Raises
        ------
        AssertionError
            If data is empty, lagtime is not positive,
            the shapes do not match, or lagtime is too long for any of the trajectories.
        """
        assert len(data) > 0 and all(data[0].shape[1:] == x.shape[1:] for x in data), "Shape mismatch!"
        return TrajectoriesDataset([TrajectoryDataset(lagtime, traj) for traj in data])

    @property
    def lagtime(self):
        r""" The lagtime.

        :type: int
        """
        return self.subsets[0].lagtime

    @property
    def trajectories(self):
        r""" Contained raw trajectories.

        :type: list of ndarray
        """
        return [x.trajectory for x in self.subsets]
