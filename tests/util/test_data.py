import pytest
from numpy.testing import assert_equal, assert_raises, assert_, assert_array_almost_equal

pytest.importorskip("torch")

from deeptime.util.data import timeshifted_split, TrajectoryDataset, TimeLaggedDataset
import numpy as np


def test_astype():
    data = np.zeros((5, 2), dtype=np.float64)
    ds = TimeLaggedDataset(data, data)
    assert_equal(ds.data.dtype, np.float64)
    assert_equal(ds.data_lagged.dtype, np.float64)
    ds = ds.astype(np.float32)
    assert_equal(ds.data.dtype, np.float32)
    assert_equal(ds.data_lagged.dtype, np.float32)


def test_timeshifted_split_wrong_args():
    data = [np.zeros(shape=(100, 3), dtype=np.float32),
            np.zeros(shape=(10, 3), dtype=np.float32)]
    with assert_raises(ValueError):  # negative chunksize
        list(timeshifted_split(data, lagtime=1, chunksize=-1))
    with assert_raises(ValueError):  # too long lagtime
        list(timeshifted_split(data, lagtime=15))
    with assert_raises(ValueError):  # too long lagtime
        list(timeshifted_split(data, lagtime=10))
    list(timeshifted_split(data, lagtime=9))  # sanity this should not raise


@pytest.mark.parametrize("data", [np.arange(N) for N in [5, 6, 7, 8, 9, 10]],
                         ids=lambda N: "len={}".format(len(N)))
def test_timeshifted_split_chunksize(data):
    chunks = []
    chunks_lagged = []
    for X, Y in timeshifted_split(data, lagtime=1, chunksize=2):
        chunks.append(X)
        chunks_lagged.append(Y)
        np.testing.assert_(0 < len(X) <= 2)
        np.testing.assert_(0 < len(Y) <= 2)
    np.testing.assert_equal(np.concatenate(chunks), data[:-1])
    np.testing.assert_equal(np.concatenate(chunks_lagged), data[1:])


@pytest.mark.parametrize("data", [np.arange(N) for N in [5, 6, 7, 8, 9, 10]],
                         ids=lambda N: "len={}".format(len(N)))
def test_timeshifed_split_nsplits(data):
    chunks = []
    chunks_lagged = []
    n = 0
    for X, Y in timeshifted_split(data, lagtime=1, n_splits=2):
        chunks.append(X)
        chunks_lagged.append(Y)
        n += 1
    np.testing.assert_equal(n, 2)
    np.testing.assert_equal(np.concatenate(chunks), data[:-1])
    np.testing.assert_equal(np.concatenate(chunks_lagged), data[1:])


def test_timeshifted_split_nolag():
    x = np.arange(5000)
    splits = []
    for chunk in timeshifted_split(x, 0, n_splits=3):
        splits.append(chunk)

    np.testing.assert_equal(np.concatenate(splits), x)
    np.testing.assert_equal(len(splits), 3)
    for i in range(3):
        np.testing.assert_(len(splits[i]) > 0)


@pytest.mark.parametrize('lagtime', [0, 5], ids=lambda x: f"lagtime={x}")
@pytest.mark.parametrize('n_splits', [3, 7, 23], ids=lambda x: f"n_splits={x}")
def test_timeshifted_split_shuffle(lagtime, n_splits):
    x = np.arange(31, 5000)
    chunks = []
    chunks_lagged = []
    for chunk in timeshifted_split(x, lagtime=lagtime, n_splits=23, shuffle=True):
        if lagtime > 0:
            chunks.append(chunk[0])
            chunks_lagged.append(chunk[1])
        else:
            chunks.append(chunk)
            chunks_lagged.append(chunk)
    chunks = np.concatenate(chunks)
    chunks_lagged = np.concatenate(chunks_lagged)
    np.testing.assert_equal(len(chunks), len(x) - lagtime)  # we lose lagtime many frames
    np.testing.assert_equal(len(chunks_lagged), len(x) - lagtime)  # we lose lagtime many frames
    np.testing.assert_equal(chunks + lagtime, chunks_lagged)  # since data is sequential this must hold
    all_data = np.concatenate((chunks, chunks_lagged))  # check whether everything combined is the full dataset
    np.testing.assert_equal(len(np.setdiff1d(x, all_data)), 0)


@pytest.mark.parametrize("lagtime", [1, 5])
def test_timelagged_dataset(lagtime):
    pytest.importorskip("torch.utils.data")
    import torch.utils.data as data_utils
    data = np.arange(5000)
    ds = TrajectoryDataset(lagtime, data)
    np.testing.assert_equal(len(ds), 5000 - lagtime)
    sub_datasets = data_utils.random_split(ds, [1000, 2500, 1500 - lagtime])

    collected_data = []
    for sub_dataset in sub_datasets:
        loader = data_utils.DataLoader(sub_dataset, batch_size=123)
        for batch in loader:
            if lagtime > 0:
                np.testing.assert_(isinstance(batch, (list, tuple)))
                collected_data.append(batch[0].numpy())
                collected_data.append(batch[1].numpy())
            else:
                collected_data.append(batch.numpy())
    collected_data = np.unique(np.concatenate(collected_data))
    np.testing.assert_equal(len(np.setdiff1d(collected_data, data)), 0)


@pytest.mark.parametrize("lagtime", [1, 5], ids=lambda x: f"lag={x}")
@pytest.mark.parametrize("ntraj", [1, 2, 3], ids=lambda x: f"ntraj={x}")
@pytest.mark.parametrize("stride", [None, 1, 2, 3], ids=lambda x: f"stride={x}")
@pytest.mark.parametrize("start", [None, 0, 1], ids=lambda x: f"start={x}")
@pytest.mark.parametrize("stop", [None, 50], ids=lambda x: f"stop={x}")
def test_timelagged_dataset_multitraj(lagtime, ntraj, stride, start, stop):
    data = [np.random.normal(size=(7, 3)), np.random.normal(size=(555, 3)), np.random.normal(size=(55, 3))]
    data = data[:ntraj]
    assert_(len(data) == ntraj)
    with assert_raises(AssertionError):
        TrajectoryDataset.from_trajectories(1, [])  # empty data
    with assert_raises(AssertionError):
        TrajectoryDataset.from_trajectories(lagtime=7, data=data)  # lagtime too long
    with assert_raises(AssertionError):
        TrajectoryDataset.from_trajectories(lagtime=1, data=data + [np.empty((55, 7))])  # shape mismatch
    ds = TrajectoryDataset.from_trajectories(lagtime=lagtime, data=data)
    assert len(ds) == sum(len(data[i]) - lagtime for i in range(len(data)))

    # Iterate over data and see if it is the same as iterating over dataset
    out_full = ds[::]
    out_strided = ds[start:stop:stride]

    # we manually iterate over trajectories and collect them in time-lagged fashion
    X = []
    Y = []
    for traj in data:
        X.append(traj[:-lagtime])
        Y.append(traj[lagtime:])
    X = np.concatenate(X)[start:stop:stride]
    Y = np.concatenate(Y)[start:stop:stride]

    # check that manually collected and dataset yielded data coincide
    assert_equal(len(X), len(out_strided[0]))
    assert_equal(len(Y), len(out_strided[1]))
    assert_array_almost_equal(X, out_strided[0])
    assert_array_almost_equal(Y, out_strided[1])

    # get array of indices based on slice
    slice_obj = slice(start, stop, stride).indices(len(ds))
    indices = np.array(range(*slice_obj))

    # iterate over indices
    for ix in indices:
        x, y = ds[ix]
        # check this against full output
        assert_equal(x, out_full[0][ix])
        assert_equal(y, out_full[1][ix])
