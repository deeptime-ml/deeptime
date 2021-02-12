import pytest

pytest.importorskip("torch")

from deeptime.data import timeshifted_split, TimeLaggedDataset
import numpy as np


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
    np.testing.assert_equal(len(chunks), len(x)-lagtime)  # we lose lagtime many frames
    np.testing.assert_equal(len(chunks_lagged), len(x)-lagtime)  # we lose lagtime many frames
    np.testing.assert_equal(chunks+lagtime, chunks_lagged)  # since data is sequential this must hold
    all_data = np.concatenate((chunks, chunks_lagged))  # check whether everything combined is the full dataset
    np.testing.assert_equal(len(np.setdiff1d(x, all_data)), 0)


@pytest.mark.parametrize("lagtime", [1, 5])
def test_timeseries_dataset(lagtime):
    pytest.importorskip("torch.utils.data")
    import torch.utils.data as data_utils
    data = np.arange(5000)
    ds = TimeLaggedDataset.from_trajectory(lagtime, data)
    np.testing.assert_equal(len(ds), 5000-lagtime)
    sub_datasets = data_utils.random_split(ds, [1000, 2500, 1500-lagtime])

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
