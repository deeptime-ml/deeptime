import pytest
import numpy as np
import sktime.data.util as util


@pytest.mark.parametrize("data", [np.arange(N) for N in [5, 6, 7, 8, 9, 10]],
                         ids=lambda N: "len={}".format(len(N)))
def test_timeshifted_split_chunksize(data):
    chunks = []
    chunks_lagged = []
    for X, Y in util.timeshifted_split(data, lagtime=1, chunksize=2):
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
    for X, Y in util.timeshifted_split(data, lagtime=1, n_splits=2):
        chunks.append(X)
        chunks_lagged.append(Y)
        n += 1
    np.testing.assert_equal(n, 2)
    np.testing.assert_equal(np.concatenate(chunks), data[:-1])
    np.testing.assert_equal(np.concatenate(chunks_lagged), data[1:])


def test_timeshifted_split_nolag():
    x = np.arange(5000)
    splits = []
    for chunk in util.timeshifted_split(x, 0, n_splits=3):
        splits.append(chunk)

    np.testing.assert_equal(np.concatenate(splits), x)
    np.testing.assert_equal(len(splits), 3)
    for i in range(3):
        np.testing.assert_(len(splits[i]) > 0)
