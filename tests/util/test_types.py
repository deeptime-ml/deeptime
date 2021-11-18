from contextlib import contextmanager

import numpy as np
import pytest
from numpy.testing import assert_, assert_equal, assert_raises

from deeptime.util.data import TrajectoryDataset
from deeptime.util.types import to_dataset


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize("data,lagtime,expectation", [
    (np.zeros((100, 5)), 5, does_not_raise()),
    (np.zeros((100, 5)), None, assert_raises(ValueError)),
    (np.zeros((100, 5)), 0, assert_raises(AssertionError)),
    (np.zeros((100, 5)), 96, assert_raises(AssertionError)),
    ((np.zeros((100, 5)), np.zeros((100, 5))), None, does_not_raise()),
    ((np.zeros((100, 5)), np.zeros((105, 5))), None, assert_raises(AssertionError)),
    (TrajectoryDataset.from_trajectories(5, [np.zeros((55, 5)), np.zeros((55, 5))]), None, does_not_raise())
], ids=[
    "Trajectory with lagtime",
    "Trajectory without lagtime",
    "Trajectory with zero lagtime",
    "Trajectory with too large lagtime",
    "X-Y tuple of data",
    "X-Y tuple of data, length mismatch",
    "Custom concat dataset of list of trajectories",
])
def test_to_dataset(data, lagtime, expectation):
    with expectation:
        ds = to_dataset(data, lagtime=lagtime)
        assert_(len(ds) in (100, 95))
        data = ds[:]
        assert_equal(len(data), 2)
        assert_equal(len(data[0]), len(ds))
        assert_equal(data[0].shape[1], 5)
        assert_equal(len(data[1]), len(ds))
        assert_equal(data[1].shape[1], 5)
