import numpy as np
from numpy.testing import assert_equal

import deeptime as dt


def test_quadruple_well_sanity():
    traj = dt.data.quadruple_well().trajectory(np.array([[0., 0.]]), 5)
    assert_equal(traj.shape, (5, 2))
    assert_equal(traj[0], np.array([0, 0]))


def test_triple_well_2d_sanity():
    traj = dt.data.triple_well_2d().trajectory([[-1., 0.]], 5)
    assert_equal(traj.shape, (5, 2))
    assert_equal(traj[0], np.array([-1, 0]))
