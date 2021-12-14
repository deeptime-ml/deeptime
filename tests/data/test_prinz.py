import numpy as np
from numpy.testing import assert_equal

from deeptime.data import prinz_potential


def test_sanity():
    traj = prinz_potential(n_steps=500).trajectory([[1., 0.]], 50)
    assert_equal(traj.shape, (2, 50, 1))
    assert_equal(traj[0, 0, 0], 1.)  # test initial point for first traj is 1.
    assert_equal(traj[1, 0, 0], 0.)  # test initial point for secnd traj is 0.
    evals = prinz_potential()(np.random.uniform(-1, 1, size=(5, 1)))
    assert_equal(evals.shape, (5, 1))
