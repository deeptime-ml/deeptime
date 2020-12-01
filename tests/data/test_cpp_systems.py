import numpy as np
from numpy.testing import assert_equal

import deeptime as dt


def test_quadruple_well_sanity():
    traj = dt.data.quadruple_well().trajectory(np.array([[0., 0.]]), 5)
    assert_equal(traj.shape, (5, 2))
    assert_equal(traj[0], np.array([0, 0]))


def test_triple_well_2d_sanity():
    system = dt.data.triple_well_2d()
    traj = system.trajectory([[-1., 0.]], 5)
    assert_equal(traj.shape, (5, 2))
    assert_equal(traj[0], np.array([-1, 0]))

    x = np.linspace(-2, 2, num=50)
    y = np.linspace(-1, 2, num=50)
    xy = np.meshgrid(x, y)
    coords = np.dstack(xy).reshape(-1, 2)
    energies = system.potential(coords)
    
    def V(xv, yv):
        return + 3. * np.exp(- (xv * xv) - (yv - 1. / 3.) * (yv - 1. / 3.)) \
               - 3. * np.exp(- (xv * xv) - (yv - 5. / 3.) * (yv - 5. / 3.)) \
               - 5. * np.exp(-(xv - 1.) * (xv - 1.) - yv * yv) \
               - 5. * np.exp(-(xv + 1.) * (xv + 1.) - yv * yv) \
               + (2. / 10.) * xv * xv * xv * xv \
               + (2. / 10.) * np.power(yv - 1 / 3, 4)
    
    assert len(coords) == len(energies)
    for ix in range(len(coords)):
        x, y = coords[ix]
        ref_energy = V(x, y)
        np.testing.assert_allclose(energies[ix], ref_energy, err_msg=f'xy={x, y}')


def test_abc_flow_sanity():
    traj = dt.data.abc_flow().trajectory([[0., 0., 0.]], 5)
    assert_equal(traj.shape, (5, 3))
    assert_equal(traj[0], np.array([0., 0., 0.]))


def test_ornstein_uhlenbeck_sanity():
    traj = dt.data.ornstein_uhlenbeck().trajectory([[-1.]], 5)
    assert_equal(traj.shape, (5, 1))
    assert_equal(traj[0], np.array([-1]))
