import numpy as np
import pytest
from numpy.testing import assert_equal, assert_raises

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


def test_quadruple_well_asymmetric_sanity():
    traj = dt.data.quadruple_well_asymmetric().trajectory([[1., 1.]], 5)
    assert_equal(traj.shape, (5, 2))
    assert_equal(traj[0], np.array([1., 1.]))


@pytest.mark.parametrize('dim', [1, 2, 3, 4, 5])
def test_custom_sde(dim):
    def rhs(x):
        return [0.] * dim

    sigma = np.diag([1.] * dim)

    with assert_raises(ValueError):
        wrong_sigma = np.diag([1.] * 99)
        dt.data.custom_sde(dim, rhs, wrong_sigma, h=1e-3, n_steps=5)

    sde = dt.data.custom_sde(dim, rhs, sigma, h=1e-3, n_steps=5)
    traj = sde.trajectory([[1] * dim], 50)
    assert_equal(traj.shape, (50, dim))
    assert_equal(traj[0], np.ones((dim,)))


@pytest.mark.parametrize('dim', [1, 2, 3, 4, 5])
def test_custom_ode(dim):
    def rhs(x):
        return [0.] * dim

    ode = dt.data.custom_ode(dim, rhs, h=1e-3, n_steps=5)
    traj = ode.trajectory([[1] * dim], 50)
    assert_equal(traj.shape, (50, dim))
    assert_equal(traj[0], np.ones((dim,)))


@pytest.mark.parametrize('dim', [-1, 0, 1.5, 999])
def test_custom_sde_wrong_dim(dim):
    with assert_raises(ValueError):
        dt.data.custom_sde(dim, lambda x: x, np.array([0]), 1., 5)


@pytest.mark.parametrize('dim', [-1, 0, 1.5, 999])
def test_custom_ode_wrong_dim(dim):
    with assert_raises(ValueError):
        dt.data.custom_ode(dim, lambda x: x, 1., 5)

