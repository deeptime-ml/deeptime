import numpy as np
import pytest
from numpy.testing import assert_equal, assert_raises, assert_allclose, assert_, assert_array_almost_equal
from scipy.integrate import solve_ivp

import deeptime as dt
from tests.testing_utilities import nullcontext


@pytest.mark.parametrize("init", ['list', 'array'])
@pytest.mark.parametrize("system,dim,integrator,has_potential", [
    [dt.data.quadruple_well, 2, 'EulerMaruyama', True],
    [dt.data.quadruple_well_asymmetric, 2, 'EulerMaruyama', True],
    [dt.data.triple_well_2d, 2, 'EulerMaruyama', True],
    [dt.data.triple_well_1d, 1, 'EulerMaruyama', True],
    [dt.data.double_well_2d, 2, 'EulerMaruyama', True],
    [dt.data.ornstein_uhlenbeck, 1, 'EulerMaruyama', True],
    [dt.data.prinz_potential, 1, 'EulerMaruyama', True],
    [dt.data.time_dependent_quintuple_well, 2, 'EulerMaruyama', True],
    [dt.data.abc_flow, 3, 'RungeKutta', False],
    [dt.data.BickleyJet, 2, 'RungeKutta', False]
])
def test_interface(init, system, dim, integrator, has_potential):
    instance = system(h=1e-5, n_steps=10)
    assert_equal(instance.h, 1e-5)
    assert_equal(instance.n_steps, 10)
    assert_equal(instance.dimension, dim)
    assert_equal(instance.integrator, integrator)
    assert_equal(instance.has_potential_function, has_potential)
    assert_equal(instance.time_dependent, isinstance(instance, dt.data.TimeDependentSystem))

    if init == 'list':
        x0 = list(range(dim))
    else:
        x0 = np.random.normal(size=(dim,))
    if instance.time_dependent:
        assert_equal(instance.trajectory(0., x0, 50).shape, (50, dim))
        assert_equal(instance(0., x0).shape, (1, dim))
    else:
        assert_equal(instance.trajectory(x0, 50).shape, (50, dim))
        assert_equal(instance(x0).shape, (1, dim))

    if init == 'list':
        x0 = [list(range(dim)) for _ in range(5)]
    else:
        x0 = np.random.normal(size=(5, dim))
    if instance.time_dependent:
        assert_equal(instance.trajectory(1., x0, 50).shape, (5, 50, dim))
        assert_equal(instance(1., x0).shape, (5, dim))
    else:
        assert_equal(instance.trajectory(x0, 50).shape, (5, 50, dim))
        assert_equal(instance(x0).shape, (5, dim))

    if has_potential:
        if instance.time_dependent:
            assert_equal(instance.potential(55., x0).shape, (len(x0),))
        else:
            assert_equal(instance.potential(x0).shape, (len(x0),))


@pytest.mark.parametrize("vectorized", [True, False])
@pytest.mark.parametrize("system", [dt.data.abc_flow, dt.data.BickleyJet])
@pytest.mark.parametrize("ref_method", ['RK45', 'DOP853', 'LSODA'])
def test_ode_against_scipy(system, vectorized, ref_method):
    instance = system(h=1e-5, n_steps=1000)
    dim = instance.dimension
    y0 = np.array([[.5] * dim]*10)
    assert_(instance.vectorized_f)
    if instance.time_dependent:
        traj = instance.trajectory(0., y0, 2)
    else:
        traj = instance.trajectory(y0, 2)
    soln = solve_ivp(instance.f, [0, 1e-2], y0=y0[0], vectorized=vectorized, method=ref_method, t_eval=[0, 1e-2]).y
    assert_equal(soln.shape, (dim, 2))
    assert_equal(traj.shape, (10, 2, dim))

    for i in range(10):
        assert_allclose(soln[:, 0], y0[0])
        assert_allclose(traj[i, 0], y0[0])
        assert_array_almost_equal(soln[:, 1], traj[i, 1], decimal=5)


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

    sde = dt.data.custom_sde(dim, rhs, sigma, h=1e-6, n_steps=5)
    assert_equal(sde.dimension, dim)
    assert_equal(sde.has_potential_function, False)
    assert_equal(sde.integrator, "EulerMaruyama")
    traj = sde.trajectory([[1] * dim], 50)
    assert_equal(traj.shape, (50, dim))
    assert_equal(traj[0], np.ones((dim,)))
    test_points = sde(np.ones((1000, dim)))
    assert_equal(test_points.shape, (1000, dim))
    assert_allclose(np.mean(test_points), 1., rtol=1e-1, atol=1e-1)
    assert_(not sde.vectorized_f)


@pytest.mark.parametrize('dim', [1, 2, 3, 4, 5])
def test_custom_ode(dim):
    def rhs(x):
        return [0.] * dim

    ode = dt.data.custom_ode(dim, rhs, h=1e-3, n_steps=5)
    assert_equal(ode.dimension, dim)
    assert_equal(ode.has_potential_function, False)
    assert_equal(ode.integrator, "RungeKutta")
    traj = ode.trajectory([[1] * dim], 50)
    assert_equal(traj.shape, (50, dim))
    assert_equal(traj[0], np.ones((dim,)))
    test_points = ode(np.ones((100, dim)))
    assert_equal(test_points.shape, (100, dim))
    assert_allclose(test_points, 1.)
    assert_(not ode.vectorized_f)


@pytest.mark.parametrize('dim', [-1, 0, 1.5, 999])
def test_custom_sde_wrong_dim(dim):
    with assert_raises(ValueError):
        dt.data.custom_sde(dim, lambda x: x, np.array([0]), 1., 5)


@pytest.mark.parametrize('dim', [-1, 0, 1.5, 999])
def test_custom_ode_wrong_dim(dim):
    with assert_raises(ValueError):
        dt.data.custom_ode(dim, lambda x: x, 1., 5)


@pytest.mark.parametrize('vectorized_ivp', [True, False], ids=lambda x: f"vectorized={x}")
@pytest.mark.parametrize('full_periodic', [True, False], ids=lambda x: f"full_periodic={x}")
def test_bickley_integrate(vectorized_ivp, full_periodic):
    X = np.vstack((np.full((5,), 0.1), np.linspace(-2.9, 2.9, num=5))).T

    simulator = dt.data.BickleyJet(1e-4, 10000, full_periodic=full_periodic)
    assert_(simulator.periodic_bc)
    simulator_back = dt.data.BickleyJet(-1e-4, 10000, full_periodic=full_periodic)

    with assert_raises(RuntimeError) if full_periodic else nullcontext():
        simulator.periodic_bc = False
        assert_(not simulator.periodic_bc)
    with assert_raises(RuntimeError) if full_periodic else nullcontext():
        simulator_back.periodic_bc = False
        assert_(not simulator_back.periodic_bc)

    time, traj = simulator.trajectory(0, X, 11, return_time=True)

    Xfinal = traj[:, -1]
    time_back, traj_back = simulator_back.trajectory(10, traj[:, -1], 11, return_time=True)
    assert_array_almost_equal(time, np.flip(time_back, axis=1))
    assert_array_almost_equal(traj, np.flip(traj_back, axis=1))
    t_eval = np.linspace(0, 10, num=11, endpoint=True)

    periodic_traj_back = traj_back if full_periodic else dt.data.BickleyJet.apply_periodic_boundary_conditions(traj_back)

    if not full_periodic:
        periodic_traj = dt.data.BickleyJet.apply_periodic_boundary_conditions(traj, inplace=False)
    else:
        periodic_traj = traj
    assert_(np.all((periodic_traj[..., 0] <= 20) & (periodic_traj[..., 0] >= 0)))  # inside domain

    for i in range(5):
        assert_equal(t_eval, time[i])
        soln = solve_ivp(simulator.f, [0, 10], y0=X[i], vectorized=vectorized_ivp, method='RK45',
                         atol=1e-12, rtol=1e-12, t_eval=t_eval)
        solnpbc = dt.data.BickleyJet.apply_periodic_boundary_conditions(soln.y.T, inplace=False)
        assert_array_almost_equal(solnpbc, periodic_traj[i])
        soln_bwd = solve_ivp(simulator.f, [10, 0], y0=Xfinal[i], vectorized=vectorized_ivp, method='RK45',
                             atol=1e-12, rtol=1e-12, t_eval=t_eval[::-1])
        soln_bwdpbc = dt.data.BickleyJet.apply_periodic_boundary_conditions(soln_bwd.y.T, inplace=False)
        assert_array_almost_equal(soln_bwdpbc, periodic_traj_back[i], decimal=5)
        assert_array_almost_equal(periodic_traj_back[i, -1], X[i])
        assert_array_almost_equal(soln_bwdpbc[-1], X[i], decimal=5)


def test_bickley():
    U_0 = 5.4138
    L_0 = 1.77
    r_0 = 6.371
    c = np.array((0.1446, 0.205, 0.461)) * U_0
    eps = np.array((0.075, 0.15, 0.3))
    k = np.array((2, 4, 6)) * 1. / r_0

    system = dt.data.BickleyJet(1e-5, 10)
    assert_equal(system.h, 1e-5)
    assert_equal(system.n_steps, 10)
    assert_equal(system.integrator, 'RungeKutta')
    assert_equal(system.has_potential_function, False)
    assert_equal(system.dimension, 2)
    assert_equal(system.time_dependent, True)
    assert_equal(system.c, c)
    assert_equal(system.U0, U_0)
    assert_equal(system.L0, L_0)
    assert_equal(system.eps, eps)
    assert_equal(system.k, k)
    assert_equal(system.r0, r_0)

    dataset = dt.data.bickley_jet(10, n_jobs=1)
    assert_equal(dataset.data.shape, (401, 10, 2))
    assert_equal(len(dataset), 401)
    assert_equal(dataset[0].shape, (10, 2))

    dataset_endpoints = dataset.endpoints_dataset()
    assert_equal(dataset_endpoints.data.shape, (10, 2))
    assert_equal(dataset_endpoints.data_lagged.shape, (10, 2))

    dataset_endpoints_3d = dataset_endpoints.to_3d(radius=1.)
    assert_equal(dataset_endpoints_3d.data.shape, (10, 3))
    assert_equal(dataset_endpoints_3d.data_lagged.shape, (10, 3))

    dataset_clusters = dataset_endpoints_3d.cluster(13)
    assert_equal(dataset_clusters.data.shape, (10, 13 ** 3))
    assert_equal(dataset_clusters.data_lagged.shape, (10, 13 ** 3))
