import numpy as np
from numpy.testing import assert_equal, assert_raises

import deeptime
from deeptime.data import PBFSimulator


def test_invalid_args():
    with assert_raises(ValueError):
        PBFSimulator(domain_size=np.array(-1), initial_positions=np.zeros((10, 2)), interaction_distance=.1)
    with assert_raises(ValueError):
        PBFSimulator(domain_size=np.array([1, 1]), initial_positions=np.zeros((10, 1)), interaction_distance=.1)
    with assert_raises(ValueError):
        PBFSimulator(domain_size=np.array([1, 1]), initial_positions=np.zeros((10, 2)), interaction_distance=-1.)


def test_pbf_sanity():
    sim = deeptime.data.position_based_fluids(n_burn_in=5)
    sim.run(1, 0.1)
    sim.run(1, -0.1)
    traj = sim.simulate_oscillatory_force(5, 10, .05)
    sim.transform_to_density(traj, 5, 3, n_jobs=1)


def test_pbf_simulator_properties():
    simulator = PBFSimulator(domain_size=np.array([5, 5]), initial_positions=np.random.uniform(-1, 1, size=(200, 2)),
                             interaction_distance=.5, n_jobs=1, n_solver_iterations=100, gravity=10000, epsilon=1,
                             timestep=1e-5, rest_density=.2, tensile_instability_distance=.1, tensile_instability_k=.2)
    assert_equal(simulator.domain_size, [5, 5])
    assert_equal(simulator.n_particles, 200)
