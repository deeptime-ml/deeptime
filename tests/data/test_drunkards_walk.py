import numpy as np
from numpy.testing import assert_, assert_almost_equal, assert_equal, assert_allclose

import deeptime


def test_barrier():
    sim = deeptime.data.drunkards_walk(bar_location=(0, 0), home_location=(9, 9))

    sim.add_barrier((0, 9), (5, 8))
    sim.add_barrier((5, 0), (5, 4))
    transition_matrix = sim.msm.transition_matrix

    for coord_next_to_barrier in [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4)]:
        state = sim.coordinate_to_state(coord_next_to_barrier)
        x0, y0 = coord_next_to_barrier
        barrier_coord = (x0 - 1, y0)
        assert_(barrier_coord in sim.barriers)
    assert_(np.all(transition_matrix >= 0))
    for state in range(sim.n_states):
        assert_almost_equal(transition_matrix[state, :].sum(), 1.)


def test_state_coordinates_conversion():
    walk = deeptime.data.drunkards_walk((7, 13), bar_location=(1, 1), home_location=(7, 8))
    assert_equal(walk.coordinate_to_state((1, 1)), 1 + 7)
    assert_equal(walk.state_to_coordinate(8), (1, 1))

    for i in range(7):
        for j in range(13):
            assert_equal(walk.state_to_coordinate(walk.coordinate_to_state((i, j))), (i, j))


def test_transition_matrix():
    walk = deeptime.data.drunkards_walk((7, 8), bar_location=(0, 0), home_location=(6, 7))
    assert_equal(walk.msm.transition_matrix.shape, (7 * 8, 7 * 8))
    assert_allclose(walk.msm.transition_matrix.sum(-1), 1.)  # row-stochastic matrix
    assert_equal(walk.bar_location.squeeze(), (0, 0))
    assert_equal(walk.home_location.squeeze(), (6, 7))
    assert_equal(walk.bar_state, [0])
    assert_equal(walk.home_state, [6 + 7*7])
    assert_equal(walk.grid_size, (7, 8))
    assert_equal(walk.n_states, 7*8)
    trajectory = walk.walk((3, 3), 100000, stop=True)
    last_coord = tuple(trajectory[-1])
    assert_(last_coord == (0, 0) or last_coord == (6, 7))
