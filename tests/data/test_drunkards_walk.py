from numpy.testing import *
import sktime


def test_state_coordinates_conversion():
    walk = sktime.data.drunkards_walk((7, 13), bar_location=(1, 1), home_location=(7, 8))
    assert_equal(walk.coordinate_to_state((1, 1)), 1 + 7)
    assert_equal(walk.state_to_coordinate(8), (1, 1))

    for i in range(7):
        for j in range(13):
            assert_equal(walk.state_to_coordinate(walk.coordinate_to_state((i, j))), (i, j))


def test_transition_matrix():
    walk = sktime.data.drunkards_walk((7, 8), bar_location=(0, 0), home_location=(6, 7))
    assert_equal(walk.msm.transition_matrix.shape, (7 * 8, 7 * 8))
    assert_allclose(walk.msm.transition_matrix.sum(-1), 1.)  # row-stochastic matrix
    assert_equal(walk.bar_location, (0, 0))
    assert_equal(walk.home_location, (6, 7))
    assert_equal(walk.bar_state, 0)
    assert_equal(walk.home_state, 6 + 7*7)
    assert_equal(walk.grid_size, (7, 8))
    assert_equal(walk.n_states, 7*8)
