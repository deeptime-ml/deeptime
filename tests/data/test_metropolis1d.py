import numpy as np
import deeptime


def test_transition_matrix():
    transition_matrix = deeptime.data.tmatrix_metropolis1d(energies=[.1, .1, .1, .1], d=1.)
    np.testing.assert_equal(transition_matrix.shape, (4, 4))
    np.testing.assert_equal(transition_matrix, [[0.5, 0.5, 0., 0.],
                                                [0.5, 0., 0.5, 0.],
                                                [0., 0.5, 0., 0.5],
                                                [0., 0., 0.5, 0.5]])
