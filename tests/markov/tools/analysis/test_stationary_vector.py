import pytest
import numpy as np
from numpy.testing import assert_allclose

from deeptime.data import birth_death_chain
from deeptime.markov.tools.analysis import stationary_distribution


@pytest.fixture
def stationary_vector_data(sparse_mode):
    dim = 100

    """Set up meta-stable birth-death chain"""
    p = np.zeros(dim)
    p[0:-1] = 0.5

    q = np.zeros(dim)
    q[1:] = 0.5

    p[dim // 2 - 1] = 0.001
    q[dim // 2 + 1] = 0.001

    return birth_death_chain(q, p, sparse=sparse_mode)


@pytest.mark.parametrize('mode', stationary_distribution.valid_modes)
def test_statdist(mode, stationary_vector_data):
    P = stationary_vector_data.transition_matrix
    mu = stationary_vector_data.stationary_distribution
    mun = stationary_distribution(P, mode=mode)
    assert_allclose(mu, mun)
