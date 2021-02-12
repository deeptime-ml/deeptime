r"""Unit tests for the committor API-function

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import numpy as np
import pytest

from deeptime.data import birth_death_chain
from deeptime.markov.tools.analysis import committor
from tests.markov.tools.numeric import assert_allclose


@pytest.fixture
def bdc(sparse_mode):
    p = np.zeros(10)
    q = np.zeros(10)
    p[0:-1] = 0.5
    q[1:] = 0.5
    p[4] = 0.01
    q[6] = 0.1
    bdc = birth_death_chain(q, p, sparse=sparse_mode)
    yield bdc


def test_forward_committor(bdc):
    P = bdc.transition_matrix
    un = committor(P, [0, 1], [8, 9], forward=True)
    u = bdc.committor_forward(1, 8)
    assert_allclose(un, u)


def test_backward_comittor(bdc):
    P = bdc.transition_matrix
    un = committor(P, [0, 1], [8, 9], forward=False)
    u = bdc.committor_backward(1, 8)
    assert_allclose(un, u)
