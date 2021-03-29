import numpy as np
from deeptime.data import prinz_potential


def test_sanity():
    prinz_potential(n_steps=500).trajectory([[0.]], 50000)
    prinz_potential()(np.random.uniform(-1, 1, size=(500, 1)))
