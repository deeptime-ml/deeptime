import numpy as np
import pytest
from numpy.testing import assert_, assert_equal

from deeptime.util import energy2d
from deeptime.data import double_well_2d


@pytest.mark.parametrize("shift_energy", [False, True], ids=lambda x: f"shift_energy={x}")
def test_energy2d(shift_energy):
    traj = double_well_2d().trajectory([[0, 0]], length=100)
    energy = energy2d(*traj.T, bins=(20, 30), kbt=2.77, weights=np.linspace(0, 1, len(traj)), shift_energy=shift_energy)
    assert_equal(energy.x_meshgrid.shape, (20,))
    assert_equal(energy.y_meshgrid.shape, (30,))
    assert_equal(energy.energies.shape, (30, 20))
    if shift_energy:
        assert_(np.any(energy.energies == 0))
    else:
        assert_(np.all(energy.energies > 0))
