import matplotlib
import pytest
from numpy.testing import assert_

from deeptime.data import ellipsoids
from deeptime.plots import plot_energy2d
from deeptime.util import energy2d

matplotlib.use('Agg')


@pytest.mark.parametrize('shift_energy', [True, False], ids=lambda x: f"shift_energy={x}")
@pytest.mark.parametrize('cbar', [True, False], ids=lambda x: f"cbar={x}")
def test_energy2d(shift_energy, cbar):
    traj = ellipsoids().observations(20000)
    data = energy2d(*traj.T, bins=100, shift_energy=shift_energy)
    ax, contourf, cbar = plot_energy2d(data, cbar=cbar)
    assert_(ax is not None)
    assert_(contourf is not None)
    assert_(cbar is not None if cbar else cbar is None)
