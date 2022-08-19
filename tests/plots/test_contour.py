import matplotlib

matplotlib.use('Agg')

import numpy as np
import pytest

from deeptime.plots import plot_contour2d_from_xyz, plot_density


@pytest.fixture
def binomial_data():
    return np.random.binomial(10, .4, size=(100, 2))


def test_plot_contour2d(binomial_data):
    plot_contour2d_from_xyz(*binomial_data.T, np.zeros((binomial_data.shape[0],)))


@pytest.mark.parametrize("avoid_zero_counts", [False, True])
def test_plot_density(binomial_data, avoid_zero_counts):
    plot_density(*binomial_data.T, avoid_zero_counts=avoid_zero_counts)
