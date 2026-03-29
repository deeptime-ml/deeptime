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


def test_plot_density_axes_not_swapped():
    """Regression test for issue #302: plot_density was swapping x and y axes.

    The histogram returned by histogram2d_from_xy is already transposed to
    (n_bins_y, n_bins_x) shape, which is what contourf expects. A previous
    bug applied an extra .T, causing x and y to be swapped.
    """
    from deeptime.util.stats import histogram2d_from_xy

    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, size=500)
    y = rng.uniform(0, 10, size=500)

    # Use asymmetric bin counts so shape mismatch would cause an error
    n_bins_x, n_bins_y = 15, 25
    x_mesh, y_mesh, hist = histogram2d_from_xy(x, y, bins=[n_bins_x, n_bins_y], density=True)
    # hist.shape should be (n_bins_y, n_bins_x) — what contourf(x_mesh, y_mesh, Z) expects
    assert hist.shape == (n_bins_y, n_bins_x)
    assert len(x_mesh) == n_bins_x
    assert len(y_mesh) == n_bins_y

    # plot_density should not transpose hist again; with asymmetric bins,
    # an extra .T would cause contourf to raise a shape mismatch error
    ax, mappable = plot_density(x, y, n_bins=[n_bins_x, n_bins_y])
