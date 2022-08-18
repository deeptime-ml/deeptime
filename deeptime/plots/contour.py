import numpy as np

from deeptime.util.decorators import plotting_function


@plotting_function()
def plot_contour2d_from_xyz(x, y, z, n_bins=100, method='nearest', contourf_kws=None, ax=None):
    r"""Plot a two-dimensional contour map based on an interpolation over unordered data triplets $(x, y)\mapsto z$.

    .. plot:: examples/plot_contour2d_from_xyz.py

    Parameters
    ----------
    x : ndarray
        Sample $x$ coordinates as array of shape (T,).
    y : ndarray
        Sample $y$ coordinates as array of shape (T,).
    z : ndarray
        Sample $z$ values as array of shape (T,).
    n_bins : int, optional, default=100
        Resolution of the two-dimensional histogram.
    method : str, default='nearest'
        Interpolation method. See :meth:`scipy.interpolate.griddata`.
    contourf_kws : dict, optional, default=None
        dict of optional keyword arguments for matplotlib.contourf. Per default empty dict.
    ax : matplotlib.Axes, optional, default=None
        axes to plot onto. In case of no provided axes, grabs the `matplotlib.gca()`.

    Returns
    -------
    ax : matplotlib.Axes
        Axes onto which the contour was plotted.
    mappable
        Matplotlib mappable that can be used to create, e.g., colorbars.
    """
    contourf_kws = {} if contourf_kws is None else contourf_kws
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()

    # project onto grid
    from scipy.interpolate import griddata
    xgrid, ygrid = np.meshgrid(np.linspace(np.min(x), np.max(x), n_bins),
                               np.linspace(np.min(y), np.max(y), n_bins), indexing='ij')
    zgrid = griddata(np.hstack([x[:, None], y[:, None]]), z, (xgrid, ygrid), method=method)

    mappable = ax.contourf(xgrid, ygrid, zgrid, **contourf_kws)
    return ax, mappable


@plotting_function()
def plot_density(x, y, n_bins=100, weights=None, avoid_zero_count=False, contourf_kws=None, ax=None):
    r"""Plot a two-dimensional contour map based on a histogram over unordered data $(x, y)$.

    .. plot:: examples/plot_density.py

    Parameters
    ----------
    x : ndarray
        Sample $x$ coordinates as array of shape (T,).
    y : ndarray
        Sample $y$ coordinates as array of shape (T,).
    n_bins : int, optional, default=100
        Resolution of the two-dimensional histogram.
    weights : ndarray(T), optional, default=None
        Sample weights. By default, all samples have the same weight.
    avoid_zero_count : bool, optional, default=False
        Whether to clamp the histogram to its lowest value whenever it is zero. Useful for log-scale plotting.
    contourf_kws : dict, optional, default=None
        dict of optional keyword arguments for matplotlib.contourf. Per default empty dict.
    ax : matplotlib.Axes, optional, default=None
        axes to plot onto. In case of no provided axes, grabs the `matplotlib.gca()`.

    Returns
    -------
    ax : matplotlib.Axes
        Axes onto which the contour was plotted.
    mappable
        Matplotlib mappable that can be used to create, e.g., colorbars.
    """
    from deeptime.util.stats import histogram2d_from_xy

    # initialize defaults
    contourf_kws = {} if contourf_kws is None else contourf_kws
    if ax is None:
        from matplotlib.pyplot import gca
        ax = gca()

    # obtain histogram, normalize, potentially clamp
    x_meshgrid, y_meshgrid, hist = histogram2d_from_xy(x, y, bins=n_bins, weights=weights, density=True)
    if avoid_zero_count:
        hist = np.maximum(hist, np.min(hist[hist.nonzero()]))

    # plot
    mappable = ax.contourf(x_meshgrid, y_meshgrid, hist.T, **contourf_kws)
    return ax, mappable
