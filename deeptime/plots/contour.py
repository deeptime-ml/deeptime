import numpy as np

from deeptime.util.decorators import plotting_function


@plotting_function()
def plot_contour2d_from_xyz(x, y, z, n_bins=100, method='nearest', contourf_kws=None, ax=None):
    r"""Plot a two-dimensional contour map based on a histogram over unordered data triplets $(x, y)\mapsto z$.

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
    if contourf_kws is None:
        contourf_kws = {}
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
