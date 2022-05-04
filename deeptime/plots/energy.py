from deeptime.util import EnergyLandscape2d
from deeptime.util.decorators import plotting_function


class Energy2dPlot:
    r""" The result of a :meth:`plot_energy2d` call. Instances of this class can be unpacked like a tuple:

    >>> import numpy as np
    >>> from deeptime.util import energy2d
    >>> ax, contour, cbar = plot_energy2d(energy2d(*np.random.uniform(size=(100 ,2)).T))

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes that was plotted on.
    contour : matplotlib.contour.QuadContourSet
        The surface contour. See matplotlib's
        `QuadContourSet <https://matplotlib.org/stable/api/contour_api.html#matplotlib.contour.QuadContourSet>`_.
    colorbar : matplotlib.colorbar.Colorbar, optional, default=None
        If a colorbar was drawn, it is referenced here.

    See Also
    --------
    plot_energy2d
    """

    def __init__(self, ax, contour, colorbar=None):
        self.ax = ax
        self.contour = contour
        self.colorbar = colorbar

    # makes object unpackable
    def __len__(self):
        return 3

    # makes object unpackable
    def __iter__(self):
        return iter((self.ax, self.contour, self.colorbar))


@plotting_function()
def plot_energy2d(energies: EnergyLandscape2d, ax=None, levels=100, contourf_kws=None, cbar=True,
                  cbar_kws=None, cbar_ax=None):
    r""" Plot a two-dimensional energy surface. See :meth:`deeptime.util.energy2d` to obtain the energy estimation
    from data.

    .. plot:: examples/plot_energy_surface.py

    Parameters
    ----------
    energies : EnergyLandscape2d
        The estimated energies. Can be obtained via :meth:`deeptime.util.energy2d`.
    ax : matplotlib.axes.Axes, optional, default=None
        The axes to plot on. Otherwise, uses the current axes (via `plt.gca()`).
    levels : int, default=100
        Number of contour levels.
    contourf_kws : dict, optional, default=None
        Keyword arguments for
        `Axes.contourf <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.contourf.html>`_.
    cbar : bool, optional, default=True
        Whether to draw a color bar.
    cbar_kws : dict, optional, default=None
        Keyword arguments for
        `Figure.colorbar <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.colorbar>`_.
    cbar_ax : matplotlib.axes.Axes, optional, default=None
        The axes to plot the colorbar on. If left to `None`, take space from the current (or provided) axes.

    Returns
    -------
    plot : Energy2dPlot
        Plotting object encapsulating references to used axes, colorbars, and contours.

    See Also
    --------
    deeptime.util.energy2d
    """
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()

    if contourf_kws is None:
        contourf_kws = {}
    mappable = ax.contourf(energies.x_meshgrid, energies.y_meshgrid, energies.energies, levels=levels, **contourf_kws)

    if cbar:
        if cbar_kws is None:
            cbar_kws = {}
        cb = ax.figure.colorbar(mappable, cax=cbar_ax, ax=ax, **cbar_kws)
        cb.outline.set_linewidth(0)
    else:
        cb = None

    return Energy2dPlot(ax, mappable, cb)
