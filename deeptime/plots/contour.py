import numpy as np

from deeptime.util.decorators import plotting_function


@plotting_function()
def plot_contour2d_from_xyz(x, y, z, nbins=100, method='nearest', contourf_kws=None, ax=None):
    if contourf_kws is None:
        contourf_kws = {}
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()

    # project onto grid
    from scipy.interpolate import griddata
    x, y = np.meshgrid(np.linspace(x.min(), x.max(), nbins), np.linspace(y.min(), y.max(), nbins), indexing='ij')
    z = griddata(np.hstack([x[:, None], y[:, None]]), z, (x, y), method=method)

    mappable = ax.contourf(x, y, z, **contourf_kws)
    return ax, mappable
