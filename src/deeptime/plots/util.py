from deeptime.util.decorators import plotting_function


@plotting_function()
def default_colors():
    r""" Yields matplotlib default color cycle as per rc param 'axes.prop_cycle'. """
    from matplotlib import rcParams
    return rcParams['axes.prop_cycle'].by_key()['color']


@plotting_function()
def default_image_cmap():
    r""" Yields the default image color map. """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    return plt.get_cmap(rcParams["image.cmap"])


@plotting_function()
def default_line_width() -> float:
    from matplotlib import rcParams
    return rcParams['lines.linewidth']
