from deeptime.util.decorators import plotting_function


@plotting_function()
def default_colors():
    r""" Yields matplotlib default color cycle as per rc param 'axes.prop_cycle'. """
    from matplotlib import rcParams
    return rcParams['axes.prop_cycle'].by_key()['color']
