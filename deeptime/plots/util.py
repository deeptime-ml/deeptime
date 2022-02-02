from deeptime.util.decorators import plotting_function


@plotting_function
def default_colors():
    from matplotlib import rcParams
    return rcParams['axes.prop_cycle'].by_key()['color']
