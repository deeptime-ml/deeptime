from typing import Union, List

import numpy as np

from deeptime.plots.util import default_colors
from deeptime.util import confidence_interval
from deeptime.util.decorators import plotting_function
from deeptime.util.validation import LaggedModelValidation


class CKTestGrid:

    @plotting_function()
    def __init__(self, n_cells_x, n_cells_y, height=2.5, aspect=1., sharey=True):
        import matplotlib.pyplot as plt

        self._n_cells_x = n_cells_x
        self._n_cells_y = n_cells_y
        figsize = n_cells_x * height * aspect, n_cells_y * height
        fig = plt.figure(figsize=figsize)
        axes = fig.subplots(self.n_cells_y, self._n_cells_x,
                            sharex="col", sharey=sharey,
                            squeeze=False)
        self._figure = fig
        self._axes = axes
        self._sharey = sharey

    @property
    def n_cells_x(self):
        return self._n_cells_x

    @property
    def n_cells_y(self):
        return self._n_cells_y

    @property
    def figure(self):
        return self._figure

    @property
    def axes(self):
        return self._axes

    def get_axis(self, i, j):
        return self.axes[i][j]

    def plot_panel(self, i, j, data: LaggedModelValidation, color, l_est=None, r_est=None, l_pred=None, r_pred=None,
                   **plot_kwargs):
        ax = self.get_axis(i, j)
        lest = ax.plot(data.lagtimes, data.estimates[:, i, j], color='black', **plot_kwargs)
        if l_est is not None and r_est is not None:
            ax.fill_between(data.lagtimes, l_est[:, i, j], r_est[1][:, i, j], color='black', alpha=0.2)
        lpred = ax.plot(data.lagtimes, data.predictions[:, i, j], color=color, linestyle='dashed', **plot_kwargs)
        if l_pred is not None and r_pred is not None:
            ax.fill_between(data.lagtimes, l_pred[:, i, j], r_pred[:, i, j], color=color, alpha=0.2)
        ax.text(0.05, 0.05, str(i + 1) + ' -> ' + str(j + 1), transform=ax.transAxes, weight='bold')
        if self._sharey:
            ax.set_ylim(0, 1)

        return lest, lpred

def _add_ck_subplot(cktest, test_index, ax, i, j, ipos=None, jpos=None, y01=True, units='steps', dt=1., **plot_kwargs):
    # plot estimates
    for default in ['color', 'linestyle']:
        if default in plot_kwargs.keys():
            # print("ignoring plot_kwarg %s: %s"%(default, plot_kwargs[default]))
            plot_kwargs.pop(default)
    color = 'C{}'.format(test_index)

    lest = ax.plot(dt * cktest.lagtimes, cktest.estimates[:, i, j], color='black', **plot_kwargs)
    # plot error of estimates if available
    if cktest.has_errors and cktest.err_est:
        ax.fill_between(dt * cktest.lagtimes, cktest.estimates_conf[0][:, i, j], cktest.estimates_conf[1][:, i, j],
                        color='black', alpha=0.2)
    # plot predictions
    lpred = ax.plot(dt * cktest.lagtimes, cktest.predictions[:, i, j], color=color, linestyle='dashed', **plot_kwargs)
    # plot error of predictions if available
    if cktest.has_errors:
        ax.fill_between(dt * cktest.lagtimes, cktest.predictions_conf[0][:, i, j], cktest.predictions_conf[1][:, i, j],
                        color=color, alpha=0.2)
    # add label
    ax.text(0.05, 0.05, str(i + 1) + ' -> ' + str(j + 1), transform=ax.transAxes, weight='bold')
    if y01:
        ax.set_ylim(0, 1)
    # Axes labels
    if ipos is None:
        ipos = i
    if jpos is None:
        jpos = j
    if (jpos == 0):
        ax.set_ylabel('probability')
    if (ipos == cktest.nsets - 1):
        ax.set_xlabel('lag time (' + units + ')')
    # return line objects
    return lest, lpred


def plot_ck_test(data: Union[LaggedModelValidation, List[LaggedModelValidation]], height=2.5, aspect=1.,
                 conf: float = 0.95, colors=None, **plot_kwargs):
    colors = default_colors() if colors is None else colors

    if not isinstance(data, list):
        data = [data]
    assert all(x.n_components == data[0].n_components for x in data), "All validations must have the same " \
                                                                      "number of components"

    confidences = []
    for lagged_model_validation in data:
        if lagged_model_validation.has_errors:
            l_pred, r_pred = confidence_interval(lagged_model_validation.predictions_samples, conf=conf)
            l_est, r_est = confidence_interval(lagged_model_validation.estimates_samples, conf=conf)
            confidences.append(((l_est, r_est), (l_pred, r_pred)))
        else:
            confidences.append(((None, None), (None, None)))

    n_components = data[0].n_components
    grid = CKTestGrid(n_components, n_components, height=height, aspect=aspect)
    for val_ix, val in enumerate(data):
        color = colors[val_ix]
        conf = confidences[val_ix]
        for i in range(grid.n_cells_x):
            for j in range(grid.n_cells_y):
                grid.plot_panel(i, j, val, color, conf[0][0], conf[0][1], conf[1][0], conf[1][1], **plot_kwargs)

    return grid
