import numpy as np

from deeptime.plots.util import default_colors
from deeptime.util import confidence_interval
from deeptime.util.decorators import plotting_function
from deeptime.util.validation import ChapmanKolmogorovTest


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
        self._lest_handles = []
        self._lpred_handles = []
        self._tests = []

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

    def plot_cktest(self, data, color, confidences, **plot_kwargs):
        self._tests.append(data)
        lest, lpred = None, None  # line objects
        for i in range(self.n_cells_x):
            for j in range(self.n_cells_y):
                lest, lpred = self._plot_panel(i, j, data, color, *confidences, **plot_kwargs)
        self._lest_handles.append(lest[0])
        self._lpred_handles.append(lpred[0])

    def _plot_panel(self, i, j, data: ChapmanKolmogorovTest, color, l_est=None, r_est=None, l_pred=None, r_pred=None,
                    **plot_kwargs):
        ax = self.get_axis(i, j)
        lest = ax.plot(data.lagtimes, data.estimates[:, i, j], color='black', **plot_kwargs)
        if l_est is not None and len(l_est) > 0 and r_est is not None and len(r_est) > 0:
            ax.fill_between(data.lagtimes, l_est[:, i, j], r_est[:, i, j], color='black', alpha=0.2)
        lpred = ax.plot(data.lagtimes, data.predictions[:, i, j], color=color, linestyle='dashed', **plot_kwargs)
        if l_pred is not None and len(l_pred) > 0 and r_pred is not None and len(r_pred) > 0:
            ax.fill_between(data.lagtimes, l_pred[:, i, j], r_pred[:, i, j], color=color, alpha=0.2)
        ax.text(0.05, 0.05, str(i + 1) + ' -> ' + str(j + 1), transform=ax.transAxes, weight='bold')
        if self._sharey:
            ax.set_ylim(0, 1)

        return lest, lpred

    def set_axes_labels(self, i, j, xlabel=None, ylabel=None):
        if xlabel is not None:
            self.get_axis(i, j).set_xlabel(xlabel)
        if ylabel is not None:
            self.get_axis(i, j).set_ylabel(ylabel)

    def legend(self, conf=None):
        handles = []
        labels = []
        for ix, test in enumerate(self._tests):
            predlabel = 'predict {}'.format(ix) if len(self._tests) > 1 else 'predict'
            estlabel = 'estimate {}'.format(ix) if len(self._tests) > 1 else 'estimate'
            if test.has_errors and conf is not None:
                predlabel += '     conf. {:3.1f}%'.format(100.0*conf)
            handles.append(self._lest_handles[ix])
            handles.append(self._lpred_handles[ix])
            labels.append(predlabel)
            labels.append(estlabel)
        self.figure.legend(handles, labels, 'upper center', ncol=2, frameon=False)


def plot_ck_test(data: ChapmanKolmogorovTest, height=2.5, aspect=1.,
                 conf: float = 0.95, color=None, grid: CKTestGrid = None, legend=True,
                 xlabel='lagtime (steps)', ylabel='probability', **plot_kwargs):
    color = default_colors()[0] if color is None else color
    n_components = data.n_components

    if grid is not None:
        assert grid.n_cells_x == grid.n_cells_y == n_components
    else:
        grid = CKTestGrid(n_components, n_components, height=height, aspect=aspect)

    confidences_est_l = []
    confidences_est_r = []
    confidences_pred_l = []
    confidences_pred_r = []
    if data.has_errors:
        samples = data.predictions_samples
        for lag_samples in samples:
            l_pred, r_pred = confidence_interval(lag_samples, conf=conf, remove_nans=True)
            confidences_pred_l.append(l_pred)
            confidences_pred_r.append(r_pred)

        samples = data.estimates_samples
        for lag_samples in samples:
            l_est, r_est = confidence_interval(lag_samples, conf=conf, remove_nans=True)
            confidences_est_l.append(l_est)
            confidences_est_r.append(r_est)

    confidences = [confidences_est_l, confidences_est_r, confidences_pred_l, confidences_pred_r]
    confidences = [np.array(conf) for conf in confidences]
    grid.plot_cktest(data, color, confidences, **plot_kwargs)

    for i in range(grid.n_cells_x):
        grid.set_axes_labels(i, 0, ylabel=ylabel)
    for j in range(grid.n_cells_y):
        grid.set_axes_labels(grid.n_cells_y - 1, j, xlabel=xlabel)

    if legend:
        grid.legend(conf)

    return grid
