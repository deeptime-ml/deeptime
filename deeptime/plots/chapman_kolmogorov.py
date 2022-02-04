from typing import Union

import numpy as np

from deeptime.base import BayesianModel
from deeptime.markov import PCCAModel
from deeptime.markov.msm import MarkovStateModel
from deeptime.plots.util import default_colors
from deeptime.util import confidence_interval
from deeptime.util.decorators import plotting_function
from deeptime.util.platform import handle_progress_bar


class Observable:
    def __call__(self, model, mlag=1):
        raise NotImplementedError()


class MembershipsObservable(Observable):

    @staticmethod
    def _to_markov_model(model) -> MarkovStateModel:
        if hasattr(model, 'prior'):
            model = model.prior
        if hasattr(model, 'transition_model'):
            model = model.transition_model
        assert isinstance(model, MarkovStateModel), f"This should be a Markov state model but was {type(model)}."
        return model

    def __init__(self, test_model, memberships: Union[np.ndarray, PCCAModel],
                 initial_distribution: Union[str, np.ndarray] = 'stationary_distribution'):
        self.memberships = memberships if not isinstance(memberships, PCCAModel) else memberships.memberships
        self.n_states, self.n_sets = self.memberships.shape

        msm = MembershipsObservable._to_markov_model(test_model)
        n_states = msm.n_states
        symbols = msm.count_model.state_symbols
        symbols_full = msm.count_model.n_states_full
        if initial_distribution == 'stationary_distribution':
            init_dist = msm.stationary_distribution
        else:
            assert isinstance(initial_distribution, np.ndarray) and len(initial_distribution) == n_states, \
                "The initial distribution, if given explicitly, has to be defined on the Markov states of the model."
            init_dist = initial_distribution
        assert self.memberships.shape[0] == n_states, 'provided memberships and test_model n_states mismatch'
        self._test_model = test_model
        # define starting distribution
        P0 = self.memberships * init_dist[:, None]
        P0 /= P0.sum(axis=0)  # column-normalize
        self.P0 = P0

        # map from the full set (here defined by the largest state index in active set) to active
        self._full2active = np.zeros(np.max(symbols_full) + 1, dtype=int)
        self._full2active[symbols] = np.arange(len(symbols))

    def __call__(self, model, mlag=1):
        if mlag == 0 or model is None:
            return np.eye(self.n_sets)
        model = MembershipsObservable._to_markov_model(model)
        # otherwise compute or predict them by model.propagate
        pk_on_set = np.zeros((self.n_sets, self.n_sets))
        # compute observable on prior in case for Bayesian models.
        symbols = model.count_model.state_symbols
        subset = self._full2active[symbols]  # find subset we are now working on
        for i in range(self.n_sets):
            p0 = self.P0[:, i]  # starting distribution on reference active set
            p0sub = p0[subset]  # map distribution to new active set
            if subset is not None:
                p0sub /= np.sum(p0)  # renormalize
            pksub = model.propagate(p0sub, mlag)
            for j in range(self.n_sets):
                pk_on_set[i, j] = np.dot(pksub, self.memberships[subset, j])  # map onto set
        return pk_on_set


class KoopmanObservable(Observable):
    def __call__(self, model, mlag=1):
        pass


class ChapmanKolmogorovTest:

    def __init__(self, lagtimes, predictions, predictions_samples, estimates, estimates_samples, observable):
        self._lagtimes = np.array(lagtimes)
        self._predictions = np.asfarray(predictions)
        self._predictions_samples = predictions_samples
        self._estimates = np.asfarray(estimates)
        self._estimates_samples = estimates_samples
        self._observable = observable

    @property
    def n_components(self):
        return len(self.estimates[0]) if self.estimates is not None and len(self.estimates) > 0 else 0

    @property
    def lagtimes(self):
        r""" Lagtimes at which estimations and predictions were performed. """
        return self._lagtimes

    @property
    def estimates(self):
        """ Estimates at different lagtimes.

        :getter: each row contains the n observables computed at one of the T lagtimes.
        :type: ndarray(T, n, n)
        """
        return self._estimates

    @property
    def estimates_samples(self):
        r""" Returns the sampled estimates, i.e., a list of arrays as described in :attr:`estimates`. Can be `None`. """
        return self._estimates_samples

    @property
    def predictions(self):
        """ Returns tested model predictions at different lagtimes

        Returns
        -------
        Y : ndarray(T, n)
            each row contains the n observables predicted at one of the T lag
            times by the tested model.

        """
        return self._predictions

    @property
    def predictions_samples(self):
        """ Returns the confidence intervals of the estimates at different
        lagtimes (if available)

        If not available, returns None.

        Returns
        -------
        L : ndarray(T, n)
            each row contains the lower confidence bound of n observables
            computed at one of the T lag times.

        R : ndarray(T, n)
            each row contains the upper confidence bound of n observables
            computed at one of the T lag times.

        """
        return self._predictions_samples

    @property
    def has_errors(self):
        return self.predictions_samples is not None and len(self.predictions_samples) > 0

    @property
    def err_est(self):
        return self.estimates_samples is not None

    @staticmethod
    def from_models(models, observable: Observable, test_model=None, include_lag0=True, progress=None,
                    err_est=False):
        assert all(hasattr(m, 'lagtime') for m in models) and (test_model is None or hasattr(test_model, 'lagtime')), \
            "All models and the test model need to have a lagtime property."
        progress = handle_progress_bar(progress)
        models = sorted(models, key=lambda x: x.lagtime)
        if test_model is None:
            test_model = models[0]

        is_bayesian = isinstance(test_model, BayesianModel)
        assert all(isinstance(m, BayesianModel) == is_bayesian for m in models), \
            "All models must either have samples or none at all."
        lagtimes = [0] if include_lag0 else []
        lagtimes += [m.lagtime for m in models]

        predictions = []
        predictions_samples = []
        reference_lagtime = test_model.lagtime
        for lagtime in progress(lagtimes):
            predictions.append(observable(test_model, mlag=lagtime / reference_lagtime))
            if is_bayesian:
                obs = []
                for sample in test_model.samples:
                    obs.append(observable(sample, mlag=lagtime / reference_lagtime))
                predictions_samples.append(obs)

        if include_lag0:
            models = [None] + models
        estimates = []
        estimates_samples = []
        for model in progress(models):
            estimates.append(observable(model, mlag=1))
            if is_bayesian and err_est:
                if model is not None:
                    obs = []
                    for sample in model.samples:
                        obs.append(observable(sample, mlag=1))
                    else:
                        obs.append(observable(model, mlag=0))
                    estimates_samples.append(obs)
                else:
                    estimates_samples.append([observable(model, mlag=0)])

        return ChapmanKolmogorovTest(lagtimes, predictions, predictions_samples, estimates, estimates_samples,
                                     observable)


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
        if l_est is not None and len(lest) > 0 and r_est is not None and len(r_est) > 0:
            ax.fill_between(data.lagtimes, l_est[:, i, j], r_est[:, i, j], color='black', alpha=0.2)
        lpred = ax.plot(data.lagtimes, data.predictions[:, i, j], color=color, linestyle='dashed', **plot_kwargs)
        if l_pred is not None and r_pred is not None:
            ax.fill_between(data.lagtimes, l_pred[:, i, j], r_pred[:, i, j], color=color, alpha=0.2)
        ax.text(0.05, 0.05, str(i + 1) + ' -> ' + str(j + 1), transform=ax.transAxes, weight='bold')
        if self._sharey:
            ax.set_ylim(0, 1)

        return lest, lpred

    def set_axes_labels(self, i, j, xlabel, ylabel):
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
                 conf: float = 0.95, color=None, grid: CKTestGrid = None, legend=True, **plot_kwargs):
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
    else:
        confidences_pred_l.append(None)
        confidences_pred_r.append(None)
        confidences_est_l.append(None)
        confidences_est_r.append(None)

    confidences = [confidences_est_l, confidences_est_r, confidences_pred_l, confidences_pred_r]
    confidences = [np.array(conf) for conf in confidences]
    grid.plot_cktest(data, color, confidences, **plot_kwargs)

    if legend:
        grid.legend(conf)

    return grid
