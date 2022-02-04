import numpy as np

from .platform import handle_progress_bar
from ..base import Observable, BayesianModel


def ck_test(models, observable: Observable, test_model=None, include_lag0=True, err_est=False, progress=None):
    r""" Performs a Chapman-Kolmogorov test: Under the assumption of Markovian dynamics some transfer operators
    such as the transition matrix of a Markov state model or a Koopman model possess the so-called Chapman-Kolmogorov
    property. It states that for lagtimes :math:`\tau_1, \tau_2` the transfer operator behaves like
    :math:`\mathcal{T}_{\tau_1 + \tau_2} = \mathcal{T}_{\tau_1}\mathcal{T}_{\tau_2}`.

    This method performs a test to verify the above identity by evaluating an observable :math:`\rho` *twice*
    on a range of different lagtimes. The observable evaluated on each model (estimated at a different lagtime)
    is compared against the observable evaluated on the test model when it is propagated by the Chapman-Kolmogorov
    equation to the corresponding lagtime:

    .. math::
        \mathcal{T}^\mathrm{test}_{k \tau}\rho \overset{!}{=} \mathcal{T}^{\mathrm{model}}_{\tilde{\tau}}

    such that :math:`k\tau = \tilde{\tau}`.

    Parameters
    ----------
    models : list of models
        List of models estimated at different lagtimes.
    observable : Observable
        An observable. The choice mainly depends on what kind of models are compared. Observables are required
        to be callable with a model instance.
    test_model : model, optional, default=None
        The test model. Per default this uses the model with the smallest lagtime of the models list.
    include_lag0 : bool, optional, default=True
        Whether to include :math:`\tau = 0` into the test.
    err_est : bool, optional, default=False
        Whether to compute the observable on Bayesian samples of the estimated models.
    progress : ProgressBar, optional, default=None
        An optional progress bar. Tested with tqdm.

    Returns
    -------
    ck_test : ChapmanKolmogorovTest
        The results. Can be used with `plot_ck_test <deeptime.plots.plot_ck_test>`.
    """
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


class ChapmanKolmogorovTest:
    r""" Test results of the Chapman-Kolmogorov test. See :meth:`ck_test`. """

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
        r""" Whether the prediction contains samples. """
        return self.predictions_samples is not None and len(self.predictions_samples) > 0

    @property
    def err_est(self):
        r""" Whether the estimated models contain samples """
        return self.estimates_samples is not None and len(self.estimates_samples) > 0
