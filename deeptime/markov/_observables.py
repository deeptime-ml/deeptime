import numbers
from typing import Union

import numpy as np

from ..base import Observable


class MembershipsObservable(Observable):
    r""" Memberships observable which evaluates some observable through a fuzzy memberships clustering.

    Parameters
    ----------
    test_model : markov model
        A Markov model to obtain number of Markov states, symbols, and stationary distribution.
    memberships : ndarray or PCCAModel
        Fuzzy membership assignment.
    initial_distribution : 'stationary_distribution' or ndarray
        The initial distribution which is propagated.
    ignore_imaginary_parts : bool, optional, default=False
        Whether to project observable values onto reals.
    """

    def __init__(self, test_model, memberships,
                 initial_distribution: Union[str, np.ndarray] = 'stationary_distribution',
                 ignore_imaginary_parts: bool = False):
        from deeptime.markov import PCCAModel
        self.memberships = memberships if not isinstance(memberships, PCCAModel) else memberships.memberships
        self.n_states, self.n_sets = self.memberships.shape

        msm = MembershipsObservable._to_markov_model(test_model)
        symbols = msm.count_model.state_symbols
        symbols_full = msm.count_model.n_states_full
        if isinstance(initial_distribution, str) and initial_distribution == 'stationary_distribution':
            init_dist = msm.stationary_distribution
        else:
            assert isinstance(initial_distribution, np.ndarray) and len(initial_distribution) == msm.n_states, \
                "The initial distribution, if given explicitly, has to be defined on the Markov states of the model."
            init_dist = initial_distribution
        assert self.memberships.shape[0] == msm.n_states, 'provided memberships and test_model n_states mismatch'
        self._test_model = test_model
        # define starting distribution
        P0 = self.memberships * init_dist[:, None]
        P0 /= P0.sum(axis=0)  # column-normalize
        self.P0 = P0

        # map from the full set (here defined by the largest state index in active set) to active
        self._full2active = np.zeros(np.max(symbols_full) + 1, dtype=int)
        self._full2active[symbols] = np.arange(len(symbols))
        self.ignore_imaginary_parts = ignore_imaginary_parts

    def __call__(self, model, mlag=1, **kw):
        if mlag == 0 or model is None:
            return np.eye(self.n_sets)
        model = MembershipsObservable._to_markov_model(model)
        # otherwise compute or predict them by model.propagate
        integer_lag = isinstance(mlag, numbers.Integral)
        if self.ignore_imaginary_parts or (model.is_real and integer_lag and np.all(np.isreal(self.P0))):
            dtype = float
        else:
            dtype = complex
        pk_on_set = np.zeros((self.n_sets, self.n_sets), dtype=dtype)
        # compute observable on prior in case for Bayesian models.
        symbols = model.count_model.state_symbols
        subset = self._full2active[symbols]  # find subset we are now working on
        for i in range(self.n_sets):
            p0 = self.P0[:, i]  # starting distribution on reference active set
            p0sub = p0[subset]  # map distribution to new active set
            if subset is not None:
                p0sub /= np.sum(p0)  # renormalize
            pksub = model.propagate(p0sub, mlag)
            if self.ignore_imaginary_parts and np.any(np.iscomplex(pksub)):
                # ignore complex stuff because of fractional powers
                pksub = pksub.real
            for j in range(self.n_sets):
                pk_on_set[i, j] = np.dot(pksub, self.memberships[subset, j])  # map onto set
        return pk_on_set

    @staticmethod
    def _to_markov_model(model):
        if hasattr(model, 'prior'):
            model = model.prior
        if hasattr(model, 'transition_model'):
            model = model.transition_model
        from deeptime.markov.msm import MarkovStateModel
        assert isinstance(model, MarkovStateModel), f"This should be a Markov state model but was {type(model)}."
        return model
