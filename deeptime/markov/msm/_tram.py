import logging
from typing import Optional, Union, List

import numpy as np
from scipy.sparse import issparse

from deeptime.markov.tools import estimation as msmest
from ._markov_state_model import MarkovStateModelCollection, MarkovStateModel
from .._base import _MSMBaseEstimator, MembershipsChapmanKolmogorovValidator
from .._transition_counting import TransitionCountModel, TransitionCountEstimator
from ...numeric import is_square_matrix

__all__ = ['MaximumLikelihoodMSM']

log = logging.getLogger(__file__)


class MaximumLikelihoodMSM(_MSMBaseEstimator):
    r"""Maximum likelihood estimator for MSMs (:class:`MarkovStateModel <deeptime.markov.msm.MarkovStateModel>`)
    given discrete trajectories or statistics thereof. This estimator produces instances of MSMs in form of
    MSM collections (:class:`MarkovStateModelCollection`) which contain as many MSMs as there are connected
    sets in the counting. A collection of MSMs per default behaves exactly like an ordinary MSM model on the largest
    connected set. The connected set can be switched, changing the state of the collection to be have like an MSM on
    the selected state subset.

    Implementation according to :footcite:`wu2020variational`.

    Parameters
    ----------
    reversible : bool, optional, default=True
        If true compute reversible MarkovStateModel, else non-reversible MarkovStateModel
    stationary_distribution_constraint : (N,) ndarray, optional, default=None
        Stationary vector on the full set of states. Estimation will be made such the the resulting transition
        matrix has this distribution as an equilibrium distribution. Set probabilities to zero if the states which
        should be excluded from the analysis.
    sparse : bool, optional, default=False
        If true compute count matrix, transition matrix and all derived quantities using sparse matrix algebra.
        In this case python sparse matrices will be returned by the corresponding functions instead of numpy arrays.
        This behavior is suggested for very large numbers of states (e.g. > 4000) because it is likely to be much
        more efficient.
    allow_disconnected : bool, optional, default=False
        If set to true, the resulting transition matrix may have disconnected and transient states, and the
        estimated stationary distribution is only meaningful on the respective connected sets.
    maxiter : int, optional, default=1000000
        Optional parameter with reversible = True, sets the maximum number of iterations before the transition
        matrix estimation method exits.
    maxerr : float, optional, default = 1e-8
        Optional parameter with reversible = True. Convergence tolerance for transition matrix estimation. This
        specifies the maximum change of the Euclidean norm of relative stationary probabilities
        (:math:`x_i = \sum_k x_{ik}`). The relative stationary probability changes
        :math:`e_i = (x_i^{(1)} - x_i^{(2)})/(x_i^{(1)} + x_i^{(2)})` are used in order to track changes in small
        probabilities. The Euclidean norm of the change vector, :math:`|e_i|_2`, is compared to maxerr.
    transition_matrix_tolerance : float, default=1e-8
        The tolerance under which a matrix is still considered a transition matrix (only non-negative elements and
        row sums of 1).
    connectivity_threshold : float, optional, default=0.
        Number of counts required to consider two states connected.
    lagtime : int, optional, default=None
        Optional lagtime that can be provided at estimator level if fitting from timeseries directly.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, reversible: bool = True, stationary_distribution_constraint: Optional[np.ndarray] = None,
                 sparse: bool = False, allow_disconnected: bool = False, maxiter: int = int(1e6), maxerr: float = 1e-8,
                 connectivity_threshold: float = 0, transition_matrix_tolerance: float = 1e-6, lagtime=None):
        super(MaximumLikelihoodMSM, self).__init__(reversible=reversible, sparse=sparse)

        self.stationary_distribution_constraint = stationary_distribution_constraint
        self.allow_disconnected = allow_disconnected
        self.maxiter = maxiter
        self.maxerr = maxerr
        self.connectivity_threshold = connectivity_threshold
        self.transition_matrix_tolerance = transition_matrix_tolerance
        self.lagtime = lagtime

    @property
    def allow_disconnected(self) -> bool:
        r""" If set to true, the resulting transition matrix may have disconnected and transient states. """
        return self._allow_disconnected

    @allow_disconnected.setter
    def allow_disconnected(self, value: bool):
        self._allow_disconnected = bool(value)

    @property
    def stationary_distribution_constraint(self) -> Optional[np.ndarray]:
        r"""
        The stationary distribution constraint that can either be None (no constraint) or constrains the
        count and transition matrices to states with positive stationary vector entries.

        :getter: Yields the currently configured constraint vector, can be None.
        :setter: Sets a stationary distribution constraint by giving a stationary vector as value. The estimated count-
                 and transition-matrices are restricted to states that have positive entries. In case the vector is not
                 normalized, setting it here implicitly copies and normalizes it.
        :type: ndarray or None
        """
        return self._stationary_distribution_constraint

    @stationary_distribution_constraint.setter
    def stationary_distribution_constraint(self, value: Optional[np.ndarray]):
        if value is not None and (np.any(value < 0) or np.any(value > 1)):
            raise ValueError("not a distribution, contained negative entries and/or entries > 1.")
        if value is not None and np.sum(value) != 1.0:
            # re-normalize if not already normalized
            value = np.copy(value) / np.sum(value)
        self._stationary_distribution_constraint = value

    def fetch_model(self) -> Optional[MarkovStateModelCollection]:
        r"""Yields the most recent :class:`MarkovStateModelCollection` that was estimated.
        Can be None if fit was not called.

        Returns
        -------
        model : MarkovStateModelCollection or None
            The most recent markov state model or None.
        """
        return self._model

    def _fit_connected(self, counts):
        from .. import _transition_matrix as tmat

        if isinstance(counts, np.ndarray):
            if not is_square_matrix(counts) or np.any(counts < 0.):
                raise ValueError("If fitting a count matrix directly, only non-negative square matrices can be used.")
            count_model = TransitionCountModel(counts)
        elif isinstance(counts, TransitionCountModel):
            count_model = counts
        else:
            raise ValueError(f"Unknown type of counts {counts}, only n x n ndarray, TransitionCountModel,"
                             f" or TransitionCountEstimators with a count model are supported.")

        if self.stationary_distribution_constraint is not None:
            if len(self.stationary_distribution_constraint) != count_model.n_states_full:
                raise ValueError(f"Stationary distribution constraint must be defined over full "
                                 f"set of states ({count_model.n_states_full}), but contained "
                                 f"{len(self.stationary_distribution_constraint)} elements.")
            if np.any(self.stationary_distribution_constraint[count_model.state_symbols]) == 0.:
                raise ValueError("The count matrix contains symbols that have no probability in the stationary "
                                 "distribution constraint.")
            if count_model.count_matrix.sum() == 0.0:
                raise ValueError("The set of states with positive stationary probabilities is not visited by the "
                                 "trajectories. A MarkovStateModel reversible with respect to the given stationary"
                                 " vector can not be estimated")

        count_matrix = count_model.count_matrix

        # continue sparse or dense?
        if not self.sparse and issparse(count_matrix):
            # converting count matrices to arrays. As a result the
            # transition matrix and all subsequent properties will be
            # computed using dense arrays and dense matrix algebra.
            count_matrix = count_matrix.toarray()

        # restrict stationary distribution to active set
        if self.stationary_distribution_constraint is None:
            statdist = None
        else:
            statdist = self.stationary_distribution_constraint[count_model.state_symbols]
            statdist /= statdist.sum()  # renormalize

        # Estimate transition matrix
        if self.allow_disconnected:
            P = tmat.estimate_P(count_matrix, reversible=self.reversible, fixed_statdist=statdist,
                                maxiter=self.maxiter, maxerr=self.maxerr)
        else:
            opt_args = {}
            # TODO: non-rev estimate of msmtools does not comply with its own api...
            if statdist is None and self.reversible:
                opt_args['return_statdist'] = True
            P = msmest.transition_matrix(count_matrix, reversible=self.reversible,
                                         mu=statdist, maxiter=self.maxiter,
                                         maxerr=self.maxerr, **opt_args)
        # msmtools returns a tuple for statdist_active=None.
        if isinstance(P, tuple):
            P, statdist = P

        if statdist is None and self.allow_disconnected:
            statdist = tmat.stationary_distribution(P, C=count_matrix)
        return (
            P, statdist, counts
        )

    def _needs_strongly_connected_sets(self):
        return self.reversible and self.stationary_distribution_constraint is None

    def fit_from_counts(self, counts: Union[np.ndarray, TransitionCountEstimator, TransitionCountModel]):
        r""" Fits a model from counts in form of a (n, n) count matrix, a :class:`TransitionCountModel` or an instance
        of `TransitionCountEstimator`, which has been fit on data previously.

        Parameters
        ----------
        counts : (n, n) ndarray or TransitionCountModel or TransitionCountEstimator

        Returns
        -------
        self : MaximumLikelihoodMSM
            Reference to self.
        """
        if isinstance(counts, TransitionCountEstimator):
            if counts.has_model:
                counts = counts.fetch_model()
            else:
                raise ValueError("Can only fit on transition count estimator if the estimator "
                                 "has been fit to data previously.")
        elif isinstance(counts, np.ndarray):
            counts = TransitionCountModel(counts)
        elif isinstance(counts, TransitionCountModel):
            counts = counts
        else:
            raise ValueError("Unknown type of counts argument, can only be one of TransitionCountModel, "
                             "TransitionCountEstimator, (N, N) ndarray. But was: {}".format(type(counts)))
        needs_strong_connectivity = self._needs_strongly_connected_sets()
        if not self.allow_disconnected:
            sets = counts.connected_sets(connectivity_threshold=self.connectivity_threshold,
                                         directed=needs_strong_connectivity)
        else:
            sets = [counts.states]
        transition_matrices = []
        statdists = []
        count_models = []
        for subset in sets:
            try:
                sub_counts = counts.submodel(subset)
                fit_result = self._fit_connected(sub_counts)
                transition_matrices.append(fit_result[0])
                statdists.append(fit_result[1])
                count_models.append(fit_result[2])
            except ValueError as e:
                log.warning(f"Skipping state set {subset} due to error in estimation: {str(e)}.")
        if len(transition_matrices) == 0:
            raise ValueError(f"None of the {'strongly' if needs_strong_connectivity else 'weakly'} "
                             f"connected subsets could be fit to data or the state space decayed into "
                             f"individual states only!")

        self._model = MarkovStateModelCollection(transition_matrices, statdists, reversible=self.reversible,
                                                 count_models=count_models,
                                                 transition_matrix_tolerance=self.transition_matrix_tolerance)
        return self

    def fit_from_discrete_timeseries(self, discrete_timeseries: Union[np.ndarray, List[np.ndarray]],
                                     lagtime: int, count_mode: str = "sliding"):
        r"""Fits a model directly from discrete time series data. This type of data can either be a single
        trajectory in form of a 1d integer numpy array or a list thereof.

        Parameters
        ----------
        discrete_timeseries : ndarray or list of ndarray
            Discrete timeseries data.
        lagtime : int
            The lag time under which to estimate state transitions and ultimately also the transition matrix.
        count_mode : str, default="sliding"
            The count mode to use for estimating transition counts. For maximum-likelihood estimation, the recommended
            choice is "sliding". If the MSM should be used for sampling in a
            :class:`BayesianMSM <deeptime.markov.msm.BayesianMSM>`, the recommended choice is "effective", which yields
            transition counts that are statistically uncorrelated. A description can be found
            in :footcite:`noe2015statistical`.

        Returns
        -------
        self : MaximumLikelihoodMSM
            Reference to self.
        """
        count_model = TransitionCountEstimator(lagtime=lagtime, count_mode=count_mode) \
            .fit(discrete_timeseries).fetch_model()
        return self.fit_from_counts(count_model)

    def fit(self, data, *args, **kw):
        r""" Fits a new markov state model according to data.

        Parameters
        ----------
        data : TransitionCountModel or (n, n) ndarray or discrete timeseries
            Input data, can either be :class:`TransitionCountModel <deeptime.markov.TransitionCountModel>` or
            a 2-dimensional ndarray which is interpreted as count matrix or a discrete timeseries (or a list thereof)
            directly.

            In the case of a timeseries, a lagtime must be provided in the keyword arguments. In this case, also the
            keyword argument "count_mode" can be used, which defaults to "sliding".
            See also :meth:`fit_from_discrete_timeseries`.
        *args
            Dummy parameters for scikit-learn compatibility.
        **kw
            Parameters for scikit-learn compatibility and optionally lagtime if fitting with time series data.

        Returns
        -------
        self : MaximumLikelihoodMSM
            Reference to self.

        See Also
        --------
        TransitionCountModel : Transition count model
        TransitionCountEstimator : Estimating transition count models from data

        Examples
        --------
        This example is demonstrating how to fit a Markov state model collection from data which decomposes into a
        collection of two sets of states with corresponding transition matrices.

        >>> from deeptime.markov.msm import MarkovStateModel  # import MSM
        >>> msm1 = MarkovStateModel([[.7, .3], [.3, .7]])  # create first MSM
        >>> msm2 = MarkovStateModel([[.9, .05, .05], [.3, .6, .1], [.1, .1, .8]])  # create second MSM

        Now, simulate a trajectory where the states of msm2 are shifted by a fixed number `2`, i.e., msm1 describes
        states [0, 1] and msm2 describes states [2, 3, 4] in the generated trajectory.

        >>> traj = np.concatenate([msm1.simulate(1000000), 2 + msm2.simulate(1000000)])  # simulate trajectory

        Given the trajectory, we fit a collection of MSMs:

        >>> model = MaximumLikelihoodMSM(reversible=True).fit(traj, lagtime=1).fetch_model()

        The model behaves like a MSM on the largest connected set, but the behavior can be changed by selecting,
        e.g., the second largest connected set:

        >>> model.state_symbols()
        array([2, 3, 4])
        >>> model.select(1)  # change to second largest connected set
        >>> model.state_symbols()
        array([0, 1])

        And this is all the models contained in the collection:

        >>> model.n_connected_msms
        2

        Alternatively, one can fit with a previously estimated count model (that can be restricted to a subset
        of states):

        >>> counts = TransitionCountEstimator(lagtime=1, count_mode="sliding").fit(traj).fetch_model()
        >>> counts = counts.submodel([0, 1])  # select submodel with state symbols [0, 1]
        >>> msm = MaximumLikelihoodMSM(reversible=True).fit(counts).fetch_model()
        >>> msm.state_symbols()
        array([0, 1])

        And this is the only model in the collection:

        >>> msm.n_connected_msms
        1
        """
        if isinstance(data, (TransitionCountModel, TransitionCountEstimator)):
            return self.fit_from_counts(data)
        elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0] == data.shape[1]:
            return self.fit_from_counts(data)
        else:
            if 'lagtime' not in kw.keys() and self.lagtime is None:
                raise ValueError("To fit directly from a discrete timeseries, a lagtime must be provided!")
            return self.fit_from_discrete_timeseries(data, kw.pop('lagtime', self.lagtime),
                                                     kw.pop("count_mode", "sliding"))

    def chapman_kolmogorov_validator(self, n_metastable_sets: int, mlags,
                                     test_model: Optional[MarkovStateModel] = None):
        r"""Returns a Chapman-Kolmogorov validator based on this estimator and a test model.

        Parameters
        ----------
        n_metastable_sets : int
            Number of metastable sets to project the state space down to.
        mlags : int or range or list
            Multiple of lagtimes of the test_model to test against.
        test_model : MarkovStateModel, optional, default=None
            The model that is tested. If not provided, uses this estimator's encapsulated model.

        Returns
        -------
        validator : markov.MembershipsChapmanKolmogorovValidator
            The validator that can be fit on data.

        Raises
        ------
        AssertionError
            If test_model is None and this estimator has not been :meth:`fit` on data yet or the output model
            was not a discrete output model.
        """
        test_model = self.fetch_model() if test_model is None else test_model
        assert test_model is not None, "We need a test model via argument or an estimator which was already" \
                                       "fit to data."
        assert test_model.has_count_model, "The test model needs to have a count model, i.e., be estimated from data."
        pcca = test_model.pcca(n_metastable_sets)
        reference_lagtime = test_model.count_model.lagtime
        return MLMSMChapmanKolmogorovValidator(test_model, self, pcca.memberships, reference_lagtime, mlags)


def _ck_estimate_model_for_lag(estimator, model, data, lag):
    counting_mode = model.count_model.counting_mode
    counts = TransitionCountEstimator(lag, counting_mode).fit(data).fetch_model().submodel_largest()
    return estimator.fit(counts).fetch_model()


class MLMSMChapmanKolmogorovValidator(MembershipsChapmanKolmogorovValidator):

    def fit(self, data, n_jobs=None, progress=None, **kw):
        return super().fit(data, n_jobs, progress, estimate_model_for_lag=_ck_estimate_model_for_lag, **kw)
