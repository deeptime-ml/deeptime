import numbers
from typing import Optional, Union, Callable

import numpy as np
from threadpoolctl import threadpool_limits

from ..base import Estimator
from ..numeric import is_sorted, spd_inv_sqrt, schatten_norm
from ..util.parallel import joining


def vamp_score(koopman_model, r: Union[float, str],
               covariances_test=None, dim: Optional[int] = None, epsilon: float = 1e-10):
    """Compute the VAMP score between a covariance-based Koopman model and potentially a
    test model for cross-validation.

    Parameters
    ----------
    koopman_model : deeptime.decomposition.CovarianceKoopmanModel
        The model to score.
    r : float or str
        The type of score to evaluate. Can by an floating point value greater or equal to 1 or 'E', yielding the
        VAMP-r score or the VAMP-E score, respectively. :footcite:`wu2020variational`
        Typical choices (also accepted as inputs) are:

        *  'VAMP1'  Sum of singular values of the half-weighted Koopman matrix.
                    If the model is reversible, this is equal to the sum of
                    Koopman matrix eigenvalues, also called Rayleigh quotient :footcite:`wu2020variational`.
        *  'VAMP2'  Sum of squared singular values of the half-weighted Koopman
                    matrix :footcite:`wu2020variational`. If the model is reversible, this is
                    equal to the kinetic variance :footcite:`noe2015kinetic`.
        *  'VAMPE'  Approximation error of the estimated Koopman operator with respect to
                    the true Koopman operator up to an additive constant :footcite:`wu2020variational` .

    covariances_test : deeptime.covariance.CovarianceModel, optional, default=None

        If `test_model` is not None, this method computes the cross-validation score
        between self and `covariances_test`. It is assumed that self was estimated from
        the "training" data and `test_model` was estimated from the "test" data. The
        score is computed for one realization of self and `test_model`. Estimation
        of the average cross-validation score and partitioning of data into test and
        training part is not performed by this method.

        If `covariances_test` is None, this method computes the VAMP score for the model
        contained in self.

    dim : int, optional, default=None
        Artificially restrict the scoring to the top `dim` slowest processes.

    epsilon : float, default=1e-10


    Returns
    -------
    score : float
        If `test_model` is not None, returns the cross-validation VAMP score between
        self and `test_model`. Otherwise return the selected VAMP-score of self.

    Notes
    -----
    If the Koopman model was estimated using correlations that are based on data with its sample mean removed,
    this effectively removes the constant function from the singular function space and artificially lowers the score
    by 1. This is accounted for in this method, i.e., if :code:`koopman_model.cov.data_mean_removed` evaluates to
    `True`, the score is internally incremented by 1.

    The VAMP-:math:`r` and VAMP-E scores are computed according to :footcite:`wu2020variational`,
    Equation (33) and Equation (30), respectively.

    References
    ----------
    .. footbibliography::
    """
    if dim is not None:
        dim = min(koopman_model.koopman_matrix.shape[0], dim)
    if isinstance(r, str):
        r = r.lower()
        r = r.replace("vamp", "")
        if r.isnumeric():
            r = float(r)
        else:
            assert r == 'e', "only VAMP-E supported, otherwise give as float >= 1"
    else:
        assert isinstance(r, numbers.Number) and r >= 1, "score only for r >= 1 or r = \"E\""
    if covariances_test is None:
        cov_test = koopman_model.cov
    else:
        cov_test = covariances_test
    assert koopman_model.cov.data_mean_removed == cov_test.data_mean_removed, \
        "Covariances must be consistent with respect to the data"
    if koopman_model.cov.cov_00.shape != cov_test.cov_00.shape:
        raise ValueError(f"Shape mismatch, the covariances had "
                         f"shapes {koopman_model.cov.cov_00.shape} and {cov_test.cov_00.shape}.")
    if not is_sorted(koopman_model.singular_values, 'desc'):
        sort_ix = np.argsort(koopman_model.singular_values)[::-1][:dim]  # indices to sort in descending order
    else:
        sort_ix = np.arange(koopman_model.singular_values.shape[0])[:dim]  # already sorted

    U = koopman_model.instantaneous_coefficients[:, sort_ix]
    V = koopman_model.timelagged_coefficients[:, sort_ix]

    if r == 'e':
        K = np.diag(koopman_model.singular_values[sort_ix])
        # see https://arxiv.org/pdf/1707.04659.pdf eqn. (30)
        score = np.trace(2.0 * np.linalg.multi_dot([K, U.T, cov_test.cov_0t, V])
                         - np.linalg.multi_dot([K, U.T, cov_test.cov_00, U, K, V.T, cov_test.cov_tt, V]))
    else:
        # see https://arxiv.org/pdf/1707.04659.pdf eqn. (33)
        A = np.atleast_2d(spd_inv_sqrt(U.T.dot(cov_test.cov_00).dot(U), epsilon=epsilon))
        B = np.atleast_2d(U.T.dot(cov_test.cov_0t).dot(V))
        C = np.atleast_2d(spd_inv_sqrt(V.T.dot(cov_test.cov_tt).dot(V), epsilon=epsilon))
        ABC = np.linalg.multi_dot([A, B, C])
        score = schatten_norm(ABC, r) ** r
    if koopman_model.cov.data_mean_removed:
        score += 1
    return score


def vamp_score_data(data, data_lagged, transformation=None, r=2, epsilon=1e-6, dim=None):
    r""" Computes VAMP score based on data and corresponding time-lagged data.
    Can be equipped with a transformation, defaults to 'identity' transformation.

    Parameters
    ----------
    data : (T, n) ndarray
        Instantaneous data.
    data_lagged : (T, n) ndarray
        Time-lagged data.
    transformation : Callable
        Transformation on data that will be scored.
    r : int or str, optional, default=2
        The type of VAMP score evaluated, see :meth:`deeptime.decomposition.vamp_score`.
    epsilon : float, optional, default=1e-6
        Regularization parameter for the score, see :meth:`deeptime.decomposition.vamp_score`.
    dim : int, optional, default=None
        Number of components that should be scored. Defaults to all components. See
        :meth:`deeptime.decomposition.vamp_score`.

    Returns
    -------
    score : float
        The VAMP score.

    See Also
    --------
    vamp_score
    """
    if transformation is None:
        def transformation(x):
            return x
    from deeptime.decomposition import VAMP
    model = VAMP(epsilon=epsilon, observable_transform=transformation).fit((data, data_lagged)).fetch_model()
    return model.score(r=r, dim=dim, epsilon=epsilon)


def blocksplit_trajs(trajs, lag=1, sliding=True, shift=None, random_state=None):
    """ Splits trajectories into approximately uncorrelated fragments.

    Will split trajectories into fragments of lengths lag or longer. These fragments
    are overlapping in order to conserve the transition counts at given lag.
    If sliding=True, the resulting trajectories will lead to exactly the same count
    matrix as when counted from dtrajs. If sliding=False (sampling at lag), the
    count matrices are only equal when also setting shift=0.

    Parameters
    ----------
    trajs : list of ndarray(int)
        Trajectories
    lag : int
        Lag time at which counting will be done.
    sliding : bool
        True for splitting trajectories for sliding count, False if lag-sampling will be applied
    shift : None or int
        Start of first full tau-window. If None, shift will be randomly generated
    random_state : None or int or np.random.RandomState
        Random seed to use.

    Returns
    -------
    blocks : list of ndarray
        The blocks.
    """
    from sklearn.utils.random import check_random_state
    random_state = check_random_state(random_state)
    blocks = []
    for traj in trajs:
        if len(traj) <= lag:
            continue
        if shift is None:
            s = random_state.randint(min(lag, traj.size - lag))
        else:
            s = shift
        if sliding:
            if s > 0:
                blocks.append(traj[:lag + s])
            for t0 in range(s, len(traj) - lag, lag):
                blocks.append(traj[t0:t0 + 2 * lag])
        else:
            for t0 in range(s, len(traj) - lag, lag):
                blocks.append(traj[t0:t0 + lag + 1])
    return blocks


def cvsplit_trajs(trajs, random_state=None):
    """ Splits the trajectories into a training and test set with approximately equal number of trajectories

    Parameters
    ----------
    trajs : list of ndarray(int)
        Discrete trajectories
    random_state : None or int or np.random.RandomState
        Random seed to use.
    """
    from sklearn.utils.random import check_random_state
    assert len(trajs) > 1, 'Only have a single trajectory. Cannot be split into train and test set'
    random_state = check_random_state(random_state)
    I0 = random_state.choice(len(trajs), int(len(trajs) / 2), replace=False)
    I1 = np.array(list(set(list(np.arange(len(trajs)))) - set(list(I0))))
    train_set = [trajs[i] for i in I0]
    test_set = [trajs[i] for i in I1]
    return train_set, test_set


def vamp_score_cv(fit_fetch: Union[Estimator, Callable], trajs, lagtime=None, n=10, splitting_mode="sliding", r=2,
                  dim: Optional[int] = None, blocksplit: bool = True, random_state=None, n_jobs=1):
    r""" Scores the MSM using the variational approach for Markov processes and cross-validation.

    Implementation and ideas following :footcite:`noe2013variational` :footcite:`wu2020variational` and
    cross-validation :footcite:`mcgibbon2015variational`.

    Divides the data into training and test data, fits a MSM using the training
    data using the parameters of this estimator, and scores is using the test
    data.
    Currently only one way of splitting is implemented, where for each n,
    the data is randomly divided into two approximately equally large sets of
    discrete trajectory fragments with lengths of at least the lagtime.

    Currently only implemented using dense matrices - will be slow for large state spaces.

    Parameters
    ----------
    fit_fetch : callable or estimator
        Can be provided as callable for a custom fit and fetch method. Should be a function pointer or lambda which
        takes a list of discrete trajectories as input and yields a
        :class:`CovarianceKoomanModel <deeptime.decomposition.CovarianceKoopmanModel>`. Or an estimator which
        yields this kind of model.
    trajs : list of array_like
        Input data.
    lagtime : int, optional, default=None
        lag time, must be provided if blocksplitting is used, otherwise can be left None
    splitting_mode : str, optional, default="sliding"
        Can be one of "sliding" and "sample". In former case the blocks may overlap, otherwise not.
    n : number of samples
        Number of repetitions of the cross-validation. Use large n to get solid means of the score.
    r : float or str, default=2
        Available scores are based on the variational approach for Markov processes :footcite:`noe2013variational`
        :footcite:`wu2020variational`, see :meth:`deeptime.decomposition.vamp_score` for available options.
    blocksplit : bool, optional, default=True
        Whether to perform blocksplitting (see :meth:`blocksplit_dtrajs` ) before evaluating folds. Defaults to `True`.
        In case no blocksplitting is performed, individual dtrajs are used for training and validation. This means that
        at least two dtrajs must be provided (`len(dtrajs) >= 2`), otherwise this method raises an exception.
    dim : int or None, optional, default=None
        The maximum number of eigenvalues or singular values used in the score. If set to None,
        all available eigenvalues will be used.
    random_state : None or int or np.random.RandomState
        Random seed to use.
    n_jobs : int, optional, default=1
        Number of jobs for folds. In case n_jobs is 1, no parallelization.

    References
    ----------
    .. footbibliography::
    """
    from deeptime.util.parallel import handle_n_jobs
    from deeptime.util.types import ensure_timeseries_data

    if blocksplit and lagtime is None:
        raise ValueError("In case blocksplit is used, please provide a lagtime.")

    n_jobs = handle_n_jobs(n_jobs)
    if isinstance(fit_fetch, Estimator):
        fit_fetch = _FitFetch(fit_fetch)

    ttrajs = ensure_timeseries_data(trajs)  # ensure format
    if splitting_mode not in ('sliding', 'sample'):
        raise ValueError('vamp_score_cv currently only supports count modes "sliding" and "sample"')
    scores = np.empty((n,), float)
    sliding = splitting_mode == 'sliding'

    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    assert isinstance(random_state, np.random.RandomState)

    args = [(i, fit_fetch, ttrajs, r, dim, lagtime, blocksplit, sliding, random_state, n_jobs) for i in range(n)]

    if n_jobs > 1:
        from multiprocessing import get_context
        with joining(get_context("spawn").Pool(processes=n_jobs)) as pool:
            for result in pool.imap_unordered(_worker, args):
                fold, score = result
                scores[fold] = score
    else:
        for fold in range(n):
            _, score = _worker(args[fold])
            scores[fold] = score
    return scores


class _FitFetch:

    def __init__(self, est):
        self._est = est

    def __call__(self, x):
        return self._est.fit(x).fetch_model()


def _worker(args):
    from deeptime.markov.msm import MarkovStateModel
    fold, fit_fetch, ttrajs, r, dim, lagtime, blocksplit, sliding, random_state, n_jobs = args

    with threadpool_limits(limits=1 if n_jobs > 1 else None, user_api='blas'):
        if blocksplit:
            trajs_split = blocksplit_trajs(ttrajs, lag=lagtime, sliding=sliding,
                                           random_state=random_state)
        else:
            trajs_split = ttrajs
            assert len(trajs_split) > 1, "Need at least two trajectories if blocksplit " \
                                         "is not used to decompose the data."
        trajs_train, trajs_test = cvsplit_trajs(trajs_split, random_state=random_state)
        # this is supposed to construct a markov state model from data directly, for example what fit_fetch could do is
        train_model = fit_fetch(trajs_train)
        if isinstance(train_model, MarkovStateModel):
            score = train_model.score(trajs_test, r=r, dim=dim)
        else:
            test_model = fit_fetch(trajs_test)
            score = train_model.score(r=r, test_model=test_model, dim=dim)
        return fold, score
