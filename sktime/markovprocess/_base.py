import numpy as np

from sktime.base import Estimator
from sktime.markovprocess._dtraj_stats import blocksplit_dtrajs, cvsplit_dtrajs


# TODO: distinguish more between model and estimator attributes.

class _MSMBaseEstimator(Estimator):
    r"""Maximum likelihood estimator for MSMs given discrete trajectory statistics

    Parameters
    ----------
    lag : int
        lag time at which transitions are counted and the transition matrix is
        estimated.

    reversible : bool, optional, default = True
        If true compute reversible MarkovStateModel, else non-reversible MarkovStateModel

    count_mode : str, optional, default='sliding'
        mode to obtain count matrices from discrete trajectories. Should be
        one of:

        * 'sliding' : A trajectory of length T will have :math:`T-\tau` counts
          at time indexes

          .. math::

             (0 \rightarrow \tau), (1 \rightarrow \tau+1), ..., (T-\tau-1 \rightarrow T-1)

        * 'effective' : Uses an estimate of the transition counts that are
          statistically uncorrelated. Recommended when used with a
          Bayesian MarkovStateModel.
        * 'sample' : A trajectory of length T will have :math:`T/\tau` counts
          at time indexes

          .. math::

                (0 \rightarrow \tau), (\tau \rightarrow 2 \tau), ..., (((T/\tau)-1) \tau \rightarrow T)

    sparse : bool, optional, default = False
        If true compute count matrix, transition matrix and all derived
        quantities using sparse matrix algebra. In this case python sparse
        matrices will be returned by the corresponding functions instead of
        numpy arrays. This behavior is suggested for very large numbers of
        states (e.g. > 4000) because it is likely to be much more efficient.

    dt_traj : str, optional, default='1 step'
        Description of the physical time of the input trajectories. May be used
        by analysis algorithms such as plotting tools to pretty-print the axes.
        By default '1 step', i.e. there is no physical time unit. Specify by a
        number, whitespace and unit. Permitted units are (* is an arbitrary
        string). E.g. 200 picoseconds or 200ps.

    mincount_connectivity : float or '1/n'
        minimum number of counts to consider a connection between two states.
        Counts lower than that will count zero in the connectivity check and
        may thus separate the resulting transition matrix. The default
        evaluates to 1/nstates.

    """

    def __init__(self, lagtime=1, reversible=True, count_mode='sliding', sparse=False,
                 dt_traj='1 step', mincount_connectivity='1/n'):
        super(_MSMBaseEstimator, self).__init__()
        self.lagtime = lagtime

        # set basic parameters
        self.reversible = reversible

        # sparse matrix computation wanted?
        self.sparse = sparse

        # store counting mode (lowercase)
        self.count_mode = count_mode
        if self.count_mode not in ('sliding', 'effective', 'sample'):
            raise ValueError('count mode ' + count_mode + ' is unknown.')

        # time step
        self.dt_traj = dt_traj

        # connectivity
        self.mincount_connectivity = mincount_connectivity

    # TODO: this one is tricky
    def cktest(self, dtrajs, nsets, memberships=None, mlags=10, conf=0.95, err_est=False,
               n_jobs=None, show_progress=True):
        """ Conducts a Chapman-Kolmogorow test.

        Parameters
        ----------
        nsets : int
            number of sets to test on
        memberships : ndarray(nstates, nsets), optional
            optional state memberships. By default (None) will conduct a cktest
            on PCCA (metastable) sets.
        mlags : int or int-array, optional
            multiples of lag times for testing the Model, e.g. range(10).
            A single int will trigger a range, i.e. mlags=10 maps to
            mlags=range(10). The setting None will choose mlags automatically
            according to the longest available trajectory
        conf : float, optional
            confidence interval
        err_est : bool, optional
            compute errors also for all estimations (computationally expensive)
            If False, only the prediction will get error bars, which is often
            sufficient to validate a model.
        n_jobs : int, default=None
            how many jobs to use during calculation
        show_progress : bool, optional
            Show progress bars for calculation?

        Returns
        -------
        cktest : :class:`ChapmanKolmogorovValidator <pyemma.msm.ChapmanKolmogorovValidator>`


        References
        ----------
        This test was suggested in [1]_ and described in detail in [2]_.

        .. [1] F. Noe, Ch. Schuette, E. Vanden-Eijnden, L. Reich and
            T. Weikl: Constructing the Full Ensemble of Folding Pathways
            from Short Off-Equilibrium Simulations.
            Proc. Natl. Acad. Sci. USA, 106, 19011-19016 (2009)
        .. [2] Prinz, J H, H Wu, M Sarich, B Keller, M Senne, M Held, J D
            Chodera, C Schuette and F Noe. 2011. Markov models of
            molecular kinetics: Generation and validation. J Chem Phys
            134: 174105

        """
        test_model = self.fetch_model()
        if memberships is None:
            test_model.pcca(nsets)
            memberships = test_model.metastable_memberships
        from sktime.markovprocess.lagged_model_validators import ChapmanKolmogorovValidator
        ck = ChapmanKolmogorovValidator(test_estimator=self, test_model=test_model, memberships=memberships, mlags=mlags, conf=conf,
                                        n_jobs=n_jobs, err_est=err_est, show_progress=show_progress)
        ck.fit(dtrajs)
        return ck


def score_cv(estimator: _MSMBaseEstimator, dtrajs, n=10, score_method='VAMP2', score_k=10, random_state=None):
    r""" Scores the MSM using the variational approach for Markov processes [1]_ [2]_ and cross-validation [3]_ .

    Divides the data into training and test data, fits a MSM using the training
    data using the parameters of this estimator, and scores is using the test
    data.
    Currently only one way of splitting is implemented, where for each n,
    the data is randomly divided into two approximately equally large sets of
    discrete trajectory fragments with lengths of at least the lagtime.

    Currently only implemented using dense matrices - will be slow for large state spaces.

    Parameters
    ----------
    estimator : MSMBaseEstimator like
        estimator to produce models for CV.
    dtrajs : list of arrays
        Test data (discrete trajectories).
    n : number of samples
        Number of repetitions of the cross-validation. Use large n to get solid
        means of the score.
    score_method : str, optional, default='VAMP2'
        Overwrite scoring method to be used if desired. If `None`, the estimators scoring
        method will be used.
        Available scores are based on the variational approach for Markov processes [1]_ [2]_ :

        *  'VAMP1'  Sum of singular values of the symmetrized transition matrix [2]_ .
                    If the MSM is reversible, this is equal to the sum of transition
                    matrix eigenvalues, also called Rayleigh quotient [1]_ [3]_ .
        *  'VAMP2'  Sum of squared singular values of the symmetrized transition matrix [2]_ .
                    If the MSM is reversible, this is equal to the kinetic variance [4]_ .

    score_k : int or None
        The maximum number of eigenvalues or singular values used in the
        score. If set to None, all available eigenvalues will be used.

    References
    ----------
    .. [1] Noe, F. and F. Nueske: A variational approach to modeling slow processes
        in stochastic dynamical systems. SIAM Multiscale Model. Simul. 11, 635-655 (2013).
    .. [2] Wu, H and F. Noe: Variational approach for learning Markov processes
        from time series data (in preparation).
    .. [3] McGibbon, R and V. S. Pande: Variational cross-validation of slow
        dynamical modes in molecular kinetics, J. Chem. Phys. 142, 124105 (2015).
    .. [4] Noe, F. and C. Clementi: Kinetic distance and kinetic maps from molecular
        dynamics simulation. J. Chem. Theory Comput. 11, 5002-5011 (2015).

    """
    from sktime.util import ensure_dtraj_list
    dtrajs = ensure_dtraj_list(dtrajs)  # ensure format
    if estimator.count_mode not in ('sliding', 'sample'):
        raise ValueError('score_cv currently only supports count modes "sliding" and "sample"')
    sliding = estimator.count_mode == 'sliding'
    scores = []
    for fold in range(n):
        dtrajs_split = blocksplit_dtrajs(dtrajs, lag=estimator.lagtime, sliding=sliding, random_state=random_state)
        dtrajs_train, dtrajs_test = cvsplit_dtrajs(dtrajs_split, random_state=random_state)
        model = estimator.fit(dtrajs_train).fetch_model()
        s = model.score(dtrajs_test, score_method=score_method, score_k=score_k)
        scores.append(s)
    return np.array(scores)
