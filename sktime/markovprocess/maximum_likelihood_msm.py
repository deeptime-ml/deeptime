# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from msmtools import estimation as msmest

from sktime.markovprocess._base import _MSMBaseEstimator
from sktime.markovprocess.transition_counting import TransitionCountEstimator
from sktime.markovprocess.markov_state_model import MarkovStateModel

__all__ = ['MaximumLikelihoodMSM']


class MaximumLikelihoodMSM(_MSMBaseEstimator, ):
    r"""Maximum likelihood estimator for MSMs given discrete trajectory statistics

    Parameters
    ----------
    lagtime : int
        lag time at which transitions are counted and the transition matrix is
        estimated.

    reversible : bool, optional, default = True
        If true compute reversible MarkovStateModel, else non-reversible MarkovStateModel

    statdist : (M,) ndarray, optional
        Stationary vector on the full set of states. Estimation will be
        made such the the resulting transition matrix has this distribution
        as an equilibrium distribution. Set probabilities to zero if these
        states should be excluded from the analysis.

    count_mode : str, optional, default='sliding'
        mode to obtain count matrices from discrete trajectories. Should be
        one of:

        * 'sliding' : A trajectory of length T will have :math:`T-tau` counts
          at time indexes

          .. math::

             (0 \rightarrow \tau), (1 \rightarrow \tau+1), ..., (T-\tau-1 \rightarrow T-1)

        * 'effective' : Uses an estimate of the transition counts that are
          statistically uncorrelated. Recommended when used with a
          Bayesian MarkovStateModel.
        * 'sample' : A trajectory of length T will have :math:`T/tau` counts
          at time indexes

          .. math::

                (0 \rightarrow \tau), (\tau \rightarrow 2 \tau), ..., (((T/tau)-1) \tau \rightarrow T)

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
        string):

        |  'fs',  'femtosecond*'
        |  'ps',  'picosecond*'
        |  'ns',  'nanosecond*'
        |  'us',  'microsecond*'
        |  'ms',  'millisecond*'
        |  's',   'second*'

    maxiter: int, optioanl, default = 1000000
        Optional parameter with reversible = True. maximum number of iterations
        before the transition matrix estimation method exits
    maxerr : float, optional, default = 1e-8
        Optional parameter with reversible = True.
        convergence tolerance for transition matrix estimation.
        This specifies the maximum change of the Euclidean norm of relative
        stationary probabilities (:math:`x_i = \sum_k x_{ik}`). The relative
        stationary probability changes
        :math:`e_i = (x_i^{(1)} - x_i^{(2)})/(x_i^{(1)} + x_i^{(2)})` are used
        in order to track changes in small probabilities. The Euclidean norm
        of the change vector, :math:`|e_i|_2`, is compared to maxerr.

    mincount_connectivity : float or '1/n'
        minimum number of counts to consider a connection between two states.
        Counts lower than that will count zero in the connectivity check and
        may thus separate the resulting transition matrix. The default
        evaluates to 1/nstates.

    References
    ----------
    .. [1] H. Wu and F. Noe: Variational approach for learning Markov processes from time series data
        (in preparation)

    """

    def __init__(self, lagtime=1, reversible=True, statdist_constraint=None,
                 count_mode='sliding', sparse=False,
                 dt_traj='1 step', maxiter=1000000,
                 maxerr=1e-8, mincount_connectivity='1/n'):

        super(MaximumLikelihoodMSM, self).__init__(lagtime=lagtime, reversible=reversible, count_mode=count_mode,
                                                   sparse=sparse, dt_traj=dt_traj,
                                                   mincount_connectivity=mincount_connectivity)

        if statdist_constraint is not None:  # renormalize
            self.statdist_constraint = statdist_constraint.copy()
            self.statdist_constraint /= self.statdist_constraint.sum()
        else:
            self.statdist_constraint = None

        # convergence parameters
        self.maxiter = maxiter
        self.maxerr = maxerr

    def _create_model(self) -> MarkovStateModel:
        return MarkovStateModel()

    def fit(self, dtrajs, y=None):
        count_model = TransitionCountEstimator(lagtime=self.lagtime, count_mode=self.count_mode, dt_traj=self.dt_traj,
            mincount_connectivity=self.mincount_connectivity, stationary_dist_constraint=self.statdist_constraint).fit(
            dtrajs).fetch_model()

        if self.statdist_constraint is not None and count_model.count_matrix_active.sum() == 0.0:
            raise ValueError("The set of states with positive stationary"
                             "probabilities is not visited by the trajectories. A MarkovStateModel"
                             "reversible with respect to the given stationary vector can"
                             "not be estimated")

        # if active set is empty, we can't do anything.
        if count_model.active_set.size == 0:
            raise RuntimeError('Active set is empty. Cannot estimate MarkovStateModel.')

        # active count matrix and number of states
        C_active = count_model.count_matrix_active

        # continue sparse or dense?
        if not self.sparse:
            # converting count matrices to arrays. As a result the
            # transition matrix and all subsequent properties will be
            # computed using dense arrays and dense matrix algebra.
            C_active = C_active.toarray()

        # restrict stationary distribution to active set
        if self.statdist_constraint is None:
            statdist_active = None
        else:
            statdist_active = self.statdist_constraint[count_model.active_set]
            assert np.all(statdist_active > 0.0)
            statdist_active /= statdist_active.sum()  # renormalize

        opt_args = {}
        # TODO: non-rev estimate of msmtools does not comply with its own api...
        if statdist_active is None and self.reversible:
            opt_args['return_statdist'] = True

        # Estimate transition matrix
        P = msmest.transition_matrix(C_active, reversible=self.reversible,
                                     mu=statdist_active, maxiter=self.maxiter,
                                     maxerr=self.maxerr, **opt_args)
        # msmtools returns a tuple for statdist_active=None.
        if isinstance(P, tuple):
            P, statdist_active = P

        # update model parameters
        self._model.__init__(transition_matrix=P, pi=statdist_active, reversible=self.reversible,
                             dt_model=count_model.dt_traj * self.lagtime,
                             count_model=count_model)

        return self


def compute_statistically_effective_count_matrix(dtrajs, lag, active_set=None):
    """

    :param dtrajs:
    :param lag:
    :param active_set:
    :return:
    """
    from sktime.util import submatrix
    Ceff_full = msmest.effective_count_matrix(dtrajs, lag=lag)
    Ceff = submatrix(Ceff_full, active_set)
    return Ceff
