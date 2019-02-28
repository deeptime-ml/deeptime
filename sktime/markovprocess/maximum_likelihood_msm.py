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
from sktime.markovprocess.markov_state_model import MarkovStateModel

__all__ = ['MaximumLikelihoodMSMBase']


class MaximumLikelihoodMSMBase(_MSMBaseEstimator):
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
    connectivity : str, optional, default = 'largest'
        Connectivity mode. Three methods are intended (currently only 'largest'
        is implemented)

        * 'largest' : The active set is the largest reversibly connected set.
          All estimation will be done on this subset and all quantities
          (transition matrix, stationary distribution, etc) are only defined
          on this subset and are correspondingly smaller than the full set
          of states
        * 'all' : The active set is the full set of states. Estimation will be
          conducted on each reversibly connected set separately. That means
          the transition matrix will decompose into disconnected submatrices,
          the stationary vector is only defined within subsets, etc.
          Currently not implemented.
        * 'none' : The active set is the full set of states. Estimation will
          be conducted on the full set of
          states without ensuring connectivity. This only permits
          nonreversible estimation. Currently not implemented.

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

    score_method : str, optional, default='VAMP2'
        Score to be used with score function. Available are:

        |  'VAMP1'  [1]_
        |  'VAMP2'  [1]_
        |  'VAMPE'  [1]_

    score_k : int or None
        The maximum number of eigenvalues or singular values used in the
        score. If set to None, all available eigenvalues will be used.

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
                 connectivity='largest', dt_traj='1 step', maxiter=1000000,
                 maxerr=1e-8, score_method='VAMP2', score_k=10,
                 mincount_connectivity='1/n'):

        super(MaximumLikelihoodMSMBase, self).__init__(lagtime=lagtime, reversible=reversible, count_mode=count_mode,
                                                       sparse=sparse, connectivity=connectivity, dt_traj=dt_traj,
                                                       score_method=score_method, score_k=score_k,
                                                       mincount_connectivity=mincount_connectivity)

        self.statdist_constraint = statdist_constraint
        if self.statdist_constraint is not None:  # renormalize
            self.statdist_constraint /= self.statdist_constraint.sum()

        # convergence parameters
        self.maxiter = maxiter
        self.maxerr = maxerr

    def _create_model(self) -> MarkovStateModel:
        return MarkovStateModel(None)

    def _prepare_input_revpi(self, C, pi):
        """Max. state index visited by trajectories"""
        nC = C.shape[0]
        # Max. state index of the stationary vector array
        npi = pi.shape[0]
        # pi has to be defined on all states visited by the trajectories
        if nC > npi:
            raise ValueError('There are visited states for which no stationary probability is given')
        # Reduce pi to the visited set
        pi_visited = pi[0:nC]
        # Find visited states with positive stationary probabilities"""
        pos = np.where(pi_visited > 0.0)[0]
        # Reduce C to positive probability states"""
        C_pos = msmest.connected_cmatrix(C, lcc=pos)
        if C_pos.sum() == 0.0:
            raise ValueError("The set of states with positive stationary"
                             "probabilities is not visited by the trajectories. A MarkovStateModel"
                             "reversible with respect to the given stationary vector can"
                             "not be estimated")
        # Compute largest connected set of C_pos, undirected connectivity"""
        lcc = msmest.largest_connected_set(C_pos, directed=False)
        return pos[lcc]

    def fit(self, dtrajs):
        """ Estimates the MarkovStateModel """
        # get trajectory counts. This sets _C_full and _nstates_full
        dtrajstats = self._get_dtraj_stats(dtrajs)
        self._C_full = dtrajstats.count_matrix()  # full count matrix
        self._nstates_full = self._C_full.shape[0]  # number of states

        # set active set. This is at the same time a mapping from active to full
        if self.connectivity == 'largest':
            if self.statdist_constraint is None:
                # statdist not given - full connectivity on all states
                self.active_set = dtrajstats.largest_connected_set
            else:
                active_set = self._prepare_input_revpi(self._C_full,
                                                       self.statdist_constraint)
                self.active_set = active_set
        else:
            # for 'None' and 'all' all visited states are active
            self.active_set = dtrajstats.visited_set

        # if active set is empty, we can't do anything.
        if np.size(self.active_set) == 0:
            raise RuntimeError('Active set is empty. Cannot estimate MarkovStateModel.')

        # active count matrix and number of states
        self._C_active = dtrajstats.count_matrix(subset=self.active_set)

        # continue sparse or dense?
        if not self.sparse:
            # converting count matrices to arrays. As a result the
            # transition matrix and all subsequent properties will be
            # computed using dense arrays and dense matrix algebra.
            self._C_full = self._C_full.toarray()
            self._C_active = self._C_active.toarray()

        self._nstates = self._C_active.shape[0]

        # computed derived quantities
        # back-mapping from full to lcs
        self._full2active = -1 * np.ones(dtrajstats.nstates, dtype=int)
        self._full2active[self.active_set] = np.arange(len(self.active_set))

        # restrict stationary distribution to active set
        if self.statdist_constraint is None:
            statdist_active = None
        else:
            statdist_active = self.statdist_constraint[self.active_set]
            statdist_active /= statdist_active.sum()  # renormalize

        opt_args = {}
        # TODO: non-rev estimate of msmtools does not comply with its own api...
        if statdist_active is None and self.reversible:
            opt_args['return_statdist'] = True

        # Estimate transition matrix
        if self.connectivity == 'largest':
            P = msmest.transition_matrix(self._C_active, reversible=self.reversible,
                                         mu=statdist_active, maxiter=self.maxiter,
                                         maxerr=self.maxerr, **opt_args)
        elif self.connectivity == 'none':
            # reversible mode only possible if active set is connected
            # - in this case all visited states are connected and thus
            # this mode is identical to 'largest'
            if self.reversible and not msmest.is_connected(self._C_active):
                raise ValueError('Reversible MarkovStateModel estimation is not possible with connectivity mode "none", '
                                 'because the set of all visited states is not reversibly connected')
            P = msmest.transition_matrix(self._C_active, reversible=self.reversible,
                                         mu=statdist_active,
                                         maxiter=self.maxiter, maxerr=self.maxerr,
                                         **opt_args
                                         )
        else:
            raise NotImplementedError(
                'MarkovStateModel estimation with connectivity=%s is currently not implemented.' % self.connectivity)

        # msmtools returns a tuple for statdist_active = None.
        if isinstance(P, tuple):
            P, statdist_active = P

        # Done. We set our own model parameters, so this estimator is
        # equal to the estimated model.
        self._dtrajs_full = dtrajs
        self._connected_sets = dtrajstats.connected_sets

        # update model parameters
        m = self._model
        m.transition_matrix = P
        m.stationary_distribution = statdist_active
        m.reversible = self.reversible
        # lag time model:
        m.dt_model = self.timestep_traj * self.lagtime

        return self

    # TODO: change to statistically effective count matrix!
    @property
    def effective_count_matrix(self):
        """Statistically uncorrelated transition counts within the active set of states

        You can use this count matrix for Bayesian estimation or error perturbation.

        References
        ----------
        [1] Noe, F. (2015) Statistical inefficiency of Markov model count matrices
            http://publications.mi.fu-berlin.de/1699/1/autocorrelation_counts.pdf

        """
        Ceff_full = msmest.effective_count_matrix(self._dtrajs_full, self.lagtime)
        from pyemma.util.linalg import submatrix
        Ceff = submatrix(Ceff_full, self.active_set)
        return Ceff
        # return self._C_active / float(self.lagtime)
