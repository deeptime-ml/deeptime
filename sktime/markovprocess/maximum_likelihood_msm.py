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
from typing import Optional, Union

import numpy as np
from msmtools import estimation as msmest

from sktime.markovprocess import Q_
from sktime.markovprocess._base import _MSMBaseEstimator
from sktime.markovprocess.markov_state_model import MarkovStateModel
from sktime.markovprocess.transition_counting import TransitionCountModel

__all__ = ['MaximumLikelihoodMSM']


class MaximumLikelihoodMSM(_MSMBaseEstimator):
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

        * 'sliding-effective' : Same as 'sliding' but after counting all counts are
          divided by the lagtime :math:`\tau`.

        * 'effective' : Uses an estimate of the transition counts that are statistically uncorrelated.
          Recommended when used with a Bayesian MarkovStateModel.

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

    physical_time : str, optional, default='1 step'
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

    connectivity_threshold : float or '1/n'
        minimum number of counts to consider a connection between two states.
        Counts lower than that will count zero in the connectivity check and
        may thus separate the resulting transition matrix. The default
        evaluates to 1/n_states.

    References
    ----------
    .. [1] H. Wu and F. Noe: Variational approach for learning Markov processes from time series data
        (in preparation)

    """

    _MUTABLE_INPUT_DATA = True

    def __init__(self, lagtime: int = 1, reversible: bool = True,
                 stationary_distribution_constraint: Optional[np.ndarray] = None,
                 count_mode: str = 'sliding', sparse: bool = False,
                 physical_time: Union[Q_, str] = '1 step', maxiter: int = int(1e6),
                 maxerr: float = 1e-8, connectivity_threshold='1/n'):

        super(MaximumLikelihoodMSM, self).__init__(lagtime=lagtime, reversible=reversible, count_mode=count_mode,
                                                   sparse=sparse, physical_time=physical_time,
                                                   connectivity_threshold=connectivity_threshold)

        self.stationary_distribution_constraint = stationary_distribution_constraint

        # convergence parameters
        self.maxiter = maxiter
        self.maxerr = maxerr

    @property
    def stationary_distribution_constraint(self) -> Optional[np.ndarray]:
        r"""
        Yields the stationary distribution constraint that can either be None (no constraint) or constrains the
        count and transition matrices to states with positive stationary vector entries.

        Returns
        -------
        The stationary vector constraint, can be None
        """
        return self._stationary_distribution_constraint

    @stationary_distribution_constraint.setter
    def stationary_distribution_constraint(self, value: Optional[np.ndarray]):
        r"""
        Sets a stationary distribution constraint by giving a stationary vector as value. The estimated count- and
        transition-matrices are restricted to states that have positive entries. In case the vector is not normalized,
        setting it here implicitly copies and normalizes it.

        Parameters
        ----------
        value : np.ndarray or None
            the stationary vector
        """
        if value is not None and np.sum(value) != 1.0:
            # re-normalize if not already normalized
            value = np.copy(value) / np.sum(value)
        self._stationary_distribution_constraint = value

    def fetch_model(self) -> MarkovStateModel:
        return self._model

    def fit(self, data, **kw):
        if not isinstance(data, (TransitionCountModel, np.ndarray)):
            raise ValueError("Can only fit on a TransitionCountModel or a count matrix directly.")

        if isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[0] != data.shape[1] or np.any(data < 0.):
                raise ValueError("If fitting a count matrix directly, only non-negative square matrices can be used.")
            count_model = TransitionCountModel(data)
        else:
            count_model = data

        if self.stationary_distribution_constraint is not None:
            if np.any(self.stationary_distribution_constraint[count_model.state_symbols]) == 0.:
                raise ValueError("The count matrix contains symbols that have no probability in the stationary "
                                 "distribution constraint.")
            if count_model.count_matrix.sum() == 0.0:
                raise ValueError("The set of states with positive stationary probabilities is not visited by the "
                                 "trajectories. A MarkovStateModel reversible with respect to the given stationary"
                                 " vector can not be estimated")

        count_matrix = count_model.count_matrix

        # continue sparse or dense?
        if not self.sparse:
            # converting count matrices to arrays. As a result the
            # transition matrix and all subsequent properties will be
            # computed using dense arrays and dense matrix algebra.
            count_matrix = count_matrix.toarray()

        # restrict stationary distribution to active set
        if self.stationary_distribution_constraint is None:
            statdist_active = None
        else:
            statdist_active = self.stationary_distribution_constraint[count_model.state_symbols]
            statdist_active /= statdist_active.sum()  # renormalize

        opt_args = {}
        # TODO: non-rev estimate of msmtools does not comply with its own api...
        if statdist_active is None and self.reversible:
            opt_args['return_statdist'] = True

        # Estimate transition matrix
        P = msmest.transition_matrix(count_matrix, reversible=self.reversible,
                                     mu=statdist_active, maxiter=self.maxiter,
                                     maxerr=self.maxerr, **opt_args)
        # msmtools returns a tuple for statdist_active=None.
        if isinstance(P, tuple):
            P, statdist_active = P

        # create model
        self._model = MarkovStateModel(transition_matrix=P, pi=statdist_active, reversible=self.reversible,
                                       dt_model=count_model.physical_time * self.lagtime,
                                       count_model=count_model)

        return self
