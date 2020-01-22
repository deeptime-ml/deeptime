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
from typing import Optional

import numpy as np
from msmtools import estimation as msmest

from sktime.markovprocess._base import _MSMBaseEstimator
from sktime.markovprocess.markov_state_model import MarkovStateModel
from sktime.markovprocess.transition_counting import TransitionCountModel

__all__ = ['MaximumLikelihoodMSM']


class MaximumLikelihoodMSM(_MSMBaseEstimator):
    r"""Maximum likelihood estimator for MSMs given discrete trajectory statistics.

    References
    ----------
    .. [1] H. Wu and F. Noe: Variational approach for learning Markov processes from time series data
        (in preparation)

    """

    def __init__(self, reversible: bool = True, stationary_distribution_constraint: Optional[np.ndarray] = None,
                 sparse: bool = False, maxiter: int = int(1e6), maxerr: float = 1e-8):
        r"""
        Constructs a new maximum-likelihood msm estimator.

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
        maxiter : int, optional, default=1000000
            Optional parameter with reversible = True, sets the maximum number of iterations before the transition
            matrix estimation method exits.
        maxerr : float, optional, default = 1e-8
            Optional parameter with reversible = True. Convergence tolerance for transition matrix estimation. This
            specifies the maximum change of the Euclidean norm of relative stationary probabilities
            (:math:`x_i = \sum_k x_{ik}`). The relative stationary probability changes
            :math:`e_i = (x_i^{(1)} - x_i^{(2)})/(x_i^{(1)} + x_i^{(2)})` are used in order to track changes in small
            probabilities. The Euclidean norm of the change vector, :math:`|e_i|_2`, is compared to maxerr.
        """

        super(MaximumLikelihoodMSM, self).__init__(reversible=reversible, sparse=sparse)

        self.stationary_distribution_constraint = stationary_distribution_constraint
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
        if value is not None and (np.any(value < 0) or np.any(value > 1)):
            raise ValueError("not a distribution, contained negative entries and/or entries > 1.")
        if value is not None and np.sum(value) != 1.0:
            # re-normalize if not already normalized
            value = np.copy(value) / np.sum(value)
        self._stationary_distribution_constraint = value

    def fetch_model(self) -> Optional[MarkovStateModel]:
        r"""
        Yields the most recent markov state model that was estimated. Can be None if fit was not called.

        Returns
        -------
        The most recent markov state model or None
        """
        return self._model

    def fit(self, data, y=None, **kw):
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
        self._model = MarkovStateModel(transition_matrix=P, stationary_distribution=statdist_active,
                                       reversible=self.reversible, count_model=count_model)

        return self
