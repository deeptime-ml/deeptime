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
from scipy.sparse import issparse

from .markov_state_model import MarkovStateModel
from .._base import _MSMBaseEstimator
from ..transition_counting import TransitionCountModel

__all__ = ['MaximumLikelihoodMSM']


class MaximumLikelihoodMSM(_MSMBaseEstimator):
    r"""Maximum likelihood estimator for MSMs (:class:`MarkovStateModel <sktime.markov.msm.MarkovStateModel>`)
    given discrete trajectory statistics.

    Implementation according to [1]_.

    References
    ----------
    .. [1] Wu, Hao, and Frank Noé. "Variational approach for learning Markov processes from time series data."
           Journal of Nonlinear Science 30.1 (2020): 23-66.
    """

    def __init__(self, reversible: bool = True, stationary_distribution_constraint: Optional[np.ndarray] = None,
                 sparse: bool = False, allow_disconnected: bool = False, maxiter: int = int(1e6), maxerr: float = 1e-8):
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
        """

        super(MaximumLikelihoodMSM, self).__init__(reversible=reversible, sparse=sparse)

        self.stationary_distribution_constraint = stationary_distribution_constraint
        self.allow_disconnected = allow_disconnected
        self.maxiter = maxiter
        self.maxerr = maxerr

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

    def fetch_model(self) -> Optional[MarkovStateModel]:
        r"""Yields the most recent :class:`MarkovStateModel` that was estimated. Can be None if fit was not called.

        Returns
        -------
        model : MarkovStateModel or None
            The most recent markov state model or None.
        """
        return self._model

    def fit(self, data, *args, **kw):
        r""" Fits a new markov state model according to data.

        Parameters
        ----------
        data : TransitionCountModel or (n, n) ndarray
            input data, can either be :class:`TransitionCountModel <sktime.markov.TransitionCountModel>` or
            a 2-dimensional ndarray which is interpreted as count matrix.
        *args
            Dummy parameters for scikit-learn compatibility.
        **kw
            Dummy parameters for scikit-learn compatibility.

        Returns
        -------
        self : MaximumLikelihoodMSM
            Reference to self.

        See Also
        --------
        sktime.markov.TransitionCountModel : Transition count model
        sktime.markov.TransitionCountEstimator : Estimating transition count models from data
        """
        from .. import _transition_matrix as tmat
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

        # create model
        self._model = MarkovStateModel(transition_matrix=P, stationary_distribution=statdist,
                                       reversible=self.reversible, count_model=count_model)

        return self
