# This file is part of PyEMMA, SciKit-Time
#
# Copyright (c) 2019, 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

# .. moduleauthor:: F. Noe <frank DOT noe AT fu-berlin DOT de>
# .. moduleauthor:: B. Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

from typing import Optional, List
from math import ceil

import msmtools.analysis as msmana
import numpy as np
from scipy.sparse import issparse

from sktime.base import Model
from sktime.markovprocess.pcca import pcca, PCCAModel
from sktime.markovprocess.reactive_flux import ReactiveFlux
from sktime.markovprocess.sample import ensure_dtraj_list, compute_index_states
from sktime.markovprocess.transition_counting import TransitionCountModel
from sktime.numeric import mdot
from sktime.util import ensure_ndarray


class MarkovStateModel(Model):
    r"""Markov model with a given transition matrix

    Parameters
    ----------
    transition_matrix : ndarray(n,n)
        transition matrix
    stationary_distribution : ndarray(n), optional, default=None
        stationary distribution. Can be optionally given in case if it was
        already computed, e.g. by the estimator.
    reversible : bool, optional, default=None
        whether P is reversible with respect to its stationary distribution.
        If None (default), will be determined from P
    n_eigenvalues : int or None
        The number of eigenvalues / eigenvectors to be kept. If set to None,
        defaults will be used. For a dense MarkovStateModel the default is all eigenvalues.
        For a sparse MarkovStateModel the default is 10.
    ncv : int, optional, default=None
        Relevant for eigenvalue decomposition of reversible transition
        matrices. It is the number of Lanczos vectors generated, `ncv` must
        be greater than n_eigenvalues; it is recommended that ncv > 2*neig.
    count_model : TransitionCountModel, optional, default=None
        Transition count model containing count matrix and potentially data statistics. Not required for instantiation,
        default is None.
    """

    def __init__(self, transition_matrix, stationary_distribution=None, reversible=None, n_eigenvalues=None, ncv=None,
                 count_model=None):
        self._sparse = issparse(transition_matrix)
        self._is_reversible = reversible
        self._ncv = ncv

        if transition_matrix is None:
            raise ValueError("Markov state model requires a transition matrix, but it was None.")
        else:
            if not msmana.is_transition_matrix(transition_matrix, tol=1e-8):
                raise ValueError('The input transition matrix was not a stochastic matrix '
                                 '(elements >= 0, rows sum up to 1).')
            self._transition_matrix = transition_matrix

        self._n_states = np.shape(transition_matrix)[0]
        if self._is_reversible is None:
            self._is_reversible = msmana.is_reversible(self.transition_matrix)

        if stationary_distribution is None:
            from msmtools.analysis import stationary_distribution as compute_sd
            stationary_distribution = compute_sd(self.transition_matrix)
        if not np.allclose(np.sum(stationary_distribution), 1., atol=1e-14):
            raise ValueError("Stationary distribution did not sum up to 1 "
                             "(sum={})".format(np.sum(stationary_distribution)))
        self._stationary_distribution = stationary_distribution

        if n_eigenvalues is None:
            if self.is_sparse:
                # expect large matrix, don't take full state space but just (magic) the dominant 10
                n_eigenvalues = min(10, self.n_states - 1)
            else:
                # if we are dense take everything
                n_eigenvalues = self.n_states
        self._n_eigenvalues = n_eigenvalues
        self._count_model = count_model
        # initially None, compute lazily
        self._eigenvalues = None

    ################################################################################
    # Basic attributes
    ################################################################################

    @property
    def count_model(self) -> Optional[TransitionCountModel]:
        return self._count_model

    @property
    def lagtime(self) -> int:
        if self.count_model is not None:
            return self.count_model.lagtime
        return 1

    @property
    def transition_matrix(self):
        """ The transition matrix on the active set. """
        return self._transition_matrix

    @property
    def is_reversible(self) -> bool:
        """Returns whether the MarkovStateModel is reversible """
        return self._is_reversible

    @property
    def is_sparse(self) -> bool:
        """Returns whether the MarkovStateModel is sparse """
        return self._sparse

    @property
    def n_states(self) -> int:
        """ Number of active states on which all computations and estimations are done """
        return self._n_states

    @property
    def n_eigenvalues(self) -> int:
        """ number of eigenvalues to compute. """
        return self._n_eigenvalues

    @property
    def ncv(self):
        """ Number of Lanczos vectors used when computing the partial eigenvalue decomposition """
        return self._ncv

    ################################################################################
    # Spectral quantities
    ################################################################################

    @property
    def stationary_distribution(self):
        """The stationary distribution on the MarkovStateModel states"""
        return self._stationary_distribution

    def _compute_eigenvalues(self, neig):
        """ Conducts the eigenvalue decomposition and stores k eigenvalues """
        from msmtools.analysis import eigenvalues as anaeig

        if self.is_reversible:
            self._eigenvalues = anaeig(self.transition_matrix, k=neig, ncv=self._ncv,
                                       reversible=True, mu=self.stationary_distribution)
        else:
            self._eigenvalues = anaeig(self.transition_matrix, k=neig, ncv=self._ncv, reversible=False)

        if np.all(self._eigenvalues.imag == 0):
            self._eigenvalues = self._eigenvalues.real

    def _ensure_eigenvalues(self, neig=None):
        """ Ensures that at least neig eigenvalues have been computed """
        if neig is None:
            neig = self.n_eigenvalues
        # ensure that eigenvalue decomposition with k components is done.
        try:
            m = len(self._eigenvalues)  # this will raise and exception if self._eigenvalues doesn't exist yet.
            if m < neig:
                # not enough eigenpairs present - recompute:
                self._compute_eigenvalues(neig)
        except (AttributeError, TypeError) as e:
            # no eigendecomposition yet - compute:
            self._compute_eigenvalues(neig)

    def _compute_eigendecomposition(self, n_eigenvalues: int):
        r"""
        Conducts the eigenvalue decomposition and stores k eigenvalues, left and right eigenvectors.

        Parameters
        ----------
        n_eigenvalues: int
            number of eigenvalues to compute

        Returns
        -------
        A 3-tuple consisting of the normalized right eigenvectors, a diagonal matrix with the eigenvalues, and
        the normalized left eigenvectors.

        """
        from msmtools.analysis import rdl_decomposition

        R, D, L = rdl_decomposition(self.transition_matrix, k=n_eigenvalues,
                                    norm='standard' if not self.is_reversible else 'reversible',
                                    ncv=self._ncv)
        if self.is_reversible:
            # everything must be real-valued
            R = R.real
            D = D.real
            L = L.real
        else:
            # if the imaginary parts are zero, discard them.
            if np.all(R.imag == 0):
                R = np.real(R)
            if np.all(D.imag == 0):
                D = np.real(D)
            if np.all(L.imag == 0):
                L = np.real(L)

        return R, D, L

    def _ensure_eigendecomposition(self, n_eigenvalues: Optional[int] = None):
        r"""
        Ensures that eigendecomposition has been performed with at least n_eigenvalues eigenpairs.
        If not, performs eigendecomposition.

        Parameters
        ----------
        n_eigenvalues : int, optional, default=None
            Number of eigenpairs required. Defaults to n_eigenvalues, see :func:`MarkovStateModel.n_eigenvalues`.
        """
        if n_eigenvalues is None:
            n_eigenvalues = self.n_eigenvalues
        # ensure that eigenvalue decomposition with k components is done.
        try:
            # raises Attribute error if this is called for the first time
            m = self._D.shape[0]
            # compute if not enough eigenpairs were computed
            compute = m < n_eigenvalues
        except AttributeError:
            compute = True
        if compute:
            self._R, self._D, self._L = self._compute_eigendecomposition(n_eigenvalues)
            self._eigenvalues = np.diag(self._D)

    def eigenvalues(self, k=None):
        r"""Compute or fetch the transition matrix eigenvalues.

        Parameters
        ----------
        k : int
            number of eigenvalues to be returned. By default will return all
            available eigenvalues

        Returns
        -------
        ts : ndarray(k,)
            transition matrix eigenvalues :math:`\lambda_i, i = 1, ..., k`.,
            sorted by descending norm.

        """
        self._ensure_eigenvalues(neig=k)
        return self._eigenvalues[:k]

    def eigenvectors_left(self, k=None):
        r"""Compute the left transition matrix eigenvectors

        Parameters
        ----------
        k : int
            number of eigenvectors to be returned. By default all available
            eigenvectors.

        Returns
        -------
        L : ndarray(k,n)
            left eigenvectors in a row matrix. l_ij is the j'th component of
            the i'th left eigenvector

        """
        self._ensure_eigendecomposition(n_eigenvalues=k)
        return self._L[:k, :]

    def eigenvectors_right(self, k=None):
        r"""Compute the right transition matrix eigenvectors

        Parameters
        ----------
        k : int
            number of eigenvectors to be computed. By default all available
            eigenvectors.

        Returns
        -------
        R : ndarray(n,k)
            right eigenvectors in a column matrix. r_ij is the i'th component
            of the j'th right eigenvector

        """
        self._ensure_eigendecomposition(n_eigenvalues=k)
        return self._R[:, :k]

    def timescales(self, k=None):
        r"""
        The relaxation timescales corresponding to the eigenvalues

        Parameters
        ----------
        k : int
            number of timescales to be returned. By default all available
            eigenvalues, minus 1.

        Returns
        -------
        ts : ndarray(m)
            relaxation timescales in units of the input trajectory time step,
            defined by :math:`-\tau / ln | \lambda_i |, i = 2,...,k+1`.

        """
        if k is None:
            self._ensure_eigenvalues()
        else:
            self._ensure_eigenvalues(neig=k + 1)
        from msmtools.analysis.dense.decomposition import timescales_from_eigenvalues as timescales

        ts = timescales(self._eigenvalues, tau=self.lagtime)
        if k is None:
            return ts[1:]
        else:
            return ts[1:k + 1]  # exclude the stationary process

    def propagate(self, p0, k: int):
        r""" Propagates the initial distribution p0 k times

        Computes the product

        .. math::

            p_k = p_0^T P^k

        If the lag time of transition matrix :math:`P` is :math:`\tau`, this
        will provide the probability distribution at time :math:`k \tau`.

        Parameters
        ----------
        p0 : ndarray(n,)
            Initial distribution. Vector of size of the active set.

        k : int
            Number of time steps

        Returns
        ----------
        pk : ndarray(n,)
            Distribution after k steps. Vector of size of the active set.

        """
        p0 = ensure_ndarray(p0, ndim=1, size=self.n_states)
        assert k >= 0, 'k must be a non-negative integer'

        if k == 0:  # simply return p0 normalized
            return p0 / p0.sum()

        # sparse: we most likely don't have a full eigenvalue set, so just propagate
        if self.is_sparse:
            pk = np.array(p0)
            for i in range(k):
                pk = pk.T.dot(self.transition_matrix)
        else:  # dense: employ eigenvalue decomposition
            self._ensure_eigendecomposition(self.n_states)
            pk = mdot(p0.T,
                      self.eigenvectors_right(),
                      np.diag(np.power(self.eigenvalues(), k)),
                      self.eigenvectors_left()).real
        # normalize to 1.0 and return
        return pk / pk.sum()

    ################################################################################
    # Hitting problems
    ################################################################################

    def _assert_in_active(self, A):
        """
        Checks if set A is within the active set

        Parameters
        ----------
        A : int or int array
            set of states
        """
        if np.max(A) > self._n_states:
            raise ValueError('Chosen set contains states that are not included in the active set.')

    def mfpt(self, A, B):
        """Mean first passage times from set A to set B, in units of the input trajectory time step

        Parameters
        ----------
        A : int or int array
            set of starting states
        B : int or int array
            set of target states
        """
        from msmtools.analysis import mfpt
        self._assert_in_active(A)
        self._assert_in_active(B)
        return mfpt(self.transition_matrix, B, origin=A, mu=self.stationary_distribution) * self.lagtime

    def committor_forward(self, A, B):
        """Forward committor (also known as p_fold or splitting probability) from set A to set B

        Parameters
        ----------
        A : int or int array
            set of starting states
        B : int or int array
            set of target states
        """
        from msmtools.analysis import committor
        self._assert_in_active(A)
        self._assert_in_active(B)
        return committor(self.transition_matrix, A, B, forward=True)

    def committor_backward(self, A, B):
        """Backward committor from set A to set B

        Parameters
        ----------
        A : int or int array
            set of starting states
        B : int or int array
            set of target states
        """
        self._assert_in_active(A)
        self._assert_in_active(B)
        from msmtools.analysis import committor
        return committor(self.transition_matrix, A, B, forward=False, mu=self.stationary_distribution)

    def expectation(self, a: np.ndarray):
        r"""Equilibrium expectation value of a given observable.

        Parameters
        ----------
        a : (n,) ndarray
            Observable vector on the MarkovStateModel state space

        Returns
        -------
        val: float
            Equilibrium expectation value fo the given observable

        Notes
        -----
        The equilibrium expectation value of an observable :math:`a` is defined as follows

        .. math::

            \mathbb{E}_{\mu}[a] = \sum_i \pi_i a_i

        :math:`\pi=(\pi_i)` is the stationary vector of the transition matrix :math:`P`.

        """
        a = ensure_ndarray(a, ndim=1, size=self.n_states)
        return np.dot(a, self.stationary_distribution)

    def correlation(self, a, b=None, maxtime=None, k=None, ncv=None):
        r"""Time-correlation for equilibrium experiment.

        In order to simulate a time-correlation experiment (e.g. fluorescence
        correlation spectroscopy [NDD11]_, dynamical neutron scattering [LYP13]_,
        ...), first compute the mean values of your experimental observable :math:`a`
        by MarkovStateModel state:

        .. math::

            a_i = \frac{1}{N_i} \sum_{x_t \in S_i} f(x_t)

        where :math:`S_i` is the set of configurations belonging to MarkovStateModel state
        :math:`i` and :math:`f()` is a function that computes the experimental
        observable of interest for configuration :math:`x_t`. If a cross-correlation
        function is wanted, also apply the above computation to a second
        experimental observable :math:`b`.

        Then the accurate (i.e. without statistical error) autocorrelation
        function of :math:`f(x_t)` given the Markov model is computed by
        correlation(a), and the accurate cross-correlation function is computed
        by correlation(a,b). This is done by evaluating the equation

        .. math::

            acf_a(k\tau)     &= \mathbf{a}^\top \mathrm{diag}(\boldsymbol{\pi}) \mathbf{P(\tau)}^k \mathbf{a} \\
            ccf_{a,b}(k\tau) &= \mathbf{a}^\top \mathrm{diag}(\boldsymbol{\pi}) \mathbf{P(\tau)}^k \mathbf{b}

        where :math:`acf` stands for autocorrelation function and :math:`ccf`
        stands for cross-correlation function, :math:`\mathbf{P(\tau)}` is the
        transition matrix at lag time :math:`\tau`, :math:`\boldsymbol{\pi}` is the
        equilibrium distribution of :math:`\mathbf{P}`, and :math:`k` is the time index.

        Note that instead of using this method you could generate a long
        synthetic trajectory from the MarkovStateModel and then estimating the
        time-correlation of your observable(s) directly from this trajectory.
        However, there is no reason to do this because the present method
        does that calculation without any sampling, and only in the limit of
        an infinitely long synthetic trajectory the two results will agree
        exactly. The correlation function computed by the present method still
        has statistical uncertainty from the fact that the underlying MarkovStateModel
        transition matrix has statistical uncertainty when being estimated from
        data, but there is no additional (and unnecessary) uncertainty due to
        synthetic trajectory generation.

        Parameters
        ----------
        a : (n,) ndarray
            Observable, represented as vector on state space
        maxtime : int or float
            Maximum time (in units of the input trajectory time step) until
            which the correlation function will be evaluated.
            Internally, the correlation function can only be computed in
            integer multiples of the Markov model lag time, and therefore
            the actual last time point will be computed at :math:`\mathrm{ceil}(\mathrm{maxtime} / \tau)`
            By default (None), the maxtime will be set equal to the 5 times
            the slowest relaxation time of the MarkovStateModel, because after this time
            the signal is almost constant.
        b : (n,) ndarray (optional)
            Second observable, for cross-correlations
        k : int (optional)
            Number of eigenvalues and eigenvectors to use for computation.
            This option is only relevant for sparse matrices and long times
            for which an eigenvalue decomposition will be done instead of
            using the matrix power.
        ncv : int (optional)
            Only relevant for sparse matrices and large lag times where the
            relaxation will be computed using an eigenvalue decomposition.
            The number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k.

        Returns
        -------
        times : ndarray (N)
            Time points (in units of the input trajectory time step) at which
            the correlation has been computed
        correlations : ndarray (N)
            Correlation values at given times

        Examples
        --------

        This example computes the autocorrelation function of a simple observable
        on a three-state Markov model and plots the result using matplotlib:

        >>> import numpy as np
        >>> import sktime.markovprocess.markov_state_model as msm
        >>>
        >>> P = np.array([[0.99, 0.01, 0], [0.01, 0.9, 0.09], [0, 0.1, 0.9]])
        >>> a = np.array([0.0, 0.5, 1.0])
        >>> M = msm.MarkovStateModel(P)
        >>> times, acf = M.correlation(a)
        >>>
        >>> import matplotlib.pylab as plt # doctest: +SKIP
        >>> plt.plot(times, acf)  # doctest: +SKIP

        References
        ----------
        .. [NDD11] Noe, F., S. Doose, I. Daidone, M. Loellmann, J. D. Chodera, M. Sauer and J. C. Smith. 2011
            Dynamical fingerprints for probing individual relaxation processes in biomolecular dynamics with simulations
            and kinetic experiments. Proc. Natl. Acad. Sci. USA 108, 4822-4827.
        .. [LYP13] Lindner, B., Z. Yi, J.-H. Prinz, J. C. Smith and F. Noe. 2013.
            Dynamic Neutron Scattering from Conformational Dynamics I: Theory and Markov models.
            J. Chem. Phys. 139, 175101.

        """
        # input checking is done in low-level API
        # compute number of tau steps
        if maxtime is None:
            # by default, use five times the longest relaxation time, because then we have relaxed to equilibrium.
            maxtime = 5 * self.timescales()[0]
        steps = np.arange(int(ceil(float(maxtime) / self.lagtime)))
        # compute correlation
        from msmtools.analysis import correlation
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        res = correlation(self.transition_matrix, a, obs2=b, times=steps, k=k, ncv=ncv)
        # return times scaled by tau
        times = steps * self.lagtime
        return times, res

    def fingerprint_correlation(self, a, b=None, k=None, ncv=None):
        r"""Dynamical fingerprint for equilibrium time-correlation experiment.

        Parameters
        ----------
        a : (n,) ndarray
            Observable, represented as vector on MarkovStateModel state space
        b : (n,) ndarray, optional
            Second observable, for cross-correlations
        k : int, optional
            Number of eigenvalues and eigenvectors to use for computation. This
            option is only relevant for sparse matrices and long times for which
            an eigenvalue decomposition will be done instead of using the matrix
            power
        ncv : int, optional
            Only relevant for sparse matrices and large lag times, where the
            relaxation will be computed using an eigenvalue decomposition.
            The number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k

        Returns
        -------
        timescales : (k,) ndarray
            Time-scales (in units of the input trajectory time step) of the transition matrix
        amplitudes : (k,) ndarray
            Amplitudes for the correlation experiment

        References
        ----------
        Spectral densities are commonly used in spectroscopy. Dynamical
        fingerprints are a useful representation for computational
        spectroscopy results and have been introduced in [NDD11]_.

        .. [NDD11] Noe, F, S Doose, I Daidone, M Loellmann, M Sauer, J D
            Chodera and J Smith. 2010. Dynamical fingerprints for probing
            individual relaxation processes in biomolecular dynamics with
            simulations and kinetic experiments. PNAS 108 (12): 4822-4827.

        """
        # input checking is done in low-level API
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        from msmtools.analysis import fingerprint_correlation as fc
        return fc(self.transition_matrix, a, obs2=b, tau=self.lagtime, k=k, ncv=ncv)

    def relaxation(self, p0, a, maxtime=None, k=None, ncv=None):
        r"""Simulates a perturbation-relaxation experiment.

        In perturbation-relaxation experiments such as temperature-jump, pH-jump, pressure jump or rapid mixing
        experiments, an ensemble of molecules is initially prepared in an off-equilibrium distribution and the
        expectation value of some experimental observable is then followed over time as the ensemble relaxes towards
        equilibrium.

        In order to simulate such an experiment, first determine the distribution of states at which the experiment is
        started, :math:`p_0` and compute the mean values of your experimental observable :math:`a`
        by MarkovStateModel state:

        .. math::

            a_i = \frac{1}{N_i} \sum_{x_t \in S_i} f(x_t)

        where :math:`S_i` is the set of configurations belonging to MarkovStateModel state :math:`i`
        and :math:`f()` is a function that computes the experimental observable of
        interest for configuration :math:`x_t`.

        Then the accurate (i.e. without statistical error) time-dependent expectation value of :math:`f(x_t)` given the
        Markov model is computed by relaxation(p0, a). This is done by evaluating the equation

        .. math::

            E_a(k\tau) = \mathbf{p_0}^{\top} \mathbf{P(\tau)}^k \mathbf{a}

        where :math:`E` stands for the expectation value that relaxes to its equilibrium value that is identical
        to expectation(a), :math:`\mathbf{P(\tau)}` is the transition matrix at lag time :math:`\tau`,
        :math:`\boldsymbol{\pi}` is the equilibrium distribution of :math:`\mathbf{P}`, and :math:`k` is the time index.

        Note that instead of using this method you could generate many synthetic trajectory from the MarkovStateModel
        with starting points drawn from the initial distribution and then estimating the
        time-dependent expectation value by an ensemble average. However, there is no reason to do this because the
        present method does that calculation without any sampling, and only in the limit of an infinitely many
        trajectories the two results will agree exactly. The relaxation function computed by the present method still
        has statistical uncertainty from the fact that the underlying MarkovStateModel transition
        matrix has statistical uncertainty when being estimated from data, but there is no additional (and unnecessary)
        uncertainty due to synthetic trajectory generation.

        Parameters
        ----------
        p0 : (n,) ndarray
            Initial distribution for a relaxation experiment
        a : (n,) ndarray
            Observable, represented as vector on state space
        maxtime : int or float, optional
            Maximum time (in units of the input trajectory time step) until which the correlation function will be
            evaluated. Internally, the correlation function can only be computed in integer multiples of the
            Markov model lag time, and therefore the actual last time point will be computed at
            :math:`\mathrm{ceil}(\mathrm{maxtime} / \tau)`.
            By default (None), the maxtime will be set equal to the 5 times the slowest relaxation time of the MarkovStateModel,
            because after this time the signal is constant.
        k : int (optional)
            Number of eigenvalues and eigenvectors to use for computation
        ncv : int (optional)
            Only relevant for sparse matrices and large lag times, where the relaxation will be computes using an
            eigenvalue decomposition.
            The number of Lanczos vectors generated, `ncv` must be greater than k; it is recommended that ncv > 2*k

        Returns
        -------
        times : ndarray (N)
            Time points (in units of the input trajectory time step) at which the relaxation has been computed
        res : ndarray
            Array of expectation value at given times

        """
        # input checking is done in low-level API
        # compute number of tau steps
        if maxtime is None:
            # by default, use five times the longest relaxation time, because then we have relaxed to equilibrium.
            maxtime = 5 * self.timescales()[0]
        kmax = int(ceil(float(maxtime) / self.lagtime))
        steps = np.array(list(range(kmax)), dtype=int)
        # compute relaxation function
        from msmtools.analysis import relaxation
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        res = relaxation(self.transition_matrix, p0, a, times=steps, k=k, ncv=ncv)
        # return times scaled by tau
        times = steps * self.lagtime
        return times, res

    def fingerprint_relaxation(self, p0, a, k=None, ncv=None):
        r"""Dynamical fingerprint for perturbation/relaxation experiment.

        Parameters
        ----------
        p0 : (n,) ndarray
            Initial distribution for a relaxation experiment
        a : (n,) ndarray
            Observable, represented as vector on state space
        k : int, optional
            Number of eigenvalues and eigenvectors to use for computation
        ncv : int, optional
            Only relevant for sparse matrices and large lag times, where the relaxation will be computes using an
            eigenvalue decomposition. The number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k

        Returns
        -------
        timescales : (k,) ndarray
            Time-scales (in units of the input trajectory time step) of the transition matrix
        amplitudes : (k,) ndarray
            Amplitudes for the relaxation experiment

        References
        ----------
        Spectral densities are commonly used in spectroscopy. Dynamical
        fingerprints are a useful representation for computational
        spectroscopy results and have been introduced in [NDD11]_.

        .. [NDD11] Noe, F, S Doose, I Daidone, M Loellmann, M Sauer, J D
            Chodera and J Smith. 2010. Dynamical fingerprints for probing
            individual relaxation processes in biomolecular dynamics with
            simulations and kinetic experiments. PNAS 108 (12): 4822-4827.

        """
        # input checking is done in low-level API
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        from msmtools.analysis import fingerprint_relaxation as fr
        return fr(self.transition_matrix, p0, a, tau=self.lagtime, k=k, ncv=ncv)

    def pcca(self, n_metastable_sets: int) -> PCCAModel:
        r""" Runs PCCA+ [1]_ to compute a metastable decomposition of MarkovStateModel states

        After calling this method you can access :func:`metastable_memberships`,
        :func:`metastable_distributions`, :func:`metastable_sets` and
        :func:`metastable_assignments`.

        Parameters
        ----------
        n_metastable_sets : int
            Number of metastable sets

        Returns
        -------
        pcca_obj : :class:`PCCAModel <sktime.markovprocess.PCCAModel>`
            An object containing all PCCA+ quantities.

        Notes
        -----
        If you coarse grain with PCCA+, the order of the obtained memberships
        might not be preserved.

        References
        ----------
        .. [1] Roeblitz, S and M Weber. 2013. Fuzzy spectral clustering by
            PCCA+: application to Markov state models and data
            classification. Advances in Data Analysis and Classification 7
            (2): 147-179
        """
        if not self.is_reversible:
            raise ValueError('Cannot compute PCCA+ for non-reversible matrices. '
                             'Set reversible=True when constructing the MarkovStateModel.')
        return pcca(self.transition_matrix, n_metastable_sets)

    def reactive_flux(self, A, B) -> ReactiveFlux:
        r""" A->B reactive flux from transition path theory (TPT)

        The returned :class:`ReactiveFlux <pyemma.msm.models.ReactiveFlux>` object
        can be used to extract various quantities of the flux, as well as to
        compute A -> B transition pathways, their weights, and to coarse-grain
        the flux onto sets of states.

        Parameters
        ----------
        A : array_like
            List of integer state labels for set A
        B : array_like
            List of integer state labels for set B

        Returns
        -------
        tptobj : :class:`ReactiveFlux <sktime.markovprocess.ReactiveFlux>` object
            An object containing the reactive A->B flux network
            and several additional quantities, such as the stationary probability,
            committors and set definitions.

        See also
        --------
        :class:`ReactiveFlux <sktime.markovprocess.ReactiveFlux>`
            Reactive Flux model
        """
        from msmtools.flux import flux_matrix, to_netflux
        import msmtools.analysis as msmana
        from sktime.util import ensure_ndarray
        from sktime.markovprocess import ReactiveFlux

        T = self.transition_matrix
        mu = self.stationary_distribution
        A = ensure_ndarray(A, dtype=int)
        B = ensure_ndarray(B, dtype=int)

        if len(A) == 0 or len(B) == 0:
            raise ValueError('set A or B is empty')
        n = T.shape[0]
        if len(A) > n or len(B) > n or max(A) > n or max(B) > n:
            raise ValueError('set A or B defines more states, than given transition matrix.')

        # forward committor
        qplus = msmana.committor(T, A, B, forward=True)
        # backward committor
        if msmana.is_reversible(T, mu=mu):
            qminus = 1.0 - qplus
        else:
            qminus = msmana.committor(T, A, B, forward=False, mu=mu)
        # gross flux
        grossflux = flux_matrix(T, mu, qminus, qplus, netflux=False)
        # net flux
        netflux = to_netflux(grossflux)

        # construct flux object
        return ReactiveFlux(A, B, netflux, mu=mu, qminus=qminus, qplus=qplus, gross_flux=grossflux,
                            physical_time=self.count_model.physical_time if self.count_model is not None else '1 step')

    def simulate(self, N, start=None, stop=None, dt=1):
        """
        Generates a realization of the Markov Model

        Parameters
        ----------
        N : int
            trajectory length in steps of the lag time
        start : int, optional, default = None
            starting hidden state. If not given, will sample from the stationary
            distribution of the hidden transition matrix.
        stop : int or int-array-like, optional, default = None
            stopping hidden set. If given, the trajectory will be stopped before
            N steps once a hidden state of the stop set is reached
        dt : int
            trajectory will be saved every dt time steps.
            Internally, the dt'th power of P is taken to ensure a more efficient simulation.


        Returns
        -------
        (N/dt,) ndarray
            The state trajectory with length N/dt
        """
        # todo replace with faster implementation in sktime.markovprocess.generation
        import msmtools.generation as msmgen
        return msmgen.generate_traj(self.transition_matrix, N, start=start, stop=stop, dt=dt)

    ################################################################################
    # For general statistics
    ################################################################################
    def compute_trajectory_weights(self, dtrajs):
        r"""Uses the MarkovStateModel to assign a probability weight to each trajectory frame.

        This is a powerful function for the calculation of arbitrary observables in the trajectories one has
        started the analysis with. The stationary probability of the MarkovStateModel will be used to reweigh all states.
        Returns a list of weight arrays, one for each trajectory, and with a number of elements equal to
        trajectory frames. Given :math:`N` trajectories of lengths :math:`T_1` to :math:`T_N`, this function
        returns corresponding weights:

        .. math::

            (w_{1,1}, ..., w_{1,T_1}), (w_{N,1}, ..., w_{N,T_N})

        that are normalized to one:

        .. math::

            \sum_{i=1}^N \sum_{t=1}^{T_i} w_{i,t} = 1

        Suppose you are interested in computing the expectation value of a function :math:`a(x)`, where :math:`x`
        are your input configurations. Use this function to compute the weights of all input configurations and
        obtain the estimated expectation by:

        .. math::

            \langle a \rangle = \sum_{i=1}^N \sum_{t=1}^{T_i} w_{i,t} a(x_{i,t})

        Or if you are interested in computing the time-lagged correlation between functions :math:`a(x)` and
        :math:`b(x)` you could do:

        .. math::

            \langle a(t) b(t+\tau) \rangle_t = \sum_{i=1}^N \sum_{t=1}^{T_i} w_{i,t} a(x_{i,t}) a(x_{i,t+\tau})


        Returns
        -------
        weights : list of ndarray
            The normalized trajectory weights. Given :math:`N` trajectories of lengths :math:`T_1` to :math:`T_N`,
            returns the corresponding weights:

            .. math::

                (w_{1,1}, ..., w_{1,T_1}), (w_{N,1}, ..., w_{N,T_N})

        """
        # compute stationary distribution, expanded to full set
        if self.count_model is None:
            raise RuntimeError("Count model was None but needs to be provided in this case.")
        dtrajs = ensure_dtraj_list(dtrajs)
        statdist_full = np.zeros(self.count_model.n_states_full)
        statdist_full[self.count_model.state_symbols] = self.stationary_distribution
        # histogram observed states
        from msmtools.dtraj import count_states
        hist = 1.0 * count_states(dtrajs)
        # simply read off stationary distribution and accumulate total weight
        W = []
        wtot = 0.0
        for dtraj in dtrajs:
            w = statdist_full[dtraj] / hist[dtraj]
            W.append(w)
            wtot += np.sum(w)
        # normalize
        for w in W:
            w /= wtot
        # done
        return W

    def compute_state_indices(self, dtrajs) -> List[np.ndarray]:
        r"""Generates a trajectory/time indices for the given list of states. If a count model is provided in this
        MSM and it does not represent the full state space, the discrete trajectories are first mapped to the
        active state space, inactive states are mapped to -1.

        Parameters
        ----------
        dtrajs : array_like or list of array_like
            discretized trajectories

        Returns
        -------
        A list of arrays with trajectory/time indices for the provided discretized trajectories
        """
        if self.count_model is not None:
            dtrajs = self.count_model.transform_discrete_trajectories_to_submodel(dtrajs)
        return compute_index_states(dtrajs)

    ################################################################################
    # HMM-based coarse graining
    ################################################################################

    def hmm(self, dtrajs, nhidden: int, return_estimator=False):
        """Estimates a hidden Markov state model as described in [1]_

        Parameters
        ----------
        dtrajs: list of int-array like
            discrete trajectories to use for estimation of the HMM.

        nhidden : int
            number of hidden (metastable) states

        return_estimator : boolean, optional
            if False only the Model is returned,
            if True both the Estimator and the Model is returned.

        Returns
        -------
        hmsm : :class:`MaximumLikelihoodHMSM`

        References
        ----------
        .. [1] F. Noe, H. Wu, J.-H. Prinz and N. Plattner:
            Projected and hidden Markov models for calculating kinetics and metastable states of complex molecules
            J. Chem. Phys. 139, 184114 (2013)

        """
        # check if the time-scale separation is OK
        # if hmm.n_states = msm.n_states there is no problem. Otherwise, check spectral gap
        if self.n_states > nhidden:
            ts = self.timescales()
            timescale_ratios = ts[:-1] / ts[1:]
            if timescale_ratios[nhidden - 2] < 1.5:
                import warnings
                warnings.warn('Requested coarse-grained model with {nhidden} metastable states at lag={lag}.'
                              ' The ratio of relaxation timescales between'
                              ' {nhidden} and {nhidden_1} states is only {ratio}'
                              ' while we recommend at least 1.5.'
                              ' It is possible that the resulting HMM is inaccurate. Handle with caution.'.format(
                    lag=self.lagtime,
                    nhidden=nhidden,
                    nhidden_1=nhidden + 1,
                    ratio=timescale_ratios[nhidden - 2],
                ), stacklevel=2)
        # run HMM estimate
        from sktime.markovprocess.maximum_likelihood_hmsm import MaximumLikelihoodHMSM
        estimator = MaximumLikelihoodHMSM(lagtime=self.lagtime, n_states=nhidden, msm_init=self,
                                          reversible=self.is_reversible, dt_traj=self.count_model.physical_time)
        estimator.fit(dtrajs)
        model = estimator.fetch_model()
        if return_estimator:
            return estimator, model
        return model

    def score(self, dtrajs, score_method='VAMP2', score_k=10):
        r""" Scores the MSM using the dtrajs using the variational approach for Markov processes [1]_ [2]_

        Currently only implemented using dense matrices - will be slow for large state spaces.

        Parameters
        ----------
        dtrajs : list of arrays
            test data (discrete trajectories).
        score_method : str
            Overwrite scoring method if desired. If `None`, the estimators scoring
            method will be used. See __init__ for documentation.
        score_k : int or None
            Overwrite scoring rank if desired. If `None`, the estimators scoring
            rank will be used. See __init__ for documentation.
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
            from time series data (in preparation)
        .. [3] McGibbon, R and V. S. Pande: Variational cross-validation of slow
            dynamical modes in molecular kinetics, J. Chem. Phys. 142, 124105 (2015)
        .. [4] Noe, F. and C. Clementi: Kinetic distance and kinetic maps from molecular
            dynamics simulation. J. Chem. Theory Comput. 11, 5002-5011 (2015)

        """
        from sktime.markovprocess.sample import ensure_dtraj_list
        dtrajs = ensure_dtraj_list(dtrajs)  # ensure format
        if self.count_model is None:
            raise RuntimeError('This MarkovStateModel has not been estimated from data '
                               '(e.g. count_model unassigned). Cannot proceed.')

        # determine actual scoring rank
        if score_k is None:
            score_k = self.n_states
        if score_k > self.n_states:
            import warnings
            warnings.warn('Requested scoring rank {rank} exceeds number of MSM states. '
                          'Reduced to score_k = {n_states}'.format(rank=score_k, n_states=self.n_states))
            score_k = self.n_states  # limit to n_states

        # training data
        K = self.transition_matrix  # model
        C0t_train = self.count_model.count_matrix
        from scipy.sparse import issparse
        if issparse(K):  # can't deal with sparse right now.
            K = K.toarray()
        if issparse(C0t_train):  # can't deal with sparse right now.
            C0t_train = C0t_train.toarray()
        C00_train = np.diag(C0t_train.sum(axis=1))  # empirical cov
        Ctt_train = np.diag(C0t_train.sum(axis=0))  # empirical cov

        # test data
        from msmtools.estimation import count_matrix
        C0t_test_raw = count_matrix(dtrajs, self.count_model.lagtime, sparse_return=False)
        # map to present active set
        active_set = self.count_model.state_symbols
        map_from = active_set[np.where(active_set < C0t_test_raw.shape[0])[0]]
        map_to = np.arange(len(map_from))
        C0t_test = np.zeros((self.n_states, self.n_states))
        C0t_test[np.ix_(map_to, map_to)] = C0t_test_raw[np.ix_(map_from, map_from)]
        C00_test = np.diag(C0t_test.sum(axis=1))
        Ctt_test = np.diag(C0t_test.sum(axis=0))
        assert C00_train.shape == C00_test.shape
        assert C0t_train.shape == C0t_test.shape
        assert Ctt_train.shape == Ctt_test.shape

        # score
        from sktime.metrics import vamp_score
        return vamp_score(K, C00_train, C0t_train, Ctt_train, C00_test, C0t_test, Ctt_test,
                          k=score_k, score=score_method)
