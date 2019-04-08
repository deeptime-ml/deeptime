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

import copy
from math import ceil

import numpy as np

from sktime.base import Model
from sktime.markovprocess import Q_
from sktime.markovprocess.pcca import PCCA


class MarkovStateModel(Model):
    r"""Markov model with a given transition matrix

    Parameters
    ----------
    P : ndarray(n,n)
        transition matrix

    pi : ndarray(n), optional, default=None
        stationary distribution. Can be optionally given in case if it was
        already computed, e.g. by the estimator.

    reversible : bool, optional, default=None
        whether P is reversible with respect to its stationary distribution.
        If None (default), will be determined from P

    dt_model : str, optional, default='1 step'
        Description of the physical time corresponding to one time step of the
        MarkovStateModel (aka lag time). May be used by analysis algorithms such as plotting
        tools to pretty-print the axes.
        By default '1 step', i.e. there is no physical time unit. Specify by a
        number, whitespace and unit. Permitted units are
        (* is an arbitrary string):

        *  'fs',  'femtosecond*'
        *  'ps',  'picosecond*'
        *  'ns',  'nanosecond*'
        *  'us',  'microsecond*'
        *  'ms',  'millisecond*'
        *  's',   'second*'

    neig : int or None
        The number of eigenvalues / eigenvectors to be kept. If set to None,
        defaults will be used. For a dense MarkovStateModel the default is all eigenvalues.
        For a sparse MarkovStateModel the default is 10.

    ncv : int (optional)
        Relevant for eigenvalue decomposition of reversible transition
        matrices. ncv is the number of Lanczos vectors generated, `ncv` must
        be greater than k; it is recommended that ncv > 2*k.

    """
    def __init__(self, P, pi=None, reversible=None, dt_model='1 step', neig=None, ncv=None):

        self.ncv = ncv
        # we set reversible first, so it can be derived from P, if None was given.
        self.reversible = reversible
        self.transition_matrix = P
        # pi might be derived from P, if None was given.
        self.stationary_distribution = pi
        self.dt_model = dt_model
        #self.neig = neig

    def __eq__(self, other):
        if not isinstance(other, MarkovStateModel):
            return False
        if id(self) == id(other):
            return True
        if self.transition_matrix is not None and other.transition_matrix is not None:
            if self.transition_matrix.shape != other.transition_matrix.shape:
                P_equal = False
            else:
                P_equal = np.allclose(self.transition_matrix, other.transition_matrix)
        else:
            P_equal = True
        return (P_equal and
                self.sparse == other.sparse and self.neig == other.neig and
                self.reversible == other.reversible and
                self.timestep_model == other.timestep_model)

    ################################################################################
    # Basic attributes
    ################################################################################

    @property
    def transition_matrix(self):
        """ The transition matrix on the active set. """
        return self._P

    @transition_matrix.setter
    def transition_matrix(self, value):
        self._P = value
        import msmtools.analysis as msmana
        # check input
        if self._P is not None:
            if not msmana.is_transition_matrix(self._P, tol=1e-8):
                raise ValueError('T is not a transition matrix.')
            # set states
            self.nstates = np.shape(self._P)[0]
            if self.reversible is None:
                self.reversible = msmana.is_reversible(self._P)

            from scipy.sparse import issparse
            self.sparse = issparse(self._P)
        else:
            # set dummy values for not yet known attributes.
            self.nstates = 0
            self.sparse = False

        # TODO: if spectral decomp etc. already has been computed, reset its state.

    @property
    def reversible(self):
        """Returns whether the MarkovStateModel is reversible """
        return self._reversible

    @reversible.setter
    def reversible(self, value):
        self._reversible = value

    @property
    def sparse(self):
        """Returns whether the MarkovStateModel is sparse """
        return self._sparse

    @sparse.setter
    def sparse(self, value: bool):
        self._sparse = value

    @property
    def timestep_model(self):
        """Physical time corresponding to one transition matrix step, e.g. '10 ps'"""
        return str(self._dt_model)

    @property
    def nstates(self):
        """ Number of active states on which all computations and estimations are done """
        return self._nstates

    @nstates.setter
    def nstates(self, n):
        self._nstates = n

    @property
    def neig(self):
        """ number of eigenvalues to compute. """
        return self._neig

    @neig.setter
    def neig(self, value):
        # set or correct eig param
        if value is None:
            if self.transition_matrix is not None:
                if self.sparse:
                    value = 10
                else:
                    value = self._nstates

        # set ncv for consistency
        if not hasattr(self, 'ncv'):
            self.ncv = None

        self._neig = value

    @property
    def dt_model(self) -> Q_:
        """Description of the physical time corresponding to the lag."""
        return self._dt_model

    @dt_model.setter
    def dt_model(self, value: str):
        self._dt_model = Q_(value)

    ################################################################################
    # Spectral quantities
    ################################################################################

    @property
    def stationary_distribution(self):
        """The stationary distribution on the MarkovStateModel states"""
        return self._pi

    @stationary_distribution.setter
    def stationary_distribution(self, value):
        if value is None and self.transition_matrix is not None:
            from msmtools.analysis import stationary_distribution as _statdist
            value = _statdist(self.transition_matrix)
        elif value is not None:
            # check sum is one
            np.testing.assert_allclose(np.sum(value), 1, atol=1e-14)
        self._pi = value

    def _compute_eigenvalues(self, neig):
        """ Conducts the eigenvalue decomposition and stores k eigenvalues """
        from msmtools.analysis import eigenvalues as anaeig

        if self.reversible:
            self._eigenvalues = anaeig(self.transition_matrix, k=neig, ncv=self.ncv,
                                       reversible=True, mu=self.stationary_distribution)
        else:
            self._eigenvalues = anaeig(self.transition_matrix, k=neig, ncv=self.ncv, reversible=False)

        if np.all(self._eigenvalues.imag == 0):
            self._eigenvalues = self._eigenvalues.real

    def _ensure_eigenvalues(self, neig=None):
        """ Ensures that at least neig eigenvalues have been computed """
        if neig is None:
            neig = self.neig
        # ensure that eigenvalue decomposition with k components is done.
        try:
            m = len(self._eigenvalues)  # this will raise and exception if self._eigenvalues doesn't exist yet.
            if m < neig:
                # not enough eigenpairs present - recompute:
                self._compute_eigenvalues(neig)
        except AttributeError:
            # no eigendecomposition yet - compute:
            self._compute_eigenvalues(neig)

    def _compute_eigendecomposition(self, neig):
        """ Conducts the eigenvalue decomposition and stores k eigenvalues, left and right eigenvectors """
        from msmtools.analysis import rdl_decomposition

        if self.reversible:
            self._R, self._D, self._L = rdl_decomposition(self.transition_matrix, norm='reversible',
                                                          k=neig, ncv=self.ncv)
            # everything must be real-valued
            self._R = self._R.real
            self._D = self._D.real
            self._L = self._L.real
        else:
            self._R, self._D, self._L = rdl_decomposition(self.transition_matrix, k=neig, norm='standard', ncv=self.ncv)
            # if the imaginary parts are zero, discard them.
            if np.all(self._R.imag == 0):
                self._R = np.real(self._R)
            if np.all(self._D.imag == 0):
                self._D = np.real(self._D)
            if np.all(self._L.imag == 0):
                self._L = np.real(self._L)

        self._eigenvalues = np.diag(self._D)

    def _ensure_eigendecomposition(self, neig=None):
        """Ensures that eigendecomposition has been performed with at least neig eigenpairs

        neig : int
            number of eigenpairs needed. If not given the default value will
            be used - see __init__()

        """
        if neig is None:
            neig = self.neig
        # ensure that eigenvalue decomposition with k components is done.
        try:
            m = self._D.shape[0]  # this will raise and exception if self._D doesn't exist yet.
        except AttributeError:
            # no eigendecomposition yet - compute:
            self._compute_eigendecomposition(neig)

    def eigenvalues(self, k=None):
        r"""Compute the transition matrix eigenvalues

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
        self._ensure_eigendecomposition(neig=k)
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
        self._ensure_eigendecomposition(neig=k)
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

        ts = timescales(self._eigenvalues, tau=self._dt_model)
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
        p0 = _types.ensure_ndarray(p0, ndim=1, size=self.nstates, kind='numeric')
        assert k >= 0, 'k must be a non-negative integer'

        if k == 0:  # simply return p0 normalized
            return p0 / p0.sum()

        if self.sparse:  # sparse: we don't have a full eigenvalue set, so just propagate
            pk = np.array(p0)
            for i in range(k):
                pk = np.dot(pk.T, self.transition_matrix)
        else:  # dense: employ eigenvalue decomposition
            self._ensure_eigendecomposition(self.nstates)
            from pyemma.util.linalg import mdot
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
        if np.max(A) > self._nstates:
            raise ValueError('Chosen set contains states that are not included in the active set.')

    def _mfpt(self, P, A, B, mu=None):
        self._assert_in_active(A)
        self._assert_in_active(B)
        from msmtools.analysis import mfpt
        # scale mfpt by lag time
        return self._dt_model * mfpt(P, B, origin=A, mu=mu)

    def mfpt(self, A, B):
        """Mean first passage times from set A to set B, in units of the input trajectory time step

        Parameters
        ----------
        A : int or int array
            set of starting states
        B : int or int array
            set of target states
        """
        return self._mfpt(self.transition_matrix, A, B, mu=self.stationary_distribution)

    def _committor_forward(self, P, A, B):
        self._assert_in_active(A)
        self._assert_in_active(B)
        from msmtools.analysis import committor
        return committor(P, A, B, forward=True)

    def committor_forward(self, A, B):
        """Forward committor (also known as p_fold or splitting probability) from set A to set B

        Parameters
        ----------
        A : int or int array
            set of starting states
        B : int or int array
            set of target states
        """
        return self._committor_forward(self.transition_matrix, A, B)

    def _committor_backward(self, P, A, B, mu=None):
        self._assert_in_active(A)
        self._assert_in_active(B)
        from msmtools.analysis import committor
        return committor(P, A, B, forward=False, mu=mu)

    def committor_backward(self, A, B):
        """Backward committor from set A to set B

        Parameters
        ----------
        A : int or int array
            set of starting states
        B : int or int array
            set of target states
        """
        return self._committor_backward(self.transition_matrix, A, B, mu=self.stationary_distribution)

    def expectation(self, a):
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
        # check input and go
        a = _types.ensure_ndarray(a, ndim=1, size=self.nstates, kind='numeric')
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
        >>> import pyemma.msm as msm
        >>>
        >>> P = np.array([[0.99, 0.01, 0], [0.01, 0.9, 0.09], [0, 0.1, 0.9]])
        >>> a = np.array([0.0, 0.5, 1.0])
        >>> M = msm.markov_model(P)
        >>> times, acf = M.correlation(a)
        >>>
        >>> import matplotlib.pylab as plt
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
        steps = np.arange(int(ceil(float(maxtime) / self._dt_model)))
        # compute correlation
        from msmtools.analysis import correlation
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        res = correlation(self.transition_matrix, a, obs2=b, times=steps, k=k, ncv=ncv)
        # return times scaled by tau
        times = self._dt_model * steps
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
        return fc(self.transition_matrix, a, obs2=b, tau=self._dt_model, k=k, ncv=ncv)

    def relaxation(self, p0, a, maxtime=None, k=None, ncv=None):
        r"""Simulates a perturbation-relaxation experiment.

        In perturbation-relaxation experiments such as temperature-jump, pH-jump, pressure jump or rapid mixing
        experiments, an ensemble of molecules is initially prepared in an off-equilibrium distribution and the
        expectation value of some experimental observable is then followed over time as the ensemble relaxes towards
        equilibrium.

        In order to simulate such an experiment, first determine the distribution of states at which the experiment is
        started, :math:`p_0` and compute the mean values of your experimental observable :math:`a` by MarkovStateModel state:

        .. math::

            a_i = \frac{1}{N_i} \sum_{x_t \in S_i} f(x_t)

        where :math:`S_i` is the set of configurations belonging to MarkovStateModel state :math:`i` and :math:`f()` is a function
        that computes the experimental observable of interest for configuration :math:`x_t`.

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
        has statistical uncertainty from the fact that the underlying MarkovStateModel transition matrix has statistical uncertainty
        when being estimated from data, but there is no additional (and unnecessary) uncertainty due to synthetic
        trajectory generation.

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
        kmax = int(ceil(float(maxtime) / self._dt_model))
        steps = np.array(list(range(kmax)), dtype=int)
        # compute relaxation function
        from msmtools.analysis import relaxation
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        res = relaxation(self.transition_matrix, p0, a, times=steps, k=k, ncv=ncv)
        # return times scaled by tau
        times = self._dt_model * steps
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
        return fr(self.transition_matrix, p0, a, tau=self._dt_model, k=k, ncv=ncv)

    ################################################################################
    # pcca
    ################################################################################

    def _assert_metastable(self):
        """ Tests if pcca object is available, or else raises a ValueError.

        """
        try:
            if not self._metastable_computed:
                raise ValueError('Metastable decomposition has not yet been computed. Please call pcca(m) first.')
        except AttributeError:
            raise ValueError('Metastable decomposition has not yet been computed. Please call pcca(m) first.')

    def pcca(self, m: int) -> PCCA:
        r""" Runs PCCA++ [1]_ to compute a metastable decomposition of MarkovStateModel states

        After calling this method you can access :func:`metastable_memberships`,
        :func:`metastable_distributions`, :func:`metastable_sets` and
        :func:`metastable_assignments`.

        Parameters
        ----------
        m : int
            Number of metastable sets

        Returns
        -------
        pcca_obj : :class:`PCCA <pyemma.msm.PCCA>`
            An object containing all PCCA quantities. However, you can also
            ignore this return value and instead retrieve the quantities of
            your interest with the following MarkovStateModel functions: :func:`metastable_memberships`,
            :func:`metastable_distributions`, :func:`metastable_sets` and :func:`metastable_assignments`.

        Notes
        -----
        If you coarse grain with PCCA++, the order of the obtained memberships
        might not be preserved. This also applies for :func:`metastable_memberships`,
        :func:`metastable_distributions`, :func:`metastable_sets`, :func:`metastable_assignments`

        References
        ----------
        .. [1] Roeblitz, S and M Weber. 2013. Fuzzy spectral clustering by
            PCCA+: application to Markov state models and data
            classification. Advances in Data Analysis and Classification 7
            (2): 147-179

        """
        # can we do it?
        if not self.reversible:
            raise ValueError('Cannot compute PCCA for non-reversible matrices. '
                             'Set reversible=True when constructing the MarkovStateModel.')

        # ensure that we have a pcca object with the right number of states
        try:
            # this will except if we don't have a pcca object
            if self._pcca.n_metastable != m or self._pcca.P is not self.transition_matrix:
                # incorrect number of states or new transition matrix -> recompute
                self._pcca = PCCA(self.transition_matrix, m)
        except AttributeError:
            # didn't have a pcca object yet - compute
            self._pcca = PCCA(self.transition_matrix, m)

        # set metastable properties
        self._metastable_computed = True
        self._n_metastable = self._pcca.n_metastable
        self._metastable_memberships = copy.deepcopy(self._pcca.memberships)
        self._metastable_distributions = copy.deepcopy(self._pcca.output_probabilities)
        self._metastable_sets = copy.deepcopy(self._pcca.metastable_sets)
        self._metastable_assignments = copy.deepcopy(self._pcca.metastable_assignment)

        return self._pcca

    @property
    def n_metastable(self):
        """ Number of states chosen for PCCA++ computation.
        """
        # are we ready?
        self._assert_metastable()
        return self._n_metastable

    @property
    def metastable_memberships(self):
        r""" Probabilities of MarkovStateModel states to belong to a metastable state by PCCA++

        Computes the memberships of active set states to metastable sets with
        the PCCA++ method [1]_.

        :func:`pcca` needs to be called first to make this attribute available.

        Returns
        -------
        M : ndarray((n,m))
            A matrix containing the probability or membership of each state to be
            assigned to each metastable set, i.e. p(metastable | state).
            The row sums of M are 1.

        See also
        --------
        pcca
            to compute the metastable decomposition

        References
        ----------
        .. [1] Roeblitz, S and M Weber. 2013. Fuzzy spectral clustering by
            PCCA+: application to Markov state models and data
            classification. Advances in Data Analysis and Classification 7
            (2): 147-179

        """
        # are we ready?
        self._assert_metastable()
        return self._metastable_memberships

    @property
    def metastable_distributions(self):
        r""" Probability of metastable states to visit an MarkovStateModel state by PCCA++

        Computes the probability distributions of active set states within
        each metastable set by combining the the PCCA++ method [1]_ with
        Bayesian inversion as described in [2]_.

        :func:`pcca` needs to be called first to make this attribute available.

        Returns
        -------
        p_out : ndarray (m,n)
            A matrix containing the probability distribution of each active set
            state, given that we are in one of the m metastable sets,
            i.e. p(state | metastable). The row sums of p_out are 1.

        See also
        --------
        pcca
            to compute the metastable decomposition

        References
        ----------
        .. [1] Roeblitz, S and M Weber. 2013. Fuzzy spectral clustering by
            PCCA+: application to Markov state models and data classification.
            Advances in Data Analysis and Classification 7, 147-179.
        .. [2] F. Noe, H. Wu, J.-H. Prinz and N. Plattner. 2013.
            Projected and hidden Markov models for calculating kinetics and
            metastable states of complex molecules J. Chem. Phys. 139, 184114.

        """
        # are we ready?
        self._assert_metastable()
        return self._metastable_distributions

    @property
    def metastable_sets(self):
        """ Metastable sets using PCCA++

        Computes the metastable sets of active set states within each
        metastable set using the PCCA++ method [1]_. :func:`pcca` needs
        to be called first to make this attribute available.

        This is only recommended for visualization purposes. You *cannot*
        compute any actual quantity of the coarse-grained kinetics without
        employing the fuzzy memberships!

        Returns
        -------
        sets : list of ndarray
            A list of length equal to metastable states. Each element is an
            array with microstate indexes contained in it

        See also
        --------
        pcca
            to compute the metastable decomposition

        References
        ----------
        .. [1] Roeblitz, S and M Weber. 2013. Fuzzy spectral clustering by
            PCCA+: application to Markov state models and data
            classification. Advances in Data Analysis and Classification 7
            (2): 147-179

        """
        # are we ready?
        self._assert_metastable()
        return self._metastable_sets

    @property
    def metastable_assignments(self):
        """ Assignment of states to metastable sets using PCCA++

        Computes the assignment to metastable sets for active set states using
        the PCCA++ method [1]_. :func:`pcca` needs to be called first to make
        this attribute available.

        This is only recommended for visualization purposes. You *cannot* compute
        any actual quantity of the coarse-grained kinetics without employing the
        fuzzy memberships!

        Returns
        -------
        assignments : ndarray (n,)
            For each MarkovStateModel state, the metastable state it is located in.

        See also
        --------
        pcca
            to compute the metastable decomposition

        References
        ----------
        .. [1] Roeblitz, S and M Weber. 2013. Fuzzy spectral clustering by
            PCCA+: application to Markov state models and data
            classification. Advances in Data Analysis and Classification 7
            (2): 147-179

        """
        # are we ready?
        self._assert_metastable()
        return self._metastable_assignments

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
        htraj: (N/dt, ) ndarray
            The state trajectory with length N/dt
        """
        import msmtools.generation as msmgen
        return msmgen.generate_traj(self.transition_matrix, N, start=start, stop=stop, dt=dt)

    def generate_traj(self, N, start=None, stop=None, stride=1):
        """Generates a synthetic discrete trajectory of length N and simulation time stride * lag time * N

        This information can be used
        in order to generate a synthetic molecular dynamics trajectory - see
        :func:`pyemma.coordinates.save_traj`

        Note that the time different between two samples is the Markov model lag time tau. When comparing
        quantities computing from this synthetic trajectory and from the input trajectories, the time points of this
        trajectory must be scaled by the lag time in order to have them on the same time scale.

        Parameters
        ----------
        N : int
            Number of time steps in the output trajectory. The total simulation time is stride * lag time * N
        start : int, optional, default = None
            starting state. If not given, will sample from the stationary distribution of P
        stop : int or int-array-like, optional, default = None
            stopping set. If given, the trajectory will be stopped before N steps
            once a state of the stop set is reached
        stride : int, optional, default = 1
            Multiple of lag time used as a time step. By default, the time step is equal to the lag time

        Returns
        -------
        indexes : ndarray( (N, 2) )
            trajectory and time indexes of the simulated trajectory. Each row consist of a tuple (i, t), where i is
            the index of the trajectory and t is the time index within the trajectory.
            Note that the time different between two samples is the Markov model lag time tau

        See also
        --------
        pyemma.coordinates.save_traj
            in order to save this synthetic trajectory as a trajectory file with molecular structures

        """
        # TODO: this is the only function left which does something time-related in a multiple of tau rather than dt.
        # TODO: we could generate dt-strided trajectories by sampling tau times from the current state, but that would
        # TODO: probably lead to a weird-looking trajectory. Maybe we could use a HMM to generate intermediate 'hidden'
        # TODO: frames. Anyway, this is a nontrivial issue.
        # generate synthetic states
        from msmtools.generation import generate_traj as _generate_traj

        syntraj = _generate_traj(self.transition_matrix, N, start=start, stop=stop, dt=stride)
        # result
        return sample_indexes_by_sequence(self.active_state_indexes, syntraj)

    def sample_by_state(self, nsample, subset=None, replace=True):
        """Generates samples of the connected states.

        For each state in the active set of states, generates nsample samples with trajectory/time indexes.
        This information can be used in order to generate a trajectory of length nsample * nconnected using
        :func:`pyemma.coordinates.save_traj` or nconnected trajectories of length nsample each using
        :func:`pyemma.coordinates.save_traj`

        Parameters
        ----------
        nsample : int
            Number of samples per state. If replace = False, the number of returned samples per state could be smaller
            if less than nsample indexes are available for a state.
        subset : ndarray((n)), optional, default = None
            array of states to be indexed. By default all states in the connected set will be used
        replace : boolean, optional
            Whether the sample is with or without replacement

        Returns
        -------
        indexes : list of ndarray( (N, 2) )
            list of trajectory/time index arrays with an array for each state.
            Within each index array, each row consist of a tuple (i, t), where i is
            the index of the trajectory and t is the time index within the trajectory.

        See also
        --------
        pyemma.coordinates.save_traj
            in order to save the sampled frames sequentially in a trajectory file with molecular structures
        pyemma.coordinates.save_trajs
            in order to save the sampled frames in nconnected trajectory files with molecular structures

        """
        # generate connected state indexes
        return sample_indexes_by_state(self.active_state_indexes, nsample, subset=subset, replace=replace)

    # TODO: add sample_metastable() for sampling from metastable (pcca or hmm) states.
    def sample_by_distributions(self, distributions, nsample):
        """Generates samples according to given probability distributions

        Parameters
        ----------
        distributions : list or array of ndarray ( (n) )
            m distributions over states. Each distribution must be of length n and must sum up to 1.0
        nsample : int
            Number of samples per distribution. If replace = False, the number of returned samples per state could be
            smaller if less than nsample indexes are available for a state.

        Returns
        -------
        indexes : length m list of ndarray( (nsample, 2) )
            List of the sampled indices by distribution.
            Each element is an index array with a number of rows equal to nsample, with rows consisting of a
            tuple (i, t), where i is the index of the trajectory and t is the time index within the trajectory.

        """
        # generate connected state indexes
        return sample_indexes_by_distribution(self.active_state_indexes, distributions, nsample)

    ################################################################################
    # For general statistics
    ################################################################################
    def trajectory_weights(self, dtrajs):
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
        statdist_full = np.zeros(self.nstates_full)
        statdist_full[self.active_set] = self.stationary_distribution
        # histogram observed states
        import msmtools.dtraj as msmtraj
        hist = 1.0 * msmtraj.count_states(dtrajs)
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

    ################################################################################
    # HMM-based coarse graining
    ################################################################################

    def hmm(self, dtrajs, nhidden: int):
        """Estimates a hidden Markov state model as described in [1]_

        Parameters
        ----------
        nhidden : int
            number of hidden (metastable) states

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
        # if hmm.nstates = msm.nstates there is no problem. Otherwise, check spectral gap
        if self.nstates > nhidden:
            timescale_ratios = self.timescales()[:-1] / self.timescales()[1:]
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
                ))
        # run HMM estimate
        #from pyemma.msm.estimators.maximum_likelihood_hmsm import MaximumLikelihoodHMSM
        estimator = MaximumLikelihoodHMSM(lag=self.lagtime, nstates=nhidden, msm_init=self,
                                          reversible=self.reversible, dt_traj=self.dt_traj)
        estimator.fit(dtrajs)
        return estimator.fetch_model()

    # TODO: redundant
    def coarse_grain(self, dtrajs, ncoarse: int, method='hmm') -> Model:
        r"""Returns a coarse-grained Markov model.

        Currently only the HMM method described in [1]_ is available for coarse-graining MSMs.

        Parameters
        ----------
        ncoarse : int
            number of coarse states

        Returns
        -------
        hmsm : :class:`MaximumLikelihoodHMSM`

        References
        ----------
        .. [1] F. Noe, H. Wu, J.-H. Prinz and N. Plattner:
            Projected and hidden Markov models for calculating kinetics and metastable states of complex molecules
            J. Chem. Phys. 139, 184114 (2013)

        """
        # check input
        assert 1 < ncoarse <= self.nstates, 'nstates must be an int in [2,msmobj.nstates]'

        return self.hmm(dtrajs, ncoarse)
