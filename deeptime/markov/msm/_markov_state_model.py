# .. moduleauthor:: F. Noe <frank DOT noe AT fu-berlin DOT de>
# .. moduleauthor:: B. Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

from math import ceil
from typing import Optional, List, Union

import numpy as np
import scipy
from scipy.sparse import issparse

from ...base import Model
from ...covariance import CovarianceModel
from ...decomposition import CovarianceKoopmanModel, vamp_score
from .._pcca import pcca, PCCAModel
from .._reactive_flux import ReactiveFlux
from ..sample import ensure_dtraj_list, compute_index_states
from ..tools import analysis as msmana
from .._transition_counting import TransitionCountModel
from .._util import count_states
from ...numeric import is_square_matrix, spd_inv_sqrt
from ...util.decorators import cached_property
from ...util.matrix import submatrix
from ...util.types import ensure_array


class MarkovStateModel(Model):
    r"""Markov model with a given transition matrix.

    Parameters
    ----------
    transition_matrix : (n,n) array_like
        The transition matrix.
    stationary_distribution : ndarray(n), optional, default=None
        Stationary distribution. Can be optionally given in case if it was already computed.
    reversible : bool, optional, default=None
        Whether the transition matrix is reversible with respect to its stationary distribution. If None (default),
        will be determined from the transition matrix.
    n_eigenvalues : int, optional, default=None
        The number of eigenvalues / eigenvectors to be kept. If set to None, it depends on the transition matrix.
        If it is densely stored (in terms of a numpy array), all eigenvectors and eigenvalues are computed. If it is
        sparse, only the 10 largest eigenvalues with corresponding eigenvectors are computed.
    ncv : int optional, default=None
        Relevant for eigenvalue decomposition of reversible transition matrices. It is the number of Lanczos
        vectors generated, `ncv` must be greater than n_eigenvalues; it is recommended that ncv > 2*neig.
    count_model : TransitionCountModel, optional, default=None
        In case the MSM was estimated from data, the transition count model can be provided for statistical
        information about the data. Some properties of the model require a count model so that they can be computed.
    transition_matrix_tolerance : float, default=1e-8
        The tolerance under which a matrix is still considered a transition matrix (only non-negative elements and
        row sums of 1).

    See Also
    --------
    MaximumLikelihoodMSM : maximum-likelihood estimator for MSMs
    OOMReweightedMSM : estimator for MSMs which uses Koopman reweighting
    BayesianMSM : bayesian sampling of MSMs to obtain uncertainties

    References
    ----------
    .. bibliography:: /references.bib
        :style: unsrt
        :filter: docname in docnames
        :keyprefix: msm-
    """

    def __init__(self, transition_matrix, stationary_distribution=None, reversible=None,
                 n_eigenvalues=None, ncv=None, count_model=None, transition_matrix_tolerance=1e-8):
        super().__init__()
        self._is_reversible = reversible
        self._ncv = ncv
        self._transition_matrix_tolerance = transition_matrix_tolerance

        self.update_transition_matrix(transition_matrix)

        if n_eigenvalues is None:
            if self.sparse:
                # expect large matrix, don't take full state space but just (magic) the dominant 10
                n_eigenvalues = min(10, self.n_states - 1)
            else:
                # if we are dense take everything
                n_eigenvalues = self.n_states
        self._n_eigenvalues = n_eigenvalues
        self._count_model = count_model
        self._eigenvalues = None

        self._stationary_distribution = None
        self.update_stationary_distribution(stationary_distribution)

    ################################################################################
    # Basic attributes
    ################################################################################

    def _invalidate_caches(self):
        r""" Invalidates all cached properties and causes them to be re-evaluated """
        for member in MarkovStateModel.__dict__.values():
            if isinstance(member, cached_property):
                member.invalidate()
        self._eigenvalues = None
        self._stationary_distribution = None

    @property
    def count_model(self) -> Optional[TransitionCountModel]:
        r"""
        Returns a transition count model, can be None. The transition count model statistics about data that was used
        for transition counting as well as a count matrix.

        Returns
        -------
        model : TransitionCountModel or None
            The transition count model or None.
        """
        return self._count_model

    @property
    def has_count_model(self) -> bool:
        r""" Yields whether this Markov state model has a count model.

        :type: bool
        """
        return self._count_model is not None

    @cached_property
    def koopman_model(self) -> CovarianceKoopmanModel:
        r"""Yields a :class:`CovarianceKoopmanModel` based on the transition matrix and stationary
         distribution of this Markov state model.

        Returns
        -------
        model : CovarianceKoopmanModel
            The model.

        Notes
        -----
        If :math:`P\in\mathbb{R}^{n\times n}` denotes the transition matrix and :math:`\mu\in\mathbb{R}^n` denotes
        the stationary distribution, we define covariance matrices

        .. math::
            \begin{aligned}
            C_{00} &= C_{11} = \text{diag} (\mu_1,\ldots,\mu_n)\\
            C_{01} &= C_{00}P
            \end{aligned}

        and based on these a Koopman operator :math:`K = C_{00}^{-1/2}C_{01}C_{11}^{-1/2}`.

        See Also
        --------
        to_koopman_model
        """
        return self.to_koopman_model(False)

    @cached_property
    def empirical_koopman_model(self) -> CovarianceKoopmanModel:
        r"""Yields a :class:`CovarianceKoopmanModel` based on the count matrix of this Markov state model.

        Returns
        -------
        model : CovarianceKoopmanModel
            The model.

        Notes
        -----
        If :math:`C\in\mathbb{R}^{n\times n}` denotes the count matrix and :math:`P` the transition matrix,
        we define covariance matrices based on transition count statistics

        .. math::

            \begin{aligned}
            C_{00} &= \text{diag} \left( \sum_i C_{i1}, \ldots, \sum_i C_{in} \right) \\
            C_{11} &= \text{diag} \left( \sum_i C_{1i}, \ldots, \sum_i C_{ni} \right) \\
            C_{01} &= C,
            \end{aligned}

        and reweight the operator :math:`P` to the empirical distribution via :math:`C_{01\text{, re}} = C_{00}P`.
        Based on these we define the Koopman operator :math:`K = C_{00}^{-1/2}C_{01\text{, re}}C_{11}^{-1/2}`.

        See Also
        --------
        to_koopman_model
        """
        return self.to_koopman_model(True)

    def to_koopman_model(self, empirical: bool = True, epsilon: float = 1e-10):
        r""" Computes the SVD of the symmetrized Koopman operator in the analytical or empirical distribution,
        returns as Koopman model.

        Parameters
        ----------
        empirical : bool, optional, default=True
            Determines whether the model should refer to the analytical distributions based on the transition matrix
            or the empirical distributions based on the count matrix.
        epsilon : float, optional, default=1e-10
            Regularization parameter for computing inverses of covariance matrices.

        Returns
        -------
        model : CovarianceKoopmanModel
            The model.
        """
        if empirical:
            assert self.has_count_model, "This requires statistics over the estimation data, " \
                                         "in particular a count model."
        return _svd_sym_koopman(self, empirical, epsilon=epsilon)

    @property
    def lagtime(self) -> int:
        r""" The lagtime this model was estimated at. In case no count model was provided, this property defaults
        to a lagtime of `1`.

        Returns
        -------
        lagtime : int
            The lagtime.
        """
        if self.count_model is not None:
            return self.count_model.lagtime
        return 1

    @property
    def transition_matrix(self):
        """ The transition matrix on the active set. """
        return self._transition_matrix

    @property
    def transition_matrix_tolerance(self):
        r""" The tolerance under which a matrix is considered a transition matrix. This means that all elements
        must be non-negative and the row sums must be 1. """
        return self._transition_matrix_tolerance

    def update_transition_matrix(self, value: np.ndarray):
        """ Sets the transition matrix and invalidates all cached and derived properties. """
        if value is None:
            raise ValueError("Markov state model requires a transition matrix, but it was None.")
        else:
            if not issparse(value):
                try:
                    value = np.asarray_chkfinite(value)
                except ValueError:
                    raise
            if not msmana.is_transition_matrix(value, tol=self.transition_matrix_tolerance):
                raise ValueError(f'The input transition matrix is not a stochastic matrix '
                                 f'(elements >= 0, rows sum up to 1) up to '
                                 f'tolerance {self._transition_matrix_tolerance}.')
            self._transition_matrix = value
            self._invalidate_caches()

    @cached_property
    def reversible(self) -> bool:
        """Returns whether the MarkovStateModel is reversible """
        return msmana.is_reversible(self.transition_matrix) if self._is_reversible is None else self._is_reversible

    @property
    def sparse(self) -> bool:
        """Returns whether the MarkovStateModel is sparse """
        return issparse(self.transition_matrix)

    @property
    def n_states(self) -> int:
        """ Number of active states on which all computations and estimations are done """
        return np.shape(self.transition_matrix)[0]

    @property
    def n_eigenvalues(self) -> int:
        """ number of eigenvalues to compute. """
        return self._n_eigenvalues

    @property
    def ncv(self):
        """ Number of Lanczos vectors used when computing the partial eigenvalue decomposition """
        return self._ncv

    def submodel(self, states):
        r"""
        Restricts this markov state model to a subset of states by taking a submatrix of the transition matrix
        and re-normalizing it, as well as restricting the stationary distribution and count model if given.

        Parameters
        ----------
        states : array_like of int
            states to restrict to
        Returns
        -------
        submodel : MarkovStateModel
            A onto the given states restricted MSM.
        """
        states = np.asarray(states)
        if np.any(states >= self.n_states):
            raise ValueError("At least one of the given states is not contained in this model "
                             "(n_states={}, max. given state={}).".format(self.n_states, np.max(states)))
        count_model = self.count_model
        if count_model is not None:
            count_model = count_model.submodel(states)
        transition_matrix = submatrix(self.transition_matrix, states)
        transition_matrix /= transition_matrix.sum(axis=1)[:, None]
        # set stationary distribution to None, gets recomputed in the constructor
        return MarkovStateModel(transition_matrix, stationary_distribution=None,
                                reversible=self.reversible, n_eigenvalues=min(self.n_eigenvalues, len(states)),
                                ncv=self.ncv, count_model=count_model)

    ################################################################################
    # Spectral quantities
    ################################################################################

    @cached_property
    def stationary(self):
        """ Whether the MSM is stationary, i.e. whether the initial distribution is the stationary distribution
         of the hidden transition matrix. """
        # for disconnected matrices, the stationary distribution depends on the estimator, so we can't compute
        # it directly. Therefore we test whether the initial distribution is stationary.
        return np.allclose(np.dot(self.stationary_distribution, self.transition_matrix), self.stationary_distribution)

    @cached_property
    def stationary_distribution(self):
        """The stationary distribution on the MarkovStateModel states"""
        if self._stationary_distribution is None:
            from deeptime.markov.tools.analysis import stationary_distribution as compute_sd
            stationary_distribution = compute_sd(self.transition_matrix)
            if not np.allclose(np.sum(stationary_distribution), 1., atol=1e-14):
                raise ValueError("Stationary distribution did not sum up to 1 "
                                 "(sum={})".format(np.sum(stationary_distribution)))
        else:
            stationary_distribution = self._stationary_distribution

        return stationary_distribution

    def update_stationary_distribution(self, value: Optional[np.ndarray]):
        r""" Explicitly sets the stationary distribution, re-normalizes """
        if value is not None:
            self._stationary_distribution = np.copy(value) / np.sum(value)
        else:
            self._stationary_distribution = None
        self._invalidate_caches()

    def _compute_eigenvalues(self, neig):
        """ Conducts the eigenvalue decomposition and stores k eigenvalues """
        from deeptime.markov.tools.analysis import eigenvalues as anaeig

        if self.reversible:
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
        from deeptime.markov.tools.analysis import rdl_decomposition

        R, D, L = rdl_decomposition(self.transition_matrix, k=n_eigenvalues,
                                    norm='standard' if not self.reversible else 'reversible',
                                    ncv=self._ncv)
        if self.reversible:
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
            number of eigenvectors to be returned. By default uses value of :attr:`n_eigenvalues`.

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
            number of eigenvectors to be computed. By default uses value of :attr:`n_eigenvalues`.

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
        from deeptime.markov.tools.analysis import timescales_from_eigenvalues as timescales

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
        p0 = ensure_array(p0, ndim=1, size=self.n_states)
        assert k >= 0, 'k must be a non-negative integer'

        if k == 0:  # simply return p0 normalized
            return p0 / p0.sum()

        # sparse: we most likely don't have a full eigenvalue set, so just propagate
        if self.sparse:
            pk = np.array(p0)
            for i in range(k):
                pk = self.transition_matrix.T.dot(pk)
        else:  # dense: employ eigenvalue decomposition
            self._ensure_eigendecomposition(self.n_states)
            pk = np.linalg.multi_dot([
                p0.T, self.eigenvectors_right(), np.diag(np.power(self.eigenvalues(), k)),
                self.eigenvectors_left()
            ]).real
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
        if np.max(A) > self.n_states:
            raise ValueError('Chosen set contains states that are not included in the active set.')

    def mfpt(self, A, B):
        """Mean first passage times from set A to set B, in units of the input trajectory time step.

        Parameters
        ----------
        A : int or int array
            set of starting states
        B : int or int array
            set of target states
        """
        from deeptime.markov.tools.analysis import mfpt
        self._assert_in_active(A)
        self._assert_in_active(B)
        return self.lagtime * mfpt(self.transition_matrix, B, origin=A, mu=self.stationary_distribution)

    def committor_forward(self, A, B):
        """Forward committor (also known as p_fold or splitting probability) from set A to set B.

        Parameters
        ----------
        A : int or int array
            set of starting states
        B : int or int array
            set of target states
        """
        from deeptime.markov.tools.analysis import committor
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
        from deeptime.markov.tools.analysis import committor
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
        a = ensure_array(a, ndim=1, size=self.n_states)
        return np.dot(a, self.stationary_distribution)

    def correlation(self, a, b=None, maxtime=None, k=None, ncv=None):
        r"""Time-correlation for equilibrium experiment.

        In order to simulate a time-correlation experiment (e.g. fluorescence
        correlation spectroscopy :cite:`msm-noe2011dynamical`, dynamical neutron
        scattering :cite:`msm-lindner2013dynamic`, ...), first compute the mean values of your experimental
        observable :math:`a` by MarkovStateModel state:

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

            \begin{aligned}
            acf_a(k\tau)     &= \mathbf{a}^\top \mathrm{diag}(\boldsymbol{\pi}) \mathbf{P(\tau)}^k \mathbf{a} \\
            ccf_{a,b}(k\tau) &= \mathbf{a}^\top \mathrm{diag}(\boldsymbol{\pi}) \mathbf{P(\tau)}^k \mathbf{b}
            \end{aligned}

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
        >>> import deeptime.markov.msm as msm
        >>>
        >>> P = np.array([[0.99, 0.01, 0], [0.01, 0.9, 0.09], [0, 0.1, 0.9]])
        >>> a = np.array([0.0, 0.5, 1.0])
        >>> M = msm.MarkovStateModel(P)
        >>> times, acf = M.correlation(a)
        >>>
        >>> import matplotlib.pylab as plt # doctest: +SKIP
        >>> plt.plot(times, acf)  # doctest: +SKIP
        """
        # input checking is done in low-level API
        # compute number of tau steps
        if maxtime is None:
            # by default, use five times the longest relaxation time, because then we have relaxed to equilibrium.
            maxtime = 5 * self.timescales()[0]
        steps = np.arange(int(ceil(float(maxtime) / self.lagtime)))
        # compute correlation
        from deeptime.markov.tools.analysis import correlation
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

        Spectral densities are commonly used in spectroscopy. Dynamical
        fingerprints are a useful representation for computational
        spectroscopy results and have been introduced in :cite:`msm-noe2011dynamical`.

        References
        ----------
        .. footbibliography::
        """
        # input checking is done in low-level API
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        from deeptime.markov.tools.analysis import fingerprint_correlation as fc
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
            By default (None), the maxtime will be set equal to the 5 times the slowest relaxation time of the
            MarkovStateModel, because after this time the signal is constant.
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
        from deeptime.markov.tools.analysis import relaxation
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        res = relaxation(self.transition_matrix, p0, a, times=steps, k=k)
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

        Spectral densities are commonly used in spectroscopy. Dynamical fingerprints are a useful representation
        for computational spectroscopy results and have been introduced in :cite:`msm-noe2011dynamical`.
        """
        # input checking is done in low-level API
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        from deeptime.markov.tools.analysis import fingerprint_relaxation as fr
        return fr(self.transition_matrix, p0, a, tau=self.lagtime, k=k, ncv=ncv)

    def pcca(self, n_metastable_sets: int) -> PCCAModel:
        r""" Runs PCCA+ :cite:`msm-roblitz2013fuzzy` to compute a metastable decomposition of
        MarkovStateModel states.

        After calling this method you can access :func:`metastable_memberships`,
        :func:`metastable_distributions`, :func:`metastable_sets` and
        :func:`metastable_assignments` of the returned object.

        Parameters
        ----------
        n_metastable_sets : int
            Number of metastable sets

        Returns
        -------
        pcca_obj : :class:`PCCAModel <deeptime.markov.PCCAModel>`
            An object containing all PCCA+ quantities.

        Notes
        -----
        If you coarse grain with PCCA+, the order of the obtained memberships
        might not be preserved.
        """
        if not self.reversible:
            raise ValueError('Cannot compute PCCA+ for non-reversible matrices. '
                             'Set reversible=True when constructing the MarkovStateModel.')
        return pcca(self.transition_matrix, n_metastable_sets, self.stationary_distribution)

    def reactive_flux(self, source_states, target_states) -> ReactiveFlux:
        r""" A->B reactive flux from transition path theory (TPT)

        The returned object can be used to extract various quantities of the flux, as well as to
        compute A -> B transition pathways, their weights, and to coarse-grain
        the flux onto sets of states.

        Parameters
        ----------
        source_states : array_like
            List of integer state labels for set A
        target_states : array_like
            List of integer state labels for set B

        Returns
        -------
        tptobj : ReactiveFlux
            An object containing the reactive A->B flux network
            and several additional quantities, such as the stationary probability,
            committors and set definitions.

        See also
        --------
        :class:`ReactiveFlux`
            Reactive Flux model
        """
        from .._reactive_flux import reactive_flux

        return reactive_flux(
            transition_matrix=self.transition_matrix,
            source_states=source_states,
            target_states=target_states,
            stationary_distribution=self.stationary_distribution,
            transition_matrix_tolerance=None  # set to None explicitly so no check is performed
        )

    def simulate(self, n_steps: int, start: Optional[int] = None, stop: Optional[Union[int, List[int]]] = None,
                 dt: int = 1, seed: int = -1):
        r"""Generates a realization of the Markov Model.

        Parameters
        ----------
        n_steps : int
            trajectory length in steps of the lag time
        start : int, optional, default = None
            starting state
        stop : int or int-array-like, optional, default = None
            stopping hidden set. If given, the trajectory will be stopped before
            N steps once a hidden state of the stop set is reached
        dt : int, default=1
            trajectory will be saved every dt time steps.
            Internally, the dt'th power of P is taken to ensure a more efficient simulation.
        seed : int, default=-1
            If non-negative, this fixes the seed used for sampling to the provided value so that
            results are reproducible.

        Returns
        -------
        (N/dt,) ndarray
            The state trajectory with length N/dt

        Examples
        --------
        >>> msm = MarkovStateModel(transition_matrix=np.array([[.7, .3], [.3, .7]]))
        >>> trajectory = msm.simulate(n_steps=15)
        >>> print(trajectory)  # doctest:+ELLIPSIS
        [...]
        """
        if seed is None:  # the extension code internally treats -1 as default initialization
            seed = -1
        from .._markov_bindings import simulation as sim
        if start is None:
            start = np.random.choice(self.n_states, p=self.stationary_distribution)
        if self.sparse:
            transition_matrix = self.transition_matrix.toarray()
        else:
            transition_matrix = self.transition_matrix
        if dt > 1:
            transition_matrix = np.linalg.matrix_power(transition_matrix, dt)
        if stop is not None and not isinstance(stop, (list, tuple, np.ndarray)):
            stop = [stop]
        return sim.trajectory(N=n_steps, start=start, P=transition_matrix, stop=stop, seed=seed)

    ################################################################################
    # For general statistics
    ################################################################################
    def compute_trajectory_weights(self, dtrajs):
        r"""Uses the MarkovStateModel to assign a probability weight to each trajectory frame.

        This is a powerful function for the calculation of arbitrary observables in the trajectories one has
        started the analysis with. The stationary probability of the MarkovStateModel will be used to reweigh all
        states. Returns a list of weight arrays, one for each trajectory, and with a number of elements equal to
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
            Discretized trajectories.

        Returns
        -------
        state_indices : List of ndarray
            A list of arrays with trajectory/time indices for the provided discretized trajectories.
        """
        if self.count_model is not None:
            dtrajs = self.count_model.transform_discrete_trajectories_to_submodel(dtrajs)
        if all(np.all(dtraj == -1) for dtraj in dtrajs):
            raise ValueError("None of the symbols in the discrete trajectories were represented in the "
                             "Markov state model as states.")
        return compute_index_states(dtrajs)

    ################################################################################
    # HMM-based coarse graining
    ################################################################################

    def hmm(self, dtrajs, nhidden: int, return_estimator=False):
        """Estimates a hidden Markov state model as described in :cite:`msm-noe2013projected`.

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
        hmm : deeptime.markov.hmm.HiddenMarkovModel
            A hidden markov model.
        """
        if not self.reversible:
            raise ValueError("Can only use HMM coarse-graining if the estimate was reversible. This is due to the use "
                             "of PCCA+ as initial estimate of the hidden transition matrix.")
        if self.sparse:
            raise ValueError("Not implemented for sparse transition matrices.")
        # check if the time-scale separation is OK
        # if hmm.n_states = msm.n_states there is no problem. Otherwise, check spectral gap
        if self.n_states > nhidden:
            ts = self.timescales()
            timescale_ratios = ts[:-1] / ts[1:]
            if timescale_ratios[nhidden - 2] < 1.5:
                import warnings
                warnings.warn(f'Requested coarse-grained model with {nhidden} metastable states at lag={self.lagtime}.'
                              f' The ratio of relaxation timescales between'
                              f' {nhidden} and {nhidden+1} states is only {timescale_ratios[nhidden - 2]}'
                              f' while we recommend at least 1.5.'
                              f' It is possible that the resulting HMM is inaccurate. Handle with caution.',
                              stacklevel=2)
        # run HMM estimate
        from deeptime.markov.hmm import MaximumLikelihoodHMM, init
        init_hmm = init.discrete.metastable_from_msm(self, nhidden, reversible=self.reversible)
        est = MaximumLikelihoodHMM(init_hmm, lagtime=self.lagtime, reversible=self.reversible)
        hmm = est.fit(dtrajs).fetch_model()
        if return_estimator:
            return est, hmm
        return hmm

    def score(self, dtrajs=None, r=2, dim=None):
        r""" Scores the MSM using the dtrajs using the variational approach for Markov processes.

        Implemented according to :cite:`msm-noe2013variational` and :cite:`msm-wu2020variational`.

        Currently only implemented using dense matrices - will be slow for large state spaces.

        Parameters
        ----------
        dtrajs : list of arrays, optional, default=None
            Test data (discrete trajectories). Note that if the test data contains states which are not
            represented in this model, they are ignored.
        r : float or str, optional, default=2
            Overwrite scoring method to be used if desired. Can be any float greater or equal 1 or 'E' for VAMP-r score
            or VAMP-E score, respectively.

            Special cases :cite:`msm-noe2013variational` :cite:`msm-wu2020variational`:

            * 'VAMP1': Sum of singular values of the symmetrized transition matrix.
              If the MSM is reversible, this is equal to the sum of transition
              matrix eigenvalues, also called Rayleigh quotient :cite:`msm-mcgibbon2015variational`.
            * 'VAMP2': Sum of squared singular values of the symmetrized  transition matrix
              :cite:`msm-wu2020variational`. If the MSM is reversible, this is equal to the
              kinetic variance :cite:`msm-noe2015kinetic`.
            * 'VAMPE': Approximation error of the estimated Koopman operator with respect to the true Koopman operator
              up to an additive constant :cite:`msm-wu2020variational`.
        dim : int or None, optional, default=None
            The maximum number of eigenvalues or singular values used in the
            score. If set to None, all available eigenvalues will be used.
        """
        koopman_model = self.empirical_koopman_model
        test_model = None
        if dtrajs is not None:
            assert self.has_count_model, "Needs count model if dtrajs are provided."
            # test data
            from deeptime.markov import TransitionCountEstimator
            cm = TransitionCountEstimator(self.count_model.lagtime, "sliding", sparse=False)\
                .fit(dtrajs).fetch_model()

            # map to present active set
            common_symbols = set(self.count_model.state_symbols).intersection(cm.state_symbols)
            common_states = self.count_model.symbols_to_states(common_symbols)
            c0t_test = np.zeros((self.n_states, self.n_states), dtype=self.transition_matrix.dtype)
            common_states_cm = cm.symbols_to_states(common_symbols)
            # only set counts for common states
            c0t_test[np.ix_(common_states, common_states)] = cm.count_matrix[np.ix_(common_states_cm, common_states_cm)]
            c00_test = np.diag(c0t_test.sum(axis=1))
            ctt_test = np.diag(c0t_test.sum(axis=0))
            assert koopman_model.cov.cov_00.shape == c00_test.shape
            assert koopman_model.cov.cov_0t.shape == c0t_test.shape
            assert koopman_model.cov.cov_tt.shape == ctt_test.shape
            test_model = CovarianceModel(c00_test, c0t_test, ctt_test)
        return vamp_score(koopman_model, r, test_model, dim=dim)


class MarkovStateModelCollection(MarkovStateModel):
    r"""A collection of Markov state models. An instance of this itself behaves like a
    :class:`MSM <MarkovStateModel>`, defaulting to the first (transition matrix, stationary distribution,
    count model) triplet provided. The model can be switched by a call to :meth:`select`.

    Parameters
    ----------
    transition_matrices : list of (n, n) ndarray
        List of transition matrices. Each of them must be row-stochastic.
    stationary_distributions : list of `None` or list of (n, ) ndarray
        List of stationary distributions belonging to transition matrices.
        If list of None, it will be evaluated upon construction of the instance.
    reversible : bool or None
        Whether the transition matrices are reversible with respect to its stationary distribution.
        If None, this is determined from the transition matrix.
    count_models : List of count models or list of None
        List of count models belonging to transition matrices.
    transition_matrix_tolerance : float
        Tolerance under which a matrix is considered to be a transition matrix.

    See Also
    --------
    MarkovStateModel
    MaximumLikelihoodMSM
    """

    def __init__(self, transition_matrices: List[np.ndarray],
                 stationary_distributions: List[np.ndarray],
                 reversible: Optional[bool],
                 count_models: List[TransitionCountModel],
                 transition_matrix_tolerance: float):
        n = len(transition_matrices)
        if n == 0:
            raise ValueError("Needs at least one transition matrix!")
        if len(stationary_distributions) != n or len(count_models) != n:
            raise ValueError(f"Got {n} transition matrices but {len(stationary_distributions)} "
                             f"stationary distributions and {len(count_models)} count models. For a one-to-one "
                             f"correspondence, these must match.")
        for i in range(n):
            if not is_square_matrix(transition_matrices[i]):
                raise ValueError("Transition matrices must be square matrices!")
            n_states = transition_matrices[i].shape[0]
            if stationary_distributions[i] is not None and len(stationary_distributions[i]) != n_states:
                raise ValueError("Lengths of stationary distributions must match respective number of states.")
            if count_models[i] is not None and count_models[i].n_states != n_states:
                raise ValueError("Number of states in count models must match states described in transition matrices.")
        self._transition_matrices = transition_matrices
        self._stationary_distributions = stationary_distributions
        self._reversible = reversible
        self._count_models = count_models
        self._transition_matrix_tolerance = transition_matrix_tolerance
        self._current_model = 0
        super(MarkovStateModelCollection, self).__init__(
            transition_matrix=self._transition_matrices[self.current_model],
            stationary_distribution=self._stationary_distributions[self.current_model],
            reversible=reversible,
            count_model=count_models[self.current_model],
            transition_matrix_tolerance=transition_matrix_tolerance
        )

    @property
    def current_model(self) -> int:
        r""" The currently selected model index.

        :type: int
        """
        return self._current_model

    @property
    def n_connected_msms(self) -> int:
        r"""Number of markov state models in this collection.

        :type: int
        """
        return len(self._transition_matrices)

    def state_symbols(self, model_index: Optional[int] = None) -> np.ndarray:
        r""" Yields the state symbols of a particular model in this collection. Can only be called if the corresponding
        model has a count model.

        Parameters
        ----------
        model_index : int, optional, default=None
            The model index. If None, this evaluates to :meth:`current_model`.

        Returns
        -------
        symbols : (n,) ndarray
            The state symbols associated with the model index. If model index is None, it is defaulted to the
            current model.
        """
        if model_index is None:
            model_index = self.current_model
        self._validate_model_index(model_index)
        if self._count_models[model_index] is None:
            raise ValueError("This can only be evaluated if count models are provided at construction time.")
        return self._count_models[model_index].state_symbols

    def _validate_model_index(self, model_index):
        r"""Validates whether the model index is valid (ie not out of bounds). If it is in bounds, this is a no-op,
        otherwise raises an IndexError.

        Parameters
        ----------
        model_index : int
            The model index

        Raises
        ------
        IndexError
            If the model index is out of bounds.
        """
        if model_index >= self.n_connected_msms:
            raise IndexError(f"There are only {self.n_connected_msms} MSMs in this collection, but "
                             f"information about MSM {model_index} was requested.")

    @property
    def state_fractions(self):
        r""" Yields the fractions of states represented in each of the models in this collection. """
        return [counts.selected_state_fraction for counts in self._count_models]

    @property
    def state_fraction(self):
        r""" The fraction of states represented in the selected model. """
        return self.state_fractions[self.current_model]

    @property
    def count_fractions(self):
        r""" Yields the fraction of counts represented in each of the models in this collection. Calling this method
        assumes that the MSMs in the collection stem from actual data with state statistics embedded in the count
        models. """
        return [counts.selected_count_fraction if counts is not None else None for counts in self._count_models]

    @property
    def count_fraction(self):
        r""" The fraction of counts represented in the selected model. """
        return self.count_fractions[self.current_model]

    def select(self, model_index):
        r""" Selects a different model in the collection. Changes the behavior of the collection to mimic a MSM
        associated to respective transition matrix, stationary distribution, and count model.

        Parameters
        ----------
        model_index : int
            The model index.

        Raises
        ------
        IndexError
            If model index is out of bounds.

        """
        self._validate_model_index(model_index)
        self._invalidate_caches()
        super().__init__(transition_matrix=self._transition_matrices[model_index],
                         stationary_distribution=self._stationary_distributions[model_index],
                         reversible=self.reversible, count_model=self._count_models[model_index],
                         transition_matrix_tolerance=self.transition_matrix_tolerance)
        self._current_model = model_index


def _svd_sym_koopman(msm: MarkovStateModel, empirical: bool, epsilon=1e-10) -> CovarianceKoopmanModel:
    """ Computes the SVD of the symmetrized Koopman operator in the empirical distribution, returns as Koopman model.
    """
    K = msm.transition_matrix
    if empirical:
        C = msm.count_model.count_matrix
        if issparse(C):
            C = C.toarray()
        cov0t = C
        cov00 = np.diag(cov0t.sum(axis=1))
        covtt = np.diag(cov0t.sum(axis=0))
        cov = CovarianceModel(cov_00=cov00, cov_0t=cov0t, cov_tt=covtt)
        # reweight operator to empirical distribution
        C0t_re = cov00 @ K
        # symmetrized operator and SVD
        K_sym = np.linalg.multi_dot([spd_inv_sqrt(cov.cov_00, epsilon=epsilon), C0t_re,
                                     spd_inv_sqrt(cov.cov_tt, epsilon=epsilon)])
    else:
        cov00 = np.diag(msm.stationary_distribution)
        covtt = np.diag(msm.stationary_distribution)
        cov0t = cov00 @ K
        cov = CovarianceModel(cov_00=cov00, cov_0t=cov0t, cov_tt=covtt)
        K_sym = np.linalg.multi_dot([spd_inv_sqrt(cov.cov_00, epsilon=epsilon), cov0t,
                                     spd_inv_sqrt(cov.cov_tt, epsilon=epsilon)])
    U, S, Vt = scipy.linalg.svd(K_sym, compute_uv=True)
    U = spd_inv_sqrt(cov.cov_00, epsilon=epsilon) @ U
    Vt = Vt @ spd_inv_sqrt(cov.cov_tt, epsilon=epsilon)
    return CovarianceKoopmanModel(U, S, Vt.T, cov, K.shape[0], K.shape[0])
