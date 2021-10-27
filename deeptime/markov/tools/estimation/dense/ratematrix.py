r"""
.. moduleauthor:: Fabian Paul <fab@zedat.fu-berlin.de>
"""

import logging
import warnings
import numpy as np
import scipy as sp
import scipy.sparse

from scipy.optimize import fmin_l_bfgs_b
from scipy.special import exprel

from deeptime.util.exceptions import NotConvergedWarning, NotConvergedError
from ...kahandot import kdot, ksum
from ...estimation import transition_matrix
from ...analysis import stationary_distribution


__all__ = [
    'PseudoGeneratorEstimator',
    'TruncatedLogarithmEstimator',
    'CrommelinVandenEijndenEstimator',
    'KalbfleischLawlessEstimator',
    'NotConnectedWarning',
    'estimate_rate_matrix'
]


class NotConnectedWarning(UserWarning):
    pass


def getV(lam, tau):
    n = len(lam)
    V = np.zeros((n, n))
    ones = np.ones((n, n))
    delta = tau * (lam - lam[:, np.newaxis])  # delta_ij = lam_j-lam_i
    positive = delta >= 0
    negative = delta < 0
    a1 = tau * np.exp(tau * lam)[:, np.newaxis] * ones  # tau*e^l_i
    b1 = exprel(delta)
    V[negative] = a1[negative] * b1[negative]
    a2 = tau * np.exp(tau * lam)[np.newaxis, :] * ones  # tau*e^l_j *
    b2 = exprel(-delta)
    V[positive] = a2[positive] * b2[positive]
    return V


def eigen_decomposition(M, pi):
    # eigen decomposition for reversible transition matrix
    sqrt_pi = np.sqrt(pi)
    Msym = sqrt_pi[:, np.newaxis] * M / sqrt_pi  # Msym_ij = M_ij sqrt(pi_i/pi_j)
    lam, B = np.linalg.eigh(Msym)
    A = B / sqrt_pi[:, np.newaxis]  # A_ij = B_ij / sqrt(pi_i)
    Ainv = B.T * sqrt_pi  # Ainv_ij = B_ji * sqrt(pi_j)
    A = np.ascontiguousarray(A)
    Ainv = np.ascontiguousarray(Ainv)
    return lam, A, Ainv


def vdot(a, b):
    # Kahan summation scalar product
    n = a.shape[0]
    return kdot(a.reshape((1, n)), b.reshape((n, 1)))[0, 0]


def sum1(M):
    # 1-D Kahan summation along axis 1
    n, m = M.shape
    x = np.zeros(n)
    for i in range(n):
        x[i] = ksum(np.reshape(M[i, :], (1, m)))
    return x


def raise_or_warn(msg, on_error, warning=UserWarning, exception=RuntimeError):
    if on_error == 'raise':
        raise exception(msg)
    elif on_error == 'warn':
        warnings.warn(msg, warning)
    else:
        raise ValueError('Unsupported value of on_error (%s). Should be "raise" or "warn".'%on_error)


class _RateMatrixEstimator:
    # base class: includes parametrization of K matrix and the basic class interface
    def __init__(self, C, dt=1.0, sparsity=None, t_agg=None, pi=None, tol=1.0E7, maxiter=100000, on_error='raise'):
        if not np.all(C >= 0):
            raise ValueError('Elements of C matrix should be non-negative.')
        if not C.shape[0] == C.shape[1]:
            raise ValueError('C matrix should be square.')
        self.zero_C = np.where(C == 0)
        self.nonzero_C = np.where(C != 0)
        if not dt > 0.0:
            raise ValueError('dt should be positive.')
        if sparsity is not None:
            if not np.all(sparsity >= 0):
                raise ValueError('Elements of sparsity matrix should be non-negative.')
            if not sparsity.shape[0] == sparsity.shape[1] == C.shape[0]:
                raise ValueError('sparsity matrix should be square and of the same dimension as the C matrix.')
        if pi is not None:
            if not np.all(pi > 0.0):
                raise ValueError('pi should be positive.')
            if not pi.shape[0] == C.shape[0]:
                raise ValueError('pi and the C matrix should have compatible dimensions.')
            if not np.allclose(np.sum(pi), 1.0):
                raise ValueError('pi should sum to one.')
        if t_agg is not None:
            if not t_agg > 0:
                raise ValueError('t_agg should be positive.')
            self.t_agg = t_agg
        else:
            self.t_agg = dt*C.sum()

        self.N = C.shape[0]
        self.C = C
        self.pi = pi
        self.dt = dt
        self.sparsity = sparsity
        self.tol = tol
        self.verbose = False
        self.maxiter = maxiter
        self.on_error = on_error


class _ReversibleRateMatrixEstimator(_RateMatrixEstimator):
    # this estimator requires a known stationary vector
    def __init__(self, C, pi, dt=1.0, sparsity=None, t_agg=None, tol=1.0E7, maxiter=100000, on_error='raise'):
        super(_ReversibleRateMatrixEstimator, self).__init__(C, dt=dt, sparsity=sparsity, t_agg=t_agg, pi=pi, tol=tol, maxiter=maxiter, on_error=on_error)

        if self.sparsity is None:
            self.I, self.J = np.triu_indices(self.N, k=1)
            # lower bounds
            self.lower_bounds = np.zeros(len(self.I))
            self.bounds = [(0, None)] * len(self.I)
        else:
            A = self.sparsity + self.sparsity.T
            A[np.tril_indices(self.N, k=0)] = 0
            Asparse = sp.sparse.coo_matrix(A)
            self.I = Asparse.row
            self.J = Asparse.col
            # set lower bounds
            self.bounds = [None] * len(self.I)
            self.lower_bounds = np.zeros(len(self.I))
            for i, j, n in zip(self.I, self.J, range(len(self.I))):
                self.lower_bounds[n] = 1.0 / (self.t_agg * (1.0 / self.pi[i] + 1.0 / self.pi[j]))
                assert self.lower_bounds[n] > 0.0 # self-consitency test
                self.bounds[n] = (self.lower_bounds[n], None)

        # for matrix derivatives
        self.D = [None] * (len(self.I))
        for i, j, n in zip(self.I, self.J, range(len(self.I))):
            Dn = sp.sparse.lil_matrix((self.N, self.N))
            Dn[i, j] = 1.0 / self.pi[i]
            Dn[j, i] = 1.0 / self.pi[j]
            Dn[i, i] = -1.0 / self.pi[i]
            Dn[j, j] = -1.0 / self.pi[j]
            self.D[n] = sp.sparse.coo_matrix(Dn)

    def selftest(self):
        self.count = -1
        x0 = self.initial

        direction = np.random.rand(len(self.initial)) * 1.0E-12
        f1, grad1 = self.function_and_gradient(x0)
        f2 = self.function(x0 + direction)
        df = np.dot(grad1, direction)
        # in general we would have to use |a-b|/max(|a|,|b|) but since mostly |df|>|f1-f2| we can use error/|df|
        err = np.abs((f2 - f1 - df) / (df))
        logging.info('Self-test for rate matrix yields a finite difference of '
                     '%f and a directional derivative of %f. This corresponds '
                     'to a relative error of %f.' % (f2-f1, df, err))

    def run(self):
        """Run the minimization.

        Returns
        -------
        K : (N,N) ndarray
            the optimal rate matrix
        """
        if self.verbose:
            self.selftest()
        self.count = 0
        if self.verbose:
            logging.info('initial value of the objective function is %f'
                         % self.function(self.initial))
        theta0 = self.initial
        theta, f, d = fmin_l_bfgs_b(self.function_and_gradient, theta0, fprime=None, args=(),
                                    approx_grad=False, bounds=self.bounds, factr=self.tol,
                                    pgtol=1.0E-11, disp=0, maxiter=self.maxiter, maxfun=self.maxiter, maxls=100)
        if self.verbose:
            logging.info('l_bfgs_b says: '+str(d))
            logging.info('objective function value reached: %f' % f)
        if d['warnflag'] != 0:
            raise_or_warn(str(d), on_error=self.on_error, warning=NotConvergedWarning, exception=NotConvergedError)

        K = np.zeros((self.N, self.N))
        K[self.I, self.J] = theta / self.pi[self.I]
        K[self.J, self.I] = theta / self.pi[self.J]
        np.fill_diagonal(K, -np.sum(K, axis=1))
        self.K = K
        return K


class PseudoGeneratorEstimator(_RateMatrixEstimator):
    def __init__(self, C, dt, sparsity=None, t_agg=None, pi=None, tol=1.0E7, maxiter=100000, on_error='raise'):
        super(PseudoGeneratorEstimator, self).__init__(C, dt, sparsity=sparsity, t_agg=t_agg, pi=pi, tol=tol, maxiter=maxiter, on_error=on_error)

    def run(self, maxiter=100000, on_error='raise'):
        if self.pi is None:
            self.T = transition_matrix(self.C, reversible=True)
            self.pi = stationary_distribution(self.T)
        else:
            self.T = transition_matrix(self.C, reversible=True, mu=self.pi)

        self.K = (self.T - np.eye(self.N)) / self.dt
        return self.K


class TruncatedLogarithmEstimator(_RateMatrixEstimator):
    def __init__(self, C, dt, sparsity=None, t_agg=None, pi=None, tol=1.0E7, maxiter=100000, on_error='raise'):
        super(TruncatedLogarithmEstimator, self).__init__(C, dt, sparsity=sparsity, t_agg=t_agg, pi=pi, tol=tol, maxiter=maxiter, on_error=on_error)

    def run(self, maxiter=100000, on_error='raise'):
        if self.pi is None:
            self.T = transition_matrix(self.C, reversible=True)
            self.pi = stationary_distribution(self.T)
        else:
            self.T = transition_matrix(self.C, reversible=True, mu=self.pi)

        self.K = np.maximum(np.array(sp.linalg.logm(np.dot(self.T, self.T))/(2.0*self.dt)), 0)
        np.fill_diagonal(self.K, 0)
        np.fill_diagonal(self.K, -sum1(self.K))

        return self.K


class CrommelinVandenEijndenEstimator(_ReversibleRateMatrixEstimator):
    r"""Estimate rate matrix from count matrix using Crommelin-Vanden-Eijnden algorithm.

    The algorithm consists of minimizing the following objective function:

    .. math:: f(K)=\sum_{ij}\left(\sum_{kl}U_{ik}^{-1}K_{kl}U_{lj}-L_{ij}\right)^2\left|\Lambda_{i}\Lambda_{j}\right|

    where :math:`\Lambda_i` are the eigenvalues of :math:`T` and :math:`U` is the matrix
    of its (right) eigenvectors; :math:`L_{ij} = \delta_{ij} \frac{1}{\tau} \log \left| \Lambda_i \right|`.

    :math:`T` is computed from a count matrix using the reversible
    maximum likelihood estimator.

    If sparsity is not None, the maximization is carried out under
    the constraint :math:`K_{ij}=0` and :math:`K_{ji}=0` for all i,j
    where sparsity[i,j]+sparsity[j,i]=0.

    Parameters
    ----------
    C : (N,N) ndarray
        count matrix at a lag time dt
    K0 : (N,N) ndarray
        initial guess for the rate matrix
    pi : (N) ndarray
        the stationary distribution of the desired rate matrix K
    dt : float, optional, default=1.0
        lag time that was used to estimate C
    sparsity : (N,N) ndarray or None, optional, default=None
        If sparsity is None, a fully occupied rate matrix will be estimated.
        Alternatively, with the methods 'CVE' and 'KL' a ndarray of the
        same shape as C can be given. If sparsity[i,j]=0 and sparsity[j,i]=0
        the rate matrix elements :math:`K_{ij}` and :math:`K_{ji}` will be
        constrained to zero.
    t_agg : float, optional
        the aggregated simulation time
        By default this is the total number of transition counts times
        the lag time (no sliding window counting). This value is used
        to compute the lower bound on the transition rate (that are not zero).
        If sparsity is None, this value is ignored.
    tol : float, optional, default = 1.0E7
        Tolerance of the quasi-Newton algorithm that is used to minimize
        the objective function. This is passed as the `factr` parameter to
        `scipy.optimize.fmin_l_bfgs_b`.
        Typical values for factr are: 1e12 for low accuracy; 1e7
        for moderate accuracy; 10.0 for extremely high accuracy.
    maxiter : int, optional, default = 100000
        Minimization of the objective function will do at most this number
        of steps.
    on_error : string, optional, default = 'raise'
        What to do then an error happend. When 'raise' is given, raise
        an exception. When 'warn' is given, produce a (Python) warning.

    Note
    ----
    To compute the rate matrix, call the `run` method of the estimator object.
    """

    def __init__(self, T, K0, pi, dt=1.0, sparsity=None, t_agg=None, tol=1.0E7, maxiter=100000, on_error='raise'):
        from deeptime.markov.tools.analysis import is_transition_matrix

        super(CrommelinVandenEijndenEstimator, self).__init__(T, pi, dt=dt, sparsity=sparsity, t_agg=t_agg, tol=tol, maxiter=maxiter, on_error=on_error)

        if not K0.shape[0] == K0.shape[1] == self.N:
            raise ValueError('Shapes of K0 matrix (initial guess) and count matrix do not match.')
        if not is_transition_matrix(T):
            raise_or_warn('T is not a valid transition matrix.', self.on_error)

        evals, self.U, self.Uinv = eigen_decomposition(T, self.pi)
        if not np.all(np.abs(evals) > 0.0):  # don't allow eigenvalue==exactly zero
            raise ValueError('T has eigenvalues that are exactly zero, can\'t proceed with rate matrix estimation. '
                             'If the CVE method is only used to intitialize the KL method, you might try to call the KL '
                             'method with an initial guess of the rate matrix (K0) instead of intializing with CVE.')
        assert np.allclose(self.Uinv.dot(T).dot(self.U), np.diag(evals))  # self-consistency test

        self.c = np.abs(evals)
        self.L = np.diag(np.log(np.abs(evals)) / self.dt)

        theta = self.pi[self.I] * K0[self.I, self.J]
        self.initial = np.maximum(theta, self.lower_bounds)

    def function(self, x):
        if not np.all(x>=self.lower_bounds):
            raise_or_warn('Optimizer violated the lower bounds for rate matrix elements.', self.on_error)

        # compute K
        K = np.zeros((self.N, self.N))
        K[self.I, self.J] = x / self.pi[self.I]
        K[self.J, self.I] = x / self.pi[self.J]
        np.fill_diagonal(K, -np.sum(K, axis=1))
        # compute function
        delta = self.Uinv.dot(K).dot(self.U) - self.L
        f = self.c.dot((delta * delta).dot(self.c))
        return f

    def function_and_gradient(self, x):
        if not np.all(x>=self.lower_bounds):
            raise_or_warn('Optimizer violated the lower bounds for rate matrix elements.', self.on_error)

        # compute K
        K = np.zeros((self.N, self.N))
        K[self.I, self.J] = x / self.pi[self.I]
        K[self.J, self.I] = x / self.pi[self.J]
        np.fill_diagonal(K, -np.sum(K, axis=1))
        # compute function
        delta = self.Uinv.dot(K).dot(self.U) - self.L
        f = self.c.dot((delta * delta).dot(self.c))

        if self.verbose:
            logging.info('iteration=%d, norm^2=%f' % (self.count, f))

        self.count += 1

        # compute gradient
        X = self.c[:, np.newaxis] * delta * self.c * 2.0
        Y = self.U.dot(X.T).dot(self.Uinv).T
        grad = np.zeros(len(x))
        for i in range(len(x)):
            Di = self.D[i]
            grad[i] = Y[Di.row, Di.col].dot(Di.data)  # sparse version; scipy's sparse multiply oddly returns dense
        return (f, grad)


class KalbfleischLawlessEstimator(_ReversibleRateMatrixEstimator):
    r"""Estimate rate matrix from count matrix using Kalbfleisch-Lawless algorithm.

    The algorithm consists of maximizing the following log-likelihood:

    .. math:: f(K)=\log L=\sum_{ij}C_{ij}\log(e^{K\tau})_{ij}

    where :math:`C_{ij}` are the transition counts at a lag-time :math:`\tau`.
    Here :math:`e` is the matrix exponential and the logarithm is taken element-wise.

    If sparsity is not None, the maximization is carried out under
    the constraint :math:`K_{ij}=0` and :math:`K_{ji}=0` for all i,j
    where sparsity[i,j]+sparsity[j,i]=0.

    Parameters
    ----------
    C : (N,N) ndarray
        count matrix at a lag time dt
    K0 : (N,N) ndarray
        initial guess for the rate matrix
    pi : (N) ndarray
        the stationary distribution of the desired rate matrix K
    dt : float, optional, default=1.0
        lag time that was used to estimate C
    sparsity : (N,N) ndarray or None, optional, default=None
        If sparsity is None, a fully occupied rate matrix will be estimated.
        Alternatively, with the methods 'CVE' and 'KL' a ndarray of the
        same shape as C can be given. If sparsity[i,j]=0 and sparsity[j,i]=0
        the rate matrix elements :math:`K_{ij}` and :math:`K_{ji}` will be
        constrained to zero.
    t_agg : float, optional
        the aggregated simulation time
        By default this is the total number of transition counts times
        the lag time (no sliding window counting). This value is used
        to compute the lower bound on the transition rate (that are not zero).
        If sparsity is None, this value is ignored.
    tol : float, optional, default = 1.0E7
        Tolerance of the quasi-Newton algorithm that is used to minimize
        the objective function. This is passed as the `factr` parameter to
        `scipy.optimize.fmin_l_bfgs_b`.
        Typical values for factr are: 1e12 for low accuracy; 1e7
        for moderate accuracy; 10.0 for extremely high accuracy.
    maxiter : int, optional, default = 100000
        Minimization of the objective function will do at most this number
        of steps.
    on_error : string, optional, default = 'raise'
        What to do then an error happend. When 'raise' is given, raise
        an exception. When 'warn' is given, produce a (Python) warning.

    Note
    ----
    To compute the rate matrix, call the `run` method of the estimator object.
    """

    def __init__(self, C, K0, pi, dt=1.0, sparsity=None, t_agg=None, tol=1.0E7, maxiter=100000, on_error='raise'):
        super(KalbfleischLawlessEstimator, self).__init__(C, pi, dt=dt, sparsity=sparsity, t_agg=t_agg, tol=tol, maxiter=maxiter, on_error=on_error)

        if not K0.shape[0] == K0.shape[1] == self.N:
            raise ValueError('Shapes of K0 matrix (initial guess) and count matrix do not match.')

        # specific variables for KL estimator
        self.sqrt_pi = np.sqrt(pi)

        self.initial = self.pi[self.I] * K0[self.I, self.J]
        self.initial = np.maximum(self.initial, self.lower_bounds)

    def function(self, x):
        self.count += 1
        if not np.all(x>=self.lower_bounds):
            raise_or_warn('Optimizer violated the lower bounds for rate matrix elements.', self.on_error)

        # compute function
        K = np.zeros((self.N, self.N))
        K[self.I, self.J] = x / self.pi[self.I]
        K[self.J, self.I] = x / self.pi[self.J]
        np.fill_diagonal(K, -sum1(K))
        T = sp.linalg.expm(self.dt * K)  # use eigendecomposition instead?
        T[self.zero] = 1.0  # set unused elements to dummy to avoid division by 0
        # check T!=0 for C!=0
        nonzero_C = np.where(self.C != 0)
        if np.any(np.abs(T[nonzero_C]) <= 1E-20):
            warnings.warn('Warning: during iteration T_ij became very small while C(tau)_ij > 0.', NotConnectedWarning)
        f = ksum(self.C * np.log(T))

        if self.verbose:
            logging.info('iteration=%d, log L=%f' % (self.count, f))
        return -f

    def function_and_gradient(self, x):
        if not np.all(x>=self.lower_bounds):
            raise_or_warn('Optimizer violated the lower bounds for rate matrix elements.', self.on_error)

        # compute function
        K = np.zeros((self.N, self.N))
        K[self.I, self.J] = x / self.pi[self.I]
        K[self.J, self.I] = x / self.pi[self.J]
        np.fill_diagonal(K, -sum1(K))

        # compute eigendecomposition
        lam, A, Ainv = eigen_decomposition(K, self.pi)

        # T = kdot(kdot(A,np.diag(np.exp(self.tau*lam))),Ainv)
        T = sp.linalg.expm(self.dt * K)
        T[self.zero_C] = 1.0  # set unused elements to dummy to avoid division by 0
        # check T!=0 for C!=0
        if np.any(np.abs(T[self.nonzero_C]) <= 1E-20):
            warnings.warn('Warning: during iteration T_ij became very small while C(tau)_ij > 0. Regularizing T.', NotConnectedWarning)
            for i,j in zip(*self.nonzero_C):
                if T[i,j] <= 1E-20:
                    T[i,j] = 1E-20

        f = ksum(self.C * np.log(T))

        if self.verbose:
            logging.info('iteration=%d, log L=%f' % (self.count, f))
        self.count += 1

        V = getV(lam, self.dt)

        # M = Ainv.dot(Ctau.T/T.T).dot(A)*V.T
        M = kdot(kdot(Ainv, np.ascontiguousarray(self.C.T / T.T)), A) * np.ascontiguousarray(V.T)
        # H = A.dot(M).dot(Ainv)
        H = kdot(kdot(A, M), Ainv)

        grad = np.zeros(len(x))
        for i in range(len(x)):
            Di = self.D[i]
            grad[i] = vdot(H[Di.col, Di.row], Di.data)  # this is (H.T*Di).sum()

        return (-f, -grad)


def estimate_rate_matrix(C, dt=1.0, method='KL', sparsity=None,
                         t_agg=None, pi=None, tol=1.0E7, K0=None,
                         maxiter=100000, on_error='raise'):
    r"""Estimate a reversible rate matrix from a count matrix.

    Parameters
    ----------
    C : (N,N) ndarray
        count matrix at a lag time dt
    dt : float, optional, default=1.0
        lag time that was used to estimate C
    method : {'KL', 'CVE', 'pseudo', 'truncated_log'}
        Method to use for estimation of the rate matrix.

        * 'pseudo' selects the pseudo-generator. A reversible transition
          matrix T is estimated and (T-Id)/dt is retruned as the rate matrix.

        * 'truncated_log' selects the truncated logarithm [3]_. A
          reversible transition matrix T is estimated and max(logm(T*T)/(2dt),0)
          is returned as the rate matrix. logm is the matrix logarithm and
          the maximum is taken element-wise.

        * 'CVE' selects the algorithm of Crommelin and Vanden-Eijnden [1]_.
          It consists of minimizing the following objective function:

          .. math:: f(K)=\sum_{ij}\left(\sum_{kl} U_{ik}^{-1}K_{kl}U_{lj}-L_{ij}\right)^2 \left|\Lambda_{i}\Lambda_{j}\right|

          where :math:`\Lambda_i` are the eigenvalues of :math:`T` and :math:`U`
          is the matrix of its (right) eigenvectors; :math:`L_{ij}=\delta_{ij}\frac{1}{\tau}\log\left|\Lambda_i\right|`.
          :math:`T` is computed from C using the reversible maximum likelihood
          estimator.

        * 'KL' selects the algorihtm of Kalbfleisch and Lawless [2]_.
          It consists of maximizing the following log-likelihood:

          .. math:: f(K)=\log L=\sum_{ij}C_{ij}\log(e^{K\Delta t})_{ij}

          where :math:`C_{ij}` are the transition counts at a lag-time :math:`\Delta t`.
          Here :math:`e` is the matrix exponential and the logarithm is taken
          element-wise.

    sparsity : (N,N) ndarray or None, optional, default=None
        If sparsity is None, a fully occupied rate matrix will be estimated.
        Alternatively, with the methods 'CVE' and 'KL' a ndarray of the
        same shape as C can be supplied. If sparsity[i,j]=0 and sparsity[j,i]=0
        the rate matrix elements :math:`K_{ij}` and :math:`K_{ji}` will be
        constrained to zero.
    t_agg : float, optional
        the aggregated simulation time;
        by default this is the total number of transition counts times
        the lag time (no sliding window counting). This value is used
        to compute the lower bound on the transition rate (that are not zero).
        If sparsity is None, this value is ignored.
    pi : (N) ndarray, optional
        the stationary vector of the desired rate matrix K.
        If no pi is given, the function takes the stationary vector
        of the MLE reversible T matrix that is computed from C.
    tol : float, optional, default = 1.0E7
        Tolerance of the quasi-Newton algorithm that is used to minimize
        the objective function. This is passed as the `factr` parameter to
        `scipy.optimize.fmin_l_bfgs_b`.
        Typical values for factr are: 1e12 for low accuracy; 1e7
        for moderate accuracy; 10.0 for extremely high accuracy.
    maxiter : int, optional, default = 100000
        Minimization of the objective function will do at most this number
        of steps.
    on_error : string, optional, default = 'raise'
        What to do then an error happend. When 'raise' is given, raise
        an exception. When 'warn' is given, produce a (Python) warning.

    Returns
    -------
    K : (N,N) ndarray
        the optimal rate matrix

    Notes
    -----
    In this implementation the algorithm of Crommelin and Vanden-Eijnden
    (CVE) is initialized with the pseudo-generator estimate. The
    algorithm of Kalbfleisch and Lawless (KL) is initialized using the
    CVE result.

    Example
    -------
    >>> import numpy as np
    >>> from deeptime.markov.tools.estimation import rate_matrix
    >>> C = np.array([[100,1],[50,50]])
    >>> R = rate_matrix(C)
    >>> print(np.array_str(R, precision=5))
    [[-0.01385  0.01385]
     [ 0.6993  -0.6993 ]]

    References
    ----------
    .. [1] D. Crommelin and E. Vanden-Eijnden. Data-based inference of
        generators for markov jump processes using convex optimization.
        Multiscale. Model. Sim., 7(4):1751-1778, 2009.
    .. [2] J. D. Kalbfleisch and J. F. Lawless. The analysis of panel
        data under a markov assumption. J. Am. Stat. Assoc.,
        80(392):863-871, 1985.
    .. [3] E. B. Davies. Embeddable Markov Matrices. Electron. J. Probab.
        15:1474, 2010.
    """
    if method not in ['pseudo', 'truncated_log', 'CVE', 'KL']:
        raise ValueError("method must be one of 'KL', 'CVE', 'pseudo' or 'truncated_log'")

    # special case: truncated matrix logarithm
    if method == 'truncated_log':
        e_tlog = TruncatedLogarithmEstimator(C, dt=dt, sparsity=sparsity, t_agg=t_agg, pi=pi, tol=tol, maxiter=maxiter, on_error=on_error)
        e_tlog.run()
        return e_tlog.K

    # remaining algorithms are based on each other in the order pseudo->CVE->KL
    if method == 'pseudo' or method == 'CVE' or K0 is None or pi is None:
        e_pseudo = PseudoGeneratorEstimator(C, dt=dt, sparsity=sparsity, t_agg=t_agg, pi=pi, tol=tol, maxiter=maxiter, on_error=on_error)
        e_pseudo.run()
        if pi is None:
            pi = e_pseudo.pi
        if method == 'pseudo':
            return e_pseudo.K

    if method == 'CVE' or K0 is None:
        if K0 is not None:
            K_init = K0
        else:
            K_init = e_pseudo.K
        e_CVE = CrommelinVandenEijndenEstimator(e_pseudo.T, K_init, pi, dt=dt, sparsity=sparsity, t_agg=t_agg, tol=tol, maxiter=maxiter, on_error=on_error)
        e_CVE.run()
        if method == 'CVE':
            return e_CVE.K

    if K0 is not None:
        K_init = K0
    else:
        K_init = e_CVE.K
    e_KL = KalbfleischLawlessEstimator(C, K_init, pi, dt=dt, sparsity=sparsity, t_agg=t_agg, tol=tol, maxiter=maxiter, on_error=on_error)
    e_KL.run()
    return e_KL.K
