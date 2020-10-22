import unittest
import numpy as np

from deeptime.util.exceptions import NotConvergedWarning
from tests.markov.tools.numeric import assert_allclose
import scipy
import scipy.sparse
import warnings

from os.path import abspath, join
from os import pardir

from deeptime.markov.tools.estimation.dense.mle import mle_trev_given_pi as impl_dense
from deeptime.markov.tools.estimation.sparse.mle import mle_trev_given_pi as impl_sparse

from deeptime.markov.tools.estimation import transition_matrix as apicall
from deeptime.markov.tools.analysis import stationary_distribution, is_transition_matrix

testpath = abspath(join(abspath(__file__), pardir)) + '/testfiles/'


def transition_matrix_reversible_fixpi(Z, mu, maxerr=1e-10, maxiter=10000, return_iterations=False,
                                       warn_not_converged=True):
    r"""
    maximum likelihood transition matrix with fixed stationary distribution

    developed by Fabian Paul and Frank Noe

    Parameters
    ----------
    Z: ndarray, shape (n,n)
        count matrix
    mu: ndarray, shape (n)
        stationary distribution
    maxerr: float
        Will exit (as converged) when the 2-norm of the Langrange multiplier vector changes less than maxerr
        in one iteration
    maxiter: int
        Will exit when reaching maxiter iterations without reaching convergence.
    return_iterations: bool (False)
        set true in order to return (T, it), where T is the transition matrix and it is the number of iterations needed
    warn_not_converged : bool, default=True
        Prints a warning if not converged.

    Returns
    -------
    T, the transition matrix. When return_iterations=True, (T,it) is returned with it the number of iterations needed

    """
    it = 0
    n = len(mu)
    # constants
    B = Z + Z.transpose()
    # variables
    csum = np.sum(Z, axis=1)
    if (np.min(csum) <= 0):
        raise ValueError('Count matrix has rowsum(s) of zero. Require a count matrix with positive rowsums.')
    if (np.min(mu) <= 0):
        raise ValueError('Stationary distribution has zero elements. Require a positive stationary distribution.')
    if (np.min(np.diag(Z)) == 0):
        raise ValueError(
            'Count matrix has diagonals with 0. Cannot guarantee convergence of algorithm. Suggestion: add a small prior (e.g. 1e-10) to the diagonal')
    l = 1.0 * csum
    lnew = 1.0 * csum
    q = np.zeros((n))
    A = np.zeros((n, n))
    D = np.zeros((n, n))
    # iterate lambda
    converged = False
    while (not converged) and (it < maxiter):
        # q_i = mu_i / l_i
        np.divide(mu, l, q)
        # d_ij = (mu_i / mu_j) * (l_j/l_i) + 1
        D[:] = q[:, np.newaxis]
        D /= q
        D += 1
        # a_ij = b_ij / d_ij
        np.divide(B, D, A)
        # new l_i = rowsum_i(A)
        np.sum(A, axis=1, out=lnew)
        # evaluate change
        err = np.linalg.norm(l - lnew, 2)
        # is it converged?
        converged = (err <= maxerr)
        # copy new to old l-vector
        l[:] = lnew[:]
        it += 1
    if warn_not_converged and (not converged) and (it >= maxiter):
        warnings.warn('NOT CONVERGED: 2-norm of Langrange multiplier vector is still ' +
                         str(err) + ' > ' + str(maxerr) + ' after ' + str(it) +
                         ' iterations. Increase maxiter or decrease maxerr',
                      NotConvergedWarning)
    # compute T from Langrangian multipliers
    T = np.divide(A, l[:, np.newaxis])
    # return
    if return_iterations:
        return T, it
    else:
        return T

impl_dense_Frank = transition_matrix_reversible_fixpi


class Test_mle_trev_given_pi(unittest.TestCase):

    def test_mle_trev_given_pi(self):
        C = np.loadtxt(testpath + 'C_1_lag.dat')
        pi = np.loadtxt(testpath + 'pi.dat')

        T_impl_algo_dense_type_dense = impl_dense(C, pi)
        T_impl_algo_sparse_type_sparse = impl_sparse(scipy.sparse.csr_matrix(C), pi).toarray()
        T_Frank = impl_dense_Frank(C, pi)
        T_api_algo_dense_type_dense = apicall(C, reversible=True, mu=pi, method='dense')
        T_api_algo_sparse_type_dense = apicall(C, reversible=True, mu=pi, method='sparse')
        T_api_algo_dense_type_sparse = apicall(scipy.sparse.csr_matrix(C), reversible=True, mu=pi, method='dense').toarray()
        T_api_algo_sparse_type_sparse = apicall(scipy.sparse.csr_matrix(C), reversible=True, mu=pi, method='sparse').toarray()
        T_api_algo_auto_type_dense = apicall(C, reversible=True, mu=pi, method='auto')
        T_api_algo_auto_type_sparse = apicall(scipy.sparse.csr_matrix(C), reversible=True, mu=pi, method='auto').toarray()

        assert_allclose(T_impl_algo_dense_type_dense, T_Frank)
        assert_allclose(T_impl_algo_sparse_type_sparse, T_Frank)
        assert_allclose(T_api_algo_dense_type_dense, T_Frank)
        assert_allclose(T_api_algo_sparse_type_dense, T_Frank)
        assert_allclose(T_api_algo_dense_type_sparse, T_Frank)
        assert_allclose(T_api_algo_sparse_type_sparse, T_Frank)
        assert_allclose(T_api_algo_auto_type_dense, T_Frank)
        assert_allclose(T_api_algo_auto_type_sparse, T_Frank)

        assert is_transition_matrix(T_Frank)
        assert is_transition_matrix(T_impl_algo_dense_type_dense)
        assert is_transition_matrix(T_impl_algo_sparse_type_sparse)
        assert is_transition_matrix(T_api_algo_dense_type_dense)
        assert is_transition_matrix(T_api_algo_sparse_type_dense)
        assert is_transition_matrix(T_api_algo_dense_type_sparse)
        assert is_transition_matrix(T_api_algo_sparse_type_sparse)
        assert is_transition_matrix(T_api_algo_auto_type_dense)
        assert is_transition_matrix(T_api_algo_auto_type_sparse)

        assert_allclose(stationary_distribution(T_Frank), pi)
        assert_allclose(stationary_distribution(T_impl_algo_dense_type_dense), pi)
        assert_allclose(stationary_distribution(T_impl_algo_sparse_type_sparse), pi)
        assert_allclose(stationary_distribution(T_api_algo_dense_type_dense), pi)
        assert_allclose(stationary_distribution(T_api_algo_sparse_type_dense), pi)
        assert_allclose(stationary_distribution(T_api_algo_dense_type_sparse), pi)
        assert_allclose(stationary_distribution(T_api_algo_sparse_type_sparse), pi)
        assert_allclose(stationary_distribution(T_api_algo_auto_type_dense), pi)
        assert_allclose(stationary_distribution(T_api_algo_auto_type_sparse), pi)

    def test_warnings(self):
        C = np.loadtxt(testpath + 'C_1_lag.dat')
        pi = np.loadtxt(testpath + 'pi.dat')
        ncw = NotConvergedWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore')
            warnings.simplefilter('always', category=ncw)
            impl_sparse(scipy.sparse.csr_matrix(C), pi, maxiter=1)
            assert len(w) == 1
            assert issubclass(w[-1].category, ncw)

            impl_dense(C, pi, maxiter=1)
            assert len(w) == 2
            assert issubclass(w[-1].category, ncw)
