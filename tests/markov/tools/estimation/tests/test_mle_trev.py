import unittest
import numpy as np

from deeptime.markov.tools.analysis import stationary_distribution
from deeptime.util.exceptions import NotConvergedWarning
from tests.markov.tools.numeric import assert_allclose
import scipy
import scipy.sparse
import warnings

from os.path import abspath, join
from os import pardir

from deeptime.markov.tools.estimation.sparse.mle import mle_trev as impl_sparse
from deeptime.markov.tools.estimation.dense.mle import mle_trev as impl_dense
from deeptime.markov.tools.estimation import transition_matrix as apicall

testpath = abspath(join(abspath(__file__), pardir)) + '/testfiles/'


class Test_mle_trev(unittest.TestCase):
    def test_mle_trev(self):
        C = np.loadtxt(testpath + 'C_1_lag.dat')

        T_impl_algo_sparse_type_sparse = impl_sparse(scipy.sparse.csr_matrix(C)).toarray()
        T_impl_algo_dense_type_dense = impl_dense(C)

        T_api_algo_dense_type_dense = apicall(C, reversible=True, method='dense')
        T_api_algo_sparse_type_dense = apicall(C, reversible=True, method='sparse')
        T_api_algo_dense_type_sparse = apicall(scipy.sparse.csr_matrix(C), reversible=True, method='dense').toarray()
        T_api_algo_sparse_type_sparse = apicall(scipy.sparse.csr_matrix(C), reversible=True, method='sparse').toarray()
        T_api_algo_auto_type_dense = apicall(C, reversible=True, method='auto')
        T_api_algo_auto_type_sparse = apicall(scipy.sparse.csr_matrix(C), reversible=True, method='auto').toarray()

        assert_allclose(T_impl_algo_sparse_type_sparse, T_impl_algo_dense_type_dense)
        assert_allclose(T_api_algo_dense_type_dense, T_impl_algo_dense_type_dense)
        assert_allclose(T_api_algo_sparse_type_dense, T_impl_algo_dense_type_dense)
        assert_allclose(T_api_algo_dense_type_sparse, T_impl_algo_dense_type_dense)
        assert_allclose(T_api_algo_sparse_type_sparse, T_impl_algo_dense_type_dense)
        assert_allclose(T_api_algo_auto_type_dense, T_impl_algo_dense_type_dense)
        assert_allclose(T_api_algo_auto_type_sparse, T_impl_algo_dense_type_dense)

    def test_warnings(self):
        C = np.loadtxt(testpath + 'C_1_lag.dat')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore')
            warnings.simplefilter('always', category=NotConvergedWarning)
            impl_sparse(scipy.sparse.csr_matrix(C), maxiter=1)
            assert issubclass(w[-1].category, NotConvergedWarning)

            impl_dense(C, maxiter=1)
            assert len(w) == 2
            assert issubclass(w[-1].category, NotConvergedWarning)

    def test_noninteger_counts_sparse(self):
        C = np.loadtxt(testpath + 'C_1_lag.dat')
        T_sparse_reference = impl_sparse(scipy.sparse.csr_matrix(C)).toarray()
        T_sparse_scaled_1 = impl_sparse(scipy.sparse.csr_matrix(C*10.0)).toarray()
        T_sparse_scaled_2 = impl_sparse(scipy.sparse.csr_matrix(C*0.1)).toarray()
        assert_allclose(T_sparse_reference, T_sparse_scaled_1)
        assert_allclose(T_sparse_reference, T_sparse_scaled_2)

    def test_noninteger_counts_dense(self):
        C = np.loadtxt(testpath + 'C_1_lag.dat')
        T_dense_reference = impl_dense(C)
        T_dense_scaled_1 = impl_dense(C*10.0)
        T_dense_scaled_2 = impl_dense(C*0.1)
        assert_allclose(T_dense_reference, T_dense_scaled_1)
        assert_allclose(T_dense_reference, T_dense_scaled_2)

    def test_return_statdist(self):
        C = np.loadtxt(testpath + 'C_1_lag.dat')
        # dense
        T, mu = apicall(C, reversible=True, method='dense', return_statdist=True)
        mu_manual = stationary_distribution(T)
        np.testing.assert_allclose(mu, mu_manual)
        # sparse
        T, mu = apicall(C, reversible=True, method='sparse', return_statdist=True)
        mu_manual = stationary_distribution(T)
        np.testing.assert_allclose(mu, mu_manual)
