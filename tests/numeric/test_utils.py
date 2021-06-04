import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_, assert_raises, assert_array_almost_equal, assert_almost_equal, assert_equal

from deeptime.numeric import is_diagonal_matrix, spd_eig, spd_inv, ZeroRankError, spd_inv_sqrt, spd_inv_split, \
    eig_corr, is_square_matrix, allclose_sparse, is_sorted, spd_truncated_svd


@pytest.mark.parametrize("dtype", [None, int, float, np.int32, np.uint8])
def test_is_sorted(dtype):
    data = [0, 0, 1, 1, 2, 3, 3]

    if dtype is not None:
        arr = np.array(data, dtype=dtype)
    else:
        arr = data
    assert_(is_sorted(arr, 'asc'))
    assert_(not is_sorted(arr, 'desc'))

    arr = arr[::-1]
    assert_(not is_sorted(arr, 'asc'))
    assert_(is_sorted(arr, 'desc'))


def test_is_diagonal_matrix():
    assert_(is_diagonal_matrix(np.diag([1, 2, 3, 4, 5])))
    assert_(not is_diagonal_matrix(np.array([[1, 2], [3, 4]])))


def test_is_square_matrix():
    assert_(is_square_matrix(np.ones((5, 5))))
    assert_(not is_square_matrix(np.ones((3, 5))))


@pytest.fixture
def spd_matrix():
    X = np.random.normal(size=(50, 3))
    mat = X @ X.T  # positive semi-definite with rank 3
    return mat


def test_spd_eig_invalid_inputs(spd_matrix):
    with assert_raises(ValueError):
        spd_eig(spd_matrix, method='...')

    # not symmetric
    with assert_raises(ValueError):
        spd_eig(np.arange(9).reshape(3, 3), check_sym=True)

    # zero rank
    with assert_raises(ZeroRankError):
        spd_eig(np.zeros((3, 3)))


@pytest.mark.parametrize('eps', [0, 1e-12, 1e-5], ids=lambda x: f"eps={x}")
@pytest.mark.parametrize('dim', [None, 5, 3], ids=lambda x: f"dim={x}")
def test_spd_truncated_svd(spd_matrix, eps, dim):
    sm, vm = spd_truncated_svd(spd_matrix, dim=dim, eps=eps)
    assert_(sm[0] >= sm[1] >= sm[2])
    assert_array_almost_equal(vm @ np.diag(sm) @ vm.T, spd_matrix)
    assert_array_almost_equal(vm.T @ vm, np.eye(3))


@pytest.mark.parametrize('epsilon', [1e-5, 1e-12], ids=lambda x: f"epsilon={x}")
@pytest.mark.parametrize('method', ['QR', 'schur'], ids=lambda x: f"method={x}")
@pytest.mark.parametrize('canonical_signs', [False, True], ids=lambda x: f"canonical_signs={x}")
def test_spd_eig(spd_matrix, epsilon, method, canonical_signs):
    sm, vm = spd_eig(spd_matrix, epsilon=epsilon, method=method, canonical_signs=canonical_signs)
    assert_(sm[0] >= sm[1] >= sm[2])
    assert_array_almost_equal(vm @ np.diag(sm) @ vm.T, spd_matrix)
    assert_array_almost_equal(vm.T @ vm, np.eye(3))
    if canonical_signs or True:
        # largest element in each column in vm should be positive
        for col in range(vm.shape[1]):
            assert_(np.max(vm[:, col]) > 0)


@pytest.mark.parametrize('epsilon', [1e-5, 1e-12], ids=lambda x: f"epsilon={x}")
@pytest.mark.parametrize('method', ['QR', 'schur'], ids=lambda x: f"method={x}")
def test_spd_inv(spd_matrix, epsilon, method):
    W = spd_inv(spd_matrix, epsilon=epsilon, method=method)
    sm, _ = spd_eig(spd_matrix)
    sminv, _ = spd_eig(W)
    assert_array_almost_equal(np.sort(sm), np.sort(1. / sminv))


def test_spd_inv_1d():
    with assert_raises(ZeroRankError):
        spd_inv(np.array([[1e-18]]), epsilon=1e-10)  # smaller than epsilon

    assert_almost_equal(spd_inv(np.array([[5]])), 1 / 5)


def test_spd_inv_sqrt_1d():
    W = np.array([[.5]])
    assert_almost_equal(spd_inv_sqrt(W).squeeze(), 1. / np.sqrt(.5))

    with assert_raises(ZeroRankError):
        spd_inv_sqrt(np.array([[.001]]), epsilon=.01)


@pytest.mark.parametrize('epsilon', [1e-5, 1e-12], ids=lambda x: f"epsilon={x}")
@pytest.mark.parametrize('method', ['QR', 'schur'], ids=lambda x: f"method={x}")
@pytest.mark.parametrize('return_rank', [True, False], ids=lambda x: f"return_rank={x}")
def test_spd_inv_sqrt(spd_matrix, epsilon, method, return_rank):
    M = spd_inv_sqrt(spd_matrix, epsilon=epsilon, method=method, return_rank=return_rank)
    if return_rank:
        rank = M[1]
        M = M[0]

        assert_equal(rank, 3)

    assert_array_almost_equal(M @ M.T, spd_inv(spd_matrix))


def test_spd_inv_splid_1d():
    W = np.array([[.5]])
    assert_almost_equal(spd_inv_split(W).squeeze(), 1. / np.sqrt(.5))

    with assert_raises(ZeroRankError):
        spd_inv_split(np.array([[.001]]), epsilon=.01)


@pytest.mark.parametrize('epsilon', [1e-5, 1e-12], ids=lambda x: f"epsilon={x}")
@pytest.mark.parametrize('method', ['QR', 'schur'], ids=lambda x: f"method={x}")
@pytest.mark.parametrize('canonical_signs', [False, True], ids=lambda x: f"canonical_signs={x}")
def test_spd_inv_split(spd_matrix, epsilon, method, canonical_signs):
    split = spd_inv_split(spd_matrix, epsilon=epsilon, method=method, canonical_signs=canonical_signs)
    spd_matrix_inv = split @ split.T
    sminv, _ = spd_eig(spd_matrix_inv)
    sm, _ = spd_eig(spd_matrix)
    assert_array_almost_equal(np.sort(sm), np.sort(1. / sminv)[-3:])
    if canonical_signs:
        for i in range(3):
            assert_(np.max(split[:, i]) > 0)


def test_spd_inv_split_nocutoff():
    x = np.random.normal(size=(5, 5))
    unitary = np.linalg.qr(x)[0]
    assert_array_almost_equal(unitary @ unitary.T, np.eye(5))
    spd = unitary @ np.diag([1., 2., 3., 4., 5.]) @ unitary.T
    w, _ = np.linalg.eigh(spd)
    L = spd_inv_split(spd, epsilon=0)
    spd_inv = L @ L.T
    assert_array_almost_equal(spd_inv, np.linalg.pinv(spd))


@pytest.mark.parametrize('epsilon', [1e-5, 1e-12], ids=lambda x: f"epsilon={x}")
@pytest.mark.parametrize('method', ['QR', 'schur'], ids=lambda x: f"method={x}")
@pytest.mark.parametrize('canonical_signs', [False, True], ids=lambda x: f"canonical_signs={x}")
@pytest.mark.parametrize('return_rank', [True, False], ids=lambda x: f"return_rank={x}")
@pytest.mark.parametrize('hermitian_ctt', [True, False], ids=lambda x: f"hermitian_ctt={x}")
def test_eig_corr(epsilon, method, canonical_signs, return_rank, hermitian_ctt):
    data = np.random.normal(size=(5000, 3))
    from deeptime.covariance import Covariance
    covariances = Covariance(lagtime=10, compute_c00=True, compute_ctt=True).fit(data).fetch_model()
    if not hermitian_ctt:
        covariances.cov_tt[0, 1] += 1e-6
        assert_(not np.allclose(covariances.cov_tt, covariances.cov_tt.T))
    out = eig_corr(covariances.cov_00, covariances.cov_tt, epsilon=epsilon, method=method,
                   canonical_signs=canonical_signs, return_rank=return_rank)
    eigenvalues = out[0]
    eigenvectors = out[1]
    if return_rank:
        rank = out[2]
        assert_equal(rank, len(eigenvalues))

    for r in range(len(out[0])):
        assert_array_almost_equal(covariances.cov_00 @ eigenvectors[r] * eigenvalues[r],
                                  covariances.cov_tt @ eigenvectors[r], decimal=2)


def test_allclose_sparse():
    A = sp.random(50, 50)
    B = sp.random(50, 51)
    assert allclose_sparse(A, A)
    assert not allclose_sparse(A, B)
