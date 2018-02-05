from __future__ import absolute_import
import unittest
import numpy as np
from .. import direct

__author__ = 'noe'


def sort_by_norm_and_imag_sign(evals, evecs):
    arr = np.zeros((len(evals),), dtype=[('mag', np.float64), ('sign', np.float64)])
    arr['mag'] = np.abs(evals)
    arr['sign'] = np.sign((np.imag(evals)))
    I = np.argsort(arr, order=['mag', 'sign'])[::-1]
    return evals[I], evecs[:, I]


class TestDirect(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_spd_inv_split(self):
        W = np.array([[1.0, 0.3, 0.2],
                      [0.3, 0.8, 0.5],
                      [0.2, 0.5, 0.9]])
        for method in ['QR', 'schur']:
            L = direct.spd_inv_split(W, method=method)
            # Test if decomposition is correct: inv(W) == L L.T
            assert np.allclose(np.dot(L, L.T), np.linalg.inv(W))
            # Test if matrices are orthogonal
            C = np.dot(L.T, L)
            assert np.max(np.abs(C - np.diag(np.diag(C)))) < 1e-12

        # Test if fails when given a nonsymmetric matrix
        W = np.array([[1.0, 0.2],
                      [0.3, 0.8]])
        with self.assertRaises(AssertionError):
            direct.spd_inv_split(W)

    def test_eig_corr(self):
        C0 = np.array([[1.0, 0.3, 0.2],
                       [0.3, 0.8, 0.5],
                       [0.2, 0.5, 0.9]])
        Ct_sym = np.array([[0.5, 0.1, 0.0],
                           [0.1, 0.3, 0.3],
                           [0.0, 0.3, 0.2]])
        Ct_nonsym = np.array([[0.5, 0.1, 0.3],
                              [0.1, 0.3, 0.3],
                              [0.0, 0.3, 0.2]])
        # reference solution
        import scipy
        for Ct in [Ct_sym, Ct_nonsym]:
            v0, R0 = scipy.linalg.eig(Ct, C0)
            v0, R0 = sort_by_norm_and_imag_sign(v0, R0)
            for method in ['QR', 'schur']:
                # Test correctness
                v, R = direct.eig_corr(C0, Ct, method=method)
                v, R = sort_by_norm_and_imag_sign(v, R)
                np.testing.assert_allclose(v0, v)  # eigenvalues equal?
                # eigenvectors equivalent?
                for i in range(R0.shape[1]):
                    np.testing.assert_allclose(R0[:, i] / R0[0, i], R[:, i] / R[0, i])
                # Test if eigenpair diagonalizes the Koopman matrix
                K = np.dot(np.linalg.inv(C0), Ct)
                np.testing.assert_allclose(K, R.dot(np.diag(v)).dot(np.linalg.inv(R)))


if __name__ == "__main__":
    unittest.main()
