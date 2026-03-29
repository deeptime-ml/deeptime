import numpy as np
cimport scipy.linalg.cython_lapack as scc

def eig_qr(A):
    """ Compute eigenvalues and eigenvectors of symmetric matrix A using symmetric tridiagonal QR-algorithm
     with implicit shifts. The matrix is first transformed to tridiagonal shape using lapack's dsytrd routine.
     Then, the tridiagonal QR-iteration is performed using lapack's dsteqr routine.

     Parameters:
     -----------
     A, ndarray (N, N):
        symmetric matrix.

     Returns:
     --------
     D, ndarray(N,)
        array of eigenvalues of A
     B, ndarray(N, N)
        array of eigenvectors of A.
    """

    # handle 1x1 case
    if np.size(A) == 1:  # size can handle 1x1 arrays and numbers
        return A*np.ones(1), np.ones((1, 1))

    # Definitions:
    array_args = dict(dtype=np.float64, order='F')
    cdef double[:,:] B = np.require(A, dtype=np.float64, requirements=('F', 'A'))
    cdef int n = A.shape[0], lda = A.shape[0], info, lwork = -1
    cdef char uplo = b"U"
    cdef double[:] D = np.zeros(n, **array_args)
    cdef double[:] E = np.zeros(n-1, **array_args)
    cdef double[:] Tau = np.zeros(n-1, **array_args)
    cdef double WorkFake # LAPACK writes back the optimal block size here, when lwork is -1.

    # Transform to tridiagonal shape:
    scc.dsytrd(&uplo, &n, &B[0, 0], &lda, &D[0], &E[0], &Tau[0], &WorkFake, &lwork, &info)
    assert info == 0, info
    lwork = <int>WorkFake
    cdef double[:] Work2 = np.zeros(lwork, **array_args)
    scc.dsytrd(&uplo, &n, &B[0, 0], &lda, &D[0], &E[0], &Tau[0], &Work2[0], &lwork, &info)
    assert info == 0, info
    del Work2

    # Extract transformation to tridiagonal shape:
    lwork = -1
    scc.dorgtr(&uplo, &n, &B[0, 0], &lda, &Tau[0], &WorkFake, &lwork, &info)
    assert info == 0, info
    lwork = <int>WorkFake
    cdef double[:] Work3 = np.zeros(lwork, **array_args)
    scc.dorgtr(&uplo, &n, &B[0, 0], &lda, &Tau[0], &Work3[0], &lwork, &info)
    assert info == 0, info
    del Tau, Work3

    # Run QR-iteration.
    cdef double[:] Work4 = np.zeros(max(1, 2*n - 2), **array_args)
    cdef char compz = b"V"
    scc.dsteqr(&compz, &n, &D[0], &E[0], &B[0, 0], &n, &Work4[0], &info)
    assert info == 0, info

    return np.asarray(D), np.asarray(B)
