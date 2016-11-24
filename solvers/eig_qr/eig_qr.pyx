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
    cdef double[:,:] B = np.require(A, dtype=np.float64, requirements=["F", "A"])
    cdef int n=A.shape[0], lda=A.shape[0], info, lwork=-1
    cdef char[:] uplo = np.zeros(1, "S1")
    uplo[:] = "U"
    cdef double[:] D = np.require(np.zeros(n), dtype=np.float64, requirements=["F", "A"])
    cdef double[:] E = np.require(np.zeros(n-1), dtype=np.float64, requirements=["F", "A"])
    cdef double[:] Tau = np.require(np.zeros(n-1), dtype=np.float64, requirements=["F", "A"])
    cdef double[:] Work = np.require(np.zeros(1), dtype=np.float64, requirements=["F", "A"])

    # Transform to tridiagonal shape:
    scc.dsytrd(&uplo[0], &n, &B[0, 0], &lda, &D[0], &E[0], &Tau[0], &Work[0], &lwork, &info)
    lwork = np.int(Work[0])
    cdef double[:] Work2 = np.require(np.zeros(lwork), dtype=np.float64, requirements=["F", "A"])
    scc.dsytrd(&uplo[0], &n, &B[0, 0], &lda, &D[0], &E[0], &Tau[0], &Work2[0], &lwork, &info)

    # Extract transformation to tridiagonal shape:
    lwork = -1
    scc.dorgtr(&uplo[0], &n, &B[0, 0], &lda, &Tau[0], &Work[0], &lwork, &info)
    lwork = np.int(Work[0])
    cdef double[:] Work3 = np.require(np.zeros(lwork), dtype=np.float64, requirements=["F", "A"])
    scc.dorgtr(&uplo[0], &n, &B[0, 0], &lda, &Tau[0], &Work3[0], &lwork, &info)

    # Run QR-iteration.
    cdef double[:] Work4 = np.require(np.zeros(np.maximum(1,2*n-2)), dtype=np.float64, requirements=["F", "A"])
    cdef char[:] compz = np.zeros(1, "S1")
    compz[:] = "V"
    scc.dsteqr(&compz[0], &n, &D[0], &E[0], &B[0, 0], &n, &Work4[0], &info)

    return np.asarray(D), np.asarray(B)




