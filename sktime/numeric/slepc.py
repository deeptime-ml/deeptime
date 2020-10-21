import numpy as np
from slepc4py import SLEPc
from petsc4py import PETSc


def svd(X, method='lanczos'):
    assert method in svd._mapping.keys(), f'Illegal method, must be one of {list(svd._mapping.keys())}.'

    Ap = PETSc.Mat()
    Ap.create()
    Ap.setSizes(X.shape)
    Ap.setType("aij")
    Ap.setUp()
    Ap.setValues(np.arange(X.shape[0], dtype=np.int32), np.arange(X.shape[1], dtype=np.int32), X)
    Ap.assemble()

    S = SLEPc.SVD()
    S.create()
    S.setOperator(Ap)
    S.setFromOptions()
    S.setType(svd._mapping[method])
    S.setDimensions(nsv=min(X.shape))
    S.solve()

    # Print = PETSc.Sys.Print
    #
    # Print()
    # Print("******************************")
    # Print("*** SLEPc Solution Results ***")
    # Print("******************************")
    # Print()
    #
    # its = S.getIterationNumber()
    # Print("Number of iterations of the method: %d" % its)
    #
    # eps_type = S.getType()
    # Print("Solution method: %s" % eps_type)
    #
    # nev, ncv, mpd = S.getDimensions()
    # Print("Number of requested eigenvalues: %d" % nev)
    #
    # tol, maxit = S.getTolerances()
    # Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
    #
    # nconv = S.getConverged()
    # Print("Number of converged eigenpairs %d" % nconv)

    nconv = S.getConverged()
    s = np.empty((nconv,), dtype=X.dtype)
    U = np.empty((nconv, X.shape[0]))
    Vt = np.empty((nconv, X.shape[1]))

    if nconv > 0:
        for i in range(nconv):
            U_ = PETSc.Vec().createWithArray(U[i])
            V_ = PETSc.Vec().createWithArray(Vt[i])

            s[i] = S.getValue(i)
            S.getVectors(i, U_, V_)
    return U, s, Vt


svd._mapping = dict(
    lanczos=SLEPc.SVD.Type.LANCZOS,
    cyclic=SLEPc.SVD.Type.CYCLIC,
    lapack=SLEPc.SVD.Type.LAPACK,
    trlanczos=SLEPc.SVD.Type.TRLANCZOS,
    cross=SLEPc.SVD.Type.CROSS
)
