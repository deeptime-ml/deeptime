import numpy as np
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix

try:
    from slepc4py import SLEPc
    from petsc4py import PETSc

    _successful_import = True
except ModuleNotFoundError:
    _successful_import = False


def _handle_import_error():
    if not _successful_import:
        raise RuntimeError("In order to use this module a working installation of slepc4py and petsc4py is required.")


def _to_petsc_matrix(mat: np.ndarray):
    r""" Helper method which converts a two-dimensional numpy ndarray to a PETSc matrix.

    Parameters
    ----------
    mat : (n, m) ndarray
        Input matrix.

    Returns
    -------
    petsc_mat : petsc4py.PETSc.Mat
        PETSc matrix.
    """
    if not np.can_cast(mat, PETSc.ScalarType, 'safe'):
        raise ValueError(f"Cannot safely cast input to {PETSc.ScalarType}.")
    petsc_mat = PETSc.Mat()
    petsc_mat.create()
    if not issparse(mat):
        petsc_mat.setSizes(mat.shape)
        petsc_mat.setType("aij")
        petsc_mat.setUp()
        petsc_mat.setValues(np.arange(mat.shape[0], dtype=np.int32), np.arange(mat.shape[1], dtype=np.int32), mat)
    else:
        if not isspmatrix_csr(mat):
            mat = csr_matrix(mat)
        petsc_mat.createAIJ(size=mat.shape, csr=(mat.indptr, mat.indices, mat.data))
    petsc_mat.assemble()
    return petsc_mat


def svd(X, method='lanczos'):
    r""" Compute SVD using SLEPc.

    Parameters
    ----------
    X : (n, m) ndarray
        input matrix
    method : str, optional, default='lanczos'
        SVD solver to use, can be one of `'lanczos'`, `'cyclic'`, `'lapack'`, `'trlanczos'`, `'cross'`.

    Returns
    -------
    u, s, vh
    """
    _handle_import_error()
    assert method in svd._mapping.keys(), f'Illegal method, must be one of {list(svd._mapping.keys())}.'

    mat = _to_petsc_matrix(X)
    S = SLEPc.SVD()
    S.create()
    S.setOperator(mat)
    S.setFromOptions()
    S.setType(svd._mapping[method])
    S.setDimensions(nsv=min(X.shape))
    S.solve()

    nconv = S.getConverged()
    s = np.empty((nconv,), dtype=PETSc.ScalarType)
    U = np.empty((nconv, X.shape[0]), dtype=PETSc.ScalarType)
    Vh = np.empty((nconv, X.shape[1]), dtype=PETSc.ScalarType)

    if nconv > 0:
        for i in range(nconv):
            U_ = PETSc.Vec().createWithArray(U[i])
            V_ = PETSc.Vec().createWithArray(Vh[i])

            s[i] = S.getValue(i)
            S.getVectors(i, U_, V_)
    return U, s, Vh


if _successful_import:
    svd._mapping = dict(
        lanczos=SLEPc.SVD.Type.LANCZOS,
        cyclic=SLEPc.SVD.Type.CYCLIC,
        lapack=SLEPc.SVD.Type.LAPACK,
        trlanczos=SLEPc.SVD.Type.TRLANCZOS,
        cross=SLEPc.SVD.Type.CROSS
    )
else:
    svd._mapping = dict()
