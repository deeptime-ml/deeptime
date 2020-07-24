# TODO: - matrix sum along one axis
#       - vector-vector dot product

import numpy as np
cimport numpy as np

cdef extern from "_kahandot.h":
    void _kdot(
            double *A,
            double *B,
            double *C,
            size_t n,
            size_t m,
            size_t l
        )
    double _ksum(
            double *X,
            size_t n,
            size_t m
        )

def kdot(np.ndarray[double, ndim=2, mode="c"] A not None,
        np.ndarray[double, ndim=2, mode="c"] B not None):
    
    assert A.shape[1] == B.shape[0]
    
    C = np.zeros((A.shape[0],B.shape[1]),dtype=np.float64)
    _kdot(
        <double*> np.PyArray_DATA( A ),
        <double*> np.PyArray_DATA( B ),
        <double*> np.PyArray_DATA( C ),
        A.shape[0],
        A.shape[1],
        B.shape[1]
    )
    return C
        
def ksum(np.ndarray[double, ndim=2, mode="c"] X not None):
    return _ksum(
        <double*> np.PyArray_DATA( X ),
        X.shape[0],
        X.shape[1]
        )

