#include "_kahandot.h"
#include <math.h>
#include <float.h>

#define A(i,j) (A[(i)*m+(j)])
#define B(i,j) (B[(i)*l+(j)])
#define C(i,j) (C[(i)*l+(j)])

void _kdot(double *A, double *B, double *C, size_t n, size_t m, size_t l)
{
    size_t i,j,k;
    double err, sum, t, y;
    
    for(i=0; i<n; ++i) {
        for(j=0; j<l; ++j) {
            err = 0.0;
            sum = 0.0;
            for(k=0; k<m; ++k) {
                y = A(i,k)*B(k,j) - err;
                t = sum + y;
                err = (t - sum) - y;
                sum = t;
            }
            C(i,j) = sum;
        }
    }
}

#undef A
#undef B
#undef C

#define X(i,j) (X[(i)*m+(j)])

double _ksum(double *X, size_t n, size_t m)
{
    size_t i,j;
    double err, sum, t, y;

    err = 0.0;
    sum = 0.0;
    for(i=0; i<n; ++i) {
        for(j=0; j<m; ++j) {
            y = X(i,j) - err;
            t = sum + y;
            err = (t - sum) - y;
            sum = t;
        }
    }
    return sum;
}

#undef X
