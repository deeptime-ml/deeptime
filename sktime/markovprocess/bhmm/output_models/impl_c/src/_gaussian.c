#define _USE_MATH_DEFINES
#include <math.h>


double gaussian(double o, double mu, double sigma)
/* Returns the probability density of a Gaussian with given mu and sigma evaluated at o

    Parameters
    ----------
    o : double
        observation value
    mu : double
        mean value
    sigma : double
        standard deviation
*/
{
    double C = 1.0 / (sqrt(2.0 * M_PI) * sigma);
    double d = (o - mu) / sigma;
    return C * exp(-0.5 * d * d);
}

void _p_o(double o, double* mus, double* sigmas, int N, double* p)
/* Returns the output probability for symbol o from all hidden states

    Parameters
    ----------
    o : ptr to double array, size T
        observation sequence
    mus : ptr to double array, size N
        mean values
    sigmas : ptr to double array, size N
        standard deviations
    N : int
        number of states
    p : ptr to double array, size N
        output will be written here
*/
{
    int i;
    for (i=0; i<N; i++)
        p[i] = gaussian(o, mus[i], sigmas[i]);
}

void _p_obs(double* o, double* mus, double* sigmas, int N, int T, double* p)
/* Returns the output probability for symbol o from all hidden states

    Parameters
    ----------
    o : ptr to double array, size T
        observation sequence
    mus : ptr to double array, size N
        mean values
    sigmas : ptr to double array, size N
        standard deviations
    N : int
        number of states
    T : int
        number of trajectory steps
    p : ptr to double array, size N
        output will be written here
*/
{
    int i, t;
    for (t=0; t<T; t++)
        for (i=0; i<N; i++)
        {
            p[t*N + i] = gaussian(o[t], mus[i], sigmas[i]);
        }
}
