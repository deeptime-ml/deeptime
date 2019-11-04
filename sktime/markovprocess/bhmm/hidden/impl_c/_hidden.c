#include "_hidden.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#ifndef __DIMS__
#define __DIMS__
#define DIMM2(arr, i, j)    arr[(i)*M + j]
#define DIM2(arr, i, j)     arr[(i)*N + j]
#define DIM3(arr, t, i , j) arr[(t)*N*N + (i)*N + j]
#define DIMM3(arr, t, i, j) arr[(t)*N*M + (i)*M + j]
#endif


double _forward(
        double *alpha,
        const double *A,
        const double *pobs,
        const double *pi,
        int N, int T)
{
    int i, j, t;
    double sum, logprob, scaling;

    // first alpha and scaling factors
    scaling = 0.0;
    for (i = 0; i < N; i++) {
        alpha[i]  = pi[i] * pobs[i];
        scaling += alpha[i];
    }

    // initialize likelihood
    logprob = log(scaling);

    // scale first alpha
    if (scaling != 0)
        for (i = 0; i < N; i++)
            alpha[i] /= scaling;

    // iterate trajectory
    for (t = 0; t < T-1; t++)
    {
        scaling = 0.0;
        // compute new alpha and scaling
        for (j = 0; j < N; j++)
        {
            sum = 0.0;
            for (i = 0; i < N; i++)
            {
                sum += alpha[t*N+i]*A[i*N+j];
            }
            alpha[(t+1)*N+j] = sum * pobs[(t+1)*N+j];
            scaling += alpha[(t+1)*N+j];
        }
        // scale this row
        if (scaling != 0)
            for (j = 0; j < N; j++)
                alpha[(t+1)*N+j] /= scaling;

        // update likelihood
        logprob += log(scaling);
    }

    return logprob;
}


void _backward(
        double *beta,
        const double *A,
        const double *pobs,
        int N, int T)
{
    int i, j, t;
    double sum, scaling;

    // first beta and scaling factors
    scaling = 0.0;
    for (i = 0; i < N; i++)
    {
        beta[(T-1)*N+i] = 1.0;
        scaling += beta[(T-1)*N+i];
    }

    // scale first beta
    for (i = 0; i < N; i++)
        beta[(T-1)*N+i] /= scaling;

    // iterate trajectory
    for (t = T-2; t >= 0; t--)
    {
        scaling = 0.0;
        // compute new beta and scaling
        for (i = 0; i < N; i++)
        {
            sum = 0.0;
            for (j = 0; j < N; j++)
            {
                sum += A[i*N+j] * pobs[(t+1)*N+j] * beta[(t+1)*N+j];
            }
            beta[t*N+i] = sum;
            scaling += sum;
        }
        // scale this row
        if (scaling != 0)
            for (j = 0; j < N; j++)
                beta[t*N+j] /= scaling;
    }
}


void _computeGamma(
        double *gamma,
        const double *alpha,
        const double *beta,
        int N, int T)
{
    int i, t;
    double sum;

    for (t = 0; t < T; t++) {
        sum = 0.0;
        for (i = 0; i < N; i++) {
            gamma[t*N+i] = alpha[t*N+i]*beta[t*N+i];
            sum += gamma[t*N+i];
        }
        for (i = 0; i < N; i++)
            gamma[t*N+i] /= sum;
    }
}

/*
void _compute_state_counts(
        double *state_counts,
        const double *gamma,
        int T, int N)
{
    int i, t;
    for (i = 0; i < N; i++) {
        state_counts[i] = 0.0;
        for (t = 0; t < T; t++)
            state_counts[i] += gamma[t*N+i];
    }
}*/


int _compute_transition_counts(
        double *transition_counts,
        const double *A,
        const double *pobs,
        const double *alpha,
        const double *beta,
        int N, int T)
{
    int i, j, t;
    double sum, *tmp;

    // initialize
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            transition_counts[i*N+j] = 0.0;

    tmp = (double*) malloc(N*N * sizeof(double));
    if (! tmp) {
        return _BHMM_ERR_NO_MEM;
    }
    for (t = 0; t < T-1; t++)
    {
        sum = 0.0;
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
            {
                tmp[i*N+j] = alpha[t*N+i] * A[i*N+j] * pobs[(t+1)*N+j] * beta[(t+1)*N+j];
                sum += tmp[i*N+j];
            }
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                transition_counts[i*N+j] += tmp[i*N+j] / sum;
    }
    free(tmp);
    return 0;
}


int argmax(double* v, int N)
{
    int i;
    int a = 0;
    double m = v[0];
    for (i = 1; i < N; i++)
    {
        if (v[i] > m)
        {
            a = i;
            m = v[i];
        }
    }
    return a;
}


int _compute_viterbi(
        int *path,
        const double *A,
        const double *pobs,
        const double *pi,
        int N, int T)
{
    int i, j, t, maxi, result;
    double sum;
    double *v, *vnext, *h, *vh;
    int* ptr;
    result = 0;
    // allocate v
    v = (double*) malloc(N * sizeof(double));
    vnext = (double*) malloc(N * sizeof(double));
    h = (double*) malloc(N * sizeof(double));

    // allocate ptr
    ptr = (int*) malloc(T*N * sizeof(int));

    if (! v || ! vnext || !h || ! ptr) {
        result = _BHMM_ERR_NO_MEM; // indicate no memory
        goto error;
    }

    // initialization of v
    sum = 0.0;
    for (i = 0; i < N; i++)
    {
        v[i] = pobs[i] * pi[i];
        sum += v[i];
    }
    // normalize
    for (i = 0; i < N; i++)
    {
        v[i] /= sum;
    }

    // iteration of v
    for (t = 1; t < T; t++)
    {
        sum = 0.0;
        for (j = 0; j < N; j++)
        {
            for (i = 0; i < N; i++)
            {
                h[i] = v[i] * A[i*N+j];
            }
            maxi = argmax(h, N);
            ptr[t*N + j] = maxi;
            vnext[j] = pobs[t*N + j] * v[maxi] * A[maxi*N+j];
            sum += vnext[j];
        }
        // normalize
        for (i = 0; i < N; i++)
        {
            vnext[i] /= sum;
        }
        // update v
        vh = v;
        v = vnext;
        vnext = vh;
    }

    // path reconstruction
    path[T-1] = argmax(v,N);
    for (t = T-2; t >= 0; t--)
    {
        path[t] = ptr[(t+1)*N+path[t+1]];
    }
    error:
    // free memory
    free(v);
    free(vnext);
    free(h);
    free(ptr);

    return result;
}

int _random_choice(const double* p, const int N)
{
    double dR = (double)rand();
    double dM = (double)RAND_MAX;
    double r = dR / (dM + 1.0);
    double s = 0.0;
    int i;
    for (i = 0; i < N; i++)
    {
        s += p[i];
        if (s >= r)
        {
            return(i);
        }
    }

    printf("ERROR: random select method could not select anything. Probably p is not normalized.\n");
    for (i = 0; i < N; i++)
    {
        printf("%i \t %f\n",i,p[i]);
    }
    exit(1);
}

void _normalize(double* v, const int N)
{
    int i;
    double s = 0.0;
    for (i = 0; i < N; i++)
    {
        s += v[i];
    }
    for (i = 0; i < N; i++)
    {
        v[i] /= s;
    }
}


int _sample_path(
        int *path,
        const double *alpha,
        const double *A,
        const double *pobs,
        const int N, const int T)
{
    // initialize variables
    int i,t;
    double* psel;
    psel = (double*) malloc(N * sizeof(double));
    if (! psel) {
        return _BHMM_ERR_NO_MEM;
    }

    // initialize random number generator
    srand(time(NULL));

    // Sample final state.
    //printf("t = %i ",(T-1));
    for (i = 0; i < N; i++)
    {
        psel[i] = alpha[(T-1)*N+i];
        //printf("%f\t",psel[i]);
    }
    //printf("\n");
    _normalize(psel, N);
    // Draw from this distribution.
    path[T-1] = _random_choice(psel, N);
    //printf(" drawn: %i\n",path[T-1]);

    // Work backwards from T-2 to 0.
    for (t = T-2; t >= 0; t--)
    {
        //printf("t = %i ",t);
        // Compute P(s_t = i | s_{t+1}..s_T).
        for (i = 0; i < N; i++)
        {
            psel[i] = alpha[t*N+i] * A[i*N+path[t+1]];
            //printf("%f\t",psel[i]);
        }
        //printf("\n");
        _normalize(psel, N);
        // Draw from this distribution.
        path[t] = _random_choice(psel, N);
        //printf(" drawn: %i\n",path[t]);
    }

    // free memory
    free(psel);
    return 0;
}

/*
void compute_transition_probabilities(
        double *xi,
        const double *A,
        const double *B,
        const short *O,
        const double *alpha,
        const double *beta,
        int N, int M, int T)
{
    int i, j, t;
    double sum;

    for (t = 0; t < T-1; t++) {
        sum = 0.0;
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++) {
                DIM3(xi, t, i, j) = alpha[t*N+i]*beta[(t+1)*N+j]*A[i*N+j]*B[j*M+O[t+1]];
                sum += DIM3(xi, t, i, j);
            }
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++) {
                DIM3(xi, t, i, j) /= sum;
            }
    }
}
*/
