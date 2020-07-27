
/* * This file is part of MSMTools.
 *
 * Copyright (c) 2015, 2014 Computational Molecular Biology Group
 *
 * MSMTools is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/* * moduleauthor:: F. Noe <frank DOT noe AT fu-berlin DOT de>
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "rnglib/ranlib.h"
#include "sample_rev.h"
#include "util.h"

#define _square(x) x*x


int _accept_step(double log_prob_old, double log_prob_new)
{
    if (log_prob_new > log_prob_old)  // this is faster
        return 1;
    if (genunf(0,1) < exp( my_fmin( log_prob_new-log_prob_old, 0 ) ))
        return 1;
    return 0;
}

/*
    use random_walk_stepsize = 1 by default
*/
double _update_step(double v0, double v1, double v2, double c0, double c1, double c2, int random_walk_stepsize)
{
    /*
    update the sample v0 according to
    the distribution v0^(c0-1)*(v0+v1)^(-c1)*(v0+v2)^(-c2)

    :param v0:
    :param v1:
    :param v2:
    :param c0:
    :param c1:
    :param c2:
    :param random_walk_stepsize:
    :return:
    */
    double a = c1 + c2 - c0;
    double b = (c1 - c0) * v2 + (c2 - c0) * v1;
    double c = -c0 * v1 * v2;
    double v_bar = 0.5 * ( -b + sqrt( b * b - 4 * a * c ) ) / a;
    double h = c1 / _square((v_bar + v1) ) + c2 / _square((v_bar + v2)) - c0 / _square(v_bar);
    double k = -h * v_bar * v_bar;
    double theta = -1.0 / ( h * v_bar );
    //
    double log_v0 = log(v0);
    double v0_new = 0.0;
    double log_v0_new = 0.0;
    double log_prob_old = 0.0;
    double log_prob_new = 0.0;

    // about 1.5 sec: gamma and normf generation
    // about 1 sec: logs+exps in else blocks

    if (is_positive(k) && is_positive(theta))
    {
        v0_new = gengam(1.0/theta,k);
        log_v0_new = log(v0_new);
        if (is_positive(v0_new))
        {
            if (v0 == 0)
            {
                v0 = v0_new;
                log_v0 = log_v0_new;
            }
            else
            {
                log_prob_new = (c0-1) * log_v0_new - c1 * log(v0_new+v1) - c2 * log(v0_new+v2);
                log_prob_new -= (k-1) * log_v0_new - v0_new/theta;
                log_prob_old = (c0-1) * log_v0 - c1 * log(v0+v1) - c2 * log(v0+v2);
                log_prob_old -= (k-1) * log_v0 - v0/theta;
                if (_accept_step(log_prob_old, log_prob_new))
                //if (genunf(0,1) < exp( my_fmin( log_prob_new-log_prob_old, 0 ) ))
                {
                    v0 = v0_new;
                    log_v0 = log_v0_new;
                }
            }
        }
    }

    v0_new = v0 * exp( random_walk_stepsize * snorm() );
    log_v0_new = log(v0_new);
    if (is_positive(v0_new))
    {
        if (v0 == 0)
        {
            v0 = v0_new;
            log_v0 = log_v0_new;
        }
        else
        {
            log_prob_new = c0 * log_v0_new - c1 * log(v0_new + v1) - c2 * log(v0_new + v2);
            log_prob_old = c0 * log_v0 - c1 * log(v0 + v1) - c2 * log(v0 + v2);
            if (_accept_step(log_prob_old, log_prob_new))
            //if (genunf(0,1) < exp( my_fmin( log_prob_new-log_prob_old, 0 ) ))
            {
                v0 = v0_new;
                log_v0 = log_v0_new;
            }
        }
    }

    return v0;
}

double _sum_all(double* X, int n)
{
    int i,j;
    double sum=0.0;
    for (i=0; i<n; i++)
        for (j=0; j<n; j++)
            sum += X[i*n + j];
    return sum;
}

void _normalize_all(double* X, int n)
{
    int i,j;
    double sum = _sum_all(X, n);
    for (i=0; i<n; i++)
        for (j=0; j<n; j++)
            X[i*n+j] /= sum;
}

void _normalize_all_sparse(double* X, int* I, int* J, int n, int n_idx)
{
    int k;
    // sum all
    double sum=0.0;
    for (k=0; k<n_idx; k++)
        sum += X[I[k]*n + J[k]];
    // normalize all
    for (k=0; k<n_idx; k++)
        X[I[k]*n + J[k]] /= sum;
}

double _sum_row(double* X, int n, int i)
{
    int j;
    double sum=0.0;
    for (j=0; j<n; j++)
        sum += X[i*n + j];
    return sum;
}

double _sum_row_sparse(double* X, int n, int i, int* J, int from, int to)
{
    int j;
    double sum=0.0;
    for (j=from; j<to; j++)
        sum += X[i*n + J[j]];
    return sum;
}

void _print_matrix(double* X, int n)
{
    int i,j;
    for (i=0; i<n; i++)
    {
        for (j=0; j<n; j++)
            printf("%f \t",X[i*n+j]);
        printf("\n");
    }
}

void _print_array(double* X, int n)
{
    int i;
    for (i=0; i<n; i++)
    {
        printf("%f \t",X[i]);
    }
    printf("\n");
}

/**
Gibbs sampler for reversible transiton matrix
Output: sample_mem, sample_mem[i]=eval_fun(i-th sample of transition matrix)

Parameters:
-----------
    n : int
        number of states
    n_step : int
        the number of sampling steps made before returning a new transition matrix. In each sampling step, all
        transition matrix elements are updated.

*/
void _update(double* C, double* sumC, double* X, int n, int n_step)
{
    int iter, i, j;
    double tmp1, tmp2;

    /*printf("_update ...a\n");
    printf("C = \n");
    _print_matrix(C, n);
    printf("sumC = ");
    _print_array(sumC, n);
    printf("X = \n");
    _print_matrix(X, n);
    printf("\n");*/

    for (iter = 0; iter < n_step; iter++)
    {
        for (i = 0; i < n; i++)
        {
            for (j = 0; j <= i; j++)
            {
                if (C[i*n+j] + C[j*n+i] > 0)
                {
                    if (i == j)
                    {
                        if (is_positive(C[i*n+i]) && is_positive(sumC[i] - C[i*n+i]))
                        {
                            tmp1 = genbet(C[i*n+i], sumC[i] - C[i*n+i]);
                            tmp2 = tmp1 / (1-tmp1) * (_sum_row(X, n, i) - X[i*n+i]);
                            if (is_positive(tmp2))
                            {
                                X[i*n+i] = tmp2;
                            }
                        }
                    }
                    else
                    {
                        tmp1 = _sum_row(X, n, i) - X[i*n+j];
                        tmp2 = _sum_row(X, n, j) - X[j*n+i];
                        X[i*n+j] = _update_step(X[i*n+j], tmp1, tmp2, C[i*n+j]+C[j*n+i], sumC[i], sumC[j], 1);
                        X[j*n+i] = X[i*n+j];
                    }
                }
            }
        }

        /*printf("X = \n");
        _print_matrix(X, n);
        printf("\n");*/

        _normalize_all(X, n);
    }

    /*printf("X = \n");
    _print_matrix(X, n);
    printf("\n");*/

}

void _generate_row_indexes(int* I, int n, int n_idx, int* row_indexes)
{
    int k;
    int current_row;
    row_indexes[0] = 0;  // starts with row 0
    current_row = 0;
    for (k=0; k<n_idx; k++)
    {
        // still at same row? do nothing
        if (I[k] == current_row)
            continue;
        // row has advanced one or multiple times. Update multiple row indexes until we are equal
        while (I[k] > current_row)
        {
            current_row++;
            row_indexes[current_row] = k;
        }
    }
    // stop sign
    row_indexes[n] = n_idx;
}


/**
Gibbs sampler for reversible transition matrix
Output: sample_mem, sample_mem[i]=eval_fun(i-th sample of transition matrix)

Parameters:
-----------
    C : double[][]
        count matrix
    sumC : double[]
        sum of row counts
    I : int[]
        row indexes to update
    J : int[]
        column indexes to update
    n_idx : int
        number of indexes in I and J
    n : int
        number of states
    n_step : int
        the number of sampling steps made before returning a new transition matrix. In each sampling step, all
        transition matrix elements are updated.

*/
void _update_sparse(double* C, double* sumC, double* X, double* sumX, int* I, int* J, int n, int n_idx, int n_step)
{
    int iter, i, j, k;
    double tmp1, tmp2;

    // row indexes
    int* row_indexes = (int*) malloc((n+1) * sizeof(int));
    _generate_row_indexes(I, n, n_idx, row_indexes);

    for (iter = 0; iter < n_step; iter++)
    {
        // update all X row sums once every iteration and then only do cheap updates.
        for (i = 0; i < n; i++)
            sumX[i] = _sum_row_sparse(X, n, i, J, row_indexes[i], row_indexes[i+1]);

        for (k = 0; k < n_idx; k++)
        {
            i = I[k];
            j = J[k];
            if (i == j)
            {
                if (is_positive(C[i*n+i]) && is_positive(sumC[i] - C[i*n+i]))
                {
                    tmp1 = genbet(C[i*n+i], sumC[i] - C[i*n+i]);
                    tmp2 = tmp1 / (1-tmp1) * (sumX[i] - X[i*n+i]);
                    if (is_positive(tmp2))
                    {
                        sumX[i] += tmp2 - X[i*n+i];  // update sumX
                        X[i*n+i] = tmp2;
                    }
                }
            }
            if (i < j)  // only work on the upper triangle, because we have symmetry.
            {
                tmp1 = sumX[i] - X[i*n+j];
                tmp2 = sumX[j] - X[j*n+i];
                X[i*n+j] = _update_step(X[i*n+j], tmp1, tmp2, C[i*n+j]+C[j*n+i], sumC[i], sumC[j], 1);
                X[j*n+i] = X[i*n+j];
                // update X
                sumX[i] = tmp1 + X[i*n+j];
                sumX[j] = tmp2 + X[j*n+i];
            }
        }

        _normalize_all_sparse(X, I, J, n, n_idx);
    }

    // clean up
    free(row_indexes);

}


void _update_sparse_speedtest(double* C, double* sumC, double* X, double* sumX, int* I, int* J, int n, int n_idx, int n_step)
{
    int iter;
    int k;
    int i = 0;
    int j = 3;
    double tmp1 = _sum_row(X, n, i) - X[i*n+j];
    double tmp2 = _sum_row(X, n, j) - X[j*n+i];

    for (iter=0; iter<n_step; iter++)
    {
        for (k = 0; k < n_idx; k++)
        {
            _update_step(X[i*n+j], tmp1, tmp2, C[i*n+j]+C[j*n+i], sumC[i], sumC[j], 1);
        }
    }
}
