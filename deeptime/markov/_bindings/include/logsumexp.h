//
// Created by Maaike on 15/11/2021.
//

#include "common.h"

/***************************************************************************************************
*   sorting
***************************************************************************************************/

extern void _mixed_sort(double *array, int L, int R)
/* _mixed_sort() is based on examples from http://www.linux-related.de (2004) */
{
    int l, r;
    double swap;
    if(R - L > 25) /* use quicksort */
    {
        l = L - 1;
        r = R;
        for(;;)
        {
            while(array[++l] < array[R]);
            while((array[--r] > array[R]) && (r > l));
            if(l >= r) break;
            swap = array[l];
            array[l] = array[r];
            array[r] = swap;
        }
        swap = array[l];
        array[l] = array[R];
        array[R] = swap;
        _mixed_sort(array, L, l - 1);
        _mixed_sort(array, l + 1, R);
    }
    else /* use insertion sort */
    {
        for(l=L+1; l<=R; ++l)
        {
            swap = array[l];
            for(r=l-1; (r >= L) && (swap < array[r]); --r)
                array[r + 1] = array[r];
            array[r + 1] = swap;
        }
    }
}


/***************************************************************************************************
*   direct summation schemes
***************************************************************************************************/

extern void _kahan_summation_step(
        double new_value, double *sum, double *err, double *loc, double *tmp)
{
    *loc = new_value - *err;
    *tmp = *sum + *loc;
    *err = (*tmp - *sum) - *loc;
    *sum = *tmp;
}

extern double _kahan_summation(double *array, int size)
{
    int i;
    double sum = 0.0, err = 0.0, loc, tmp;
    for(i=0; i<size; ++i)
    {
        loc = array[i] - err;
        tmp = sum + loc;
        err = (tmp - sum) - loc;
        sum = tmp;
    }
    return sum;
}

/***************************************************************************************************
*   logspace summation schemes
***************************************************************************************************/

extern double _logsumexp(double *array, int size, double array_max)
{
    int i;
    double sum = 0.0;
    if(0 == size) return -INFINITY;
    if(-INFINITY == array_max)
        return -INFINITY;
    for(i=0; i<size; ++i)
        sum += exp(array[i] - array_max);
    return array_max + log(sum);
}

extern double _logsumexp_kahan_inplace(double *array, int size, double array_max)
{
    int i;
    if(0 == size) return -INFINITY;
    if(-INFINITY == array_max)
        return -INFINITY;
    for(i=0; i<size; ++i)
        array[i] = exp(array[i] - array_max);
    return array_max + log(_kahan_summation(array, size));
}

extern double _logsumexp_sort_inplace(double *array, int size)
{
    if(0 == size) return -INFINITY;
    _mixed_sort(array, 0, size - 1);
    return _logsumexp(array, size, array[size - 1]);
}

extern double _logsumexp_sort_kahan_inplace(double *array, int size)
{
    if(0 == size) return -INFINITY;
    _mixed_sort(array, 0, size - 1);
    return _logsumexp_kahan_inplace(array, size, array[size - 1]);
}

extern double _logsumexp_pair(double a, double b)
{
    if((-INFINITY == a) && (-INFINITY == b))
        return -INFINITY;
    if(b > a)
        return b + log(1.0 + exp(a - b));
    return a + log(1.0 + exp(b - a));
}