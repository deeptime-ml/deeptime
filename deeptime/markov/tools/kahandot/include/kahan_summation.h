//
// Created by Maaike on 18/11/2021.
//

#include "common.h"

/***************************************************************************************************
*   Kahan summation
***************************************************************************************************/
template<typename dtype>
auto ksum(const dtype* const begin, const dtype* const end) -> dtype {
    auto n = std::distance(begin, end);
    dtype sum {0};
    ssize_t o {0};
    dtype correction {0};

    while (n--) {
        auto y = begin[o] - correction;
        auto t = sum + y;
        correction = (t - sum) - y;
        sum = t;
        ++o;
    }

    return sum;
}

template<typename dtype>
auto ksumArr(const np_array_nfc<dtype> &Xarr) -> dtype {
    return ksum(Xarr.data(), Xarr.data() + Xarr.size());
}

/***************************************************************************************************
*   dot product of two matrices using Kahan summation scheme
***************************************************************************************************/
template<typename dtype>
auto kdot(const np_array_nfc<dtype> &arrA, const np_array_nfc<dtype> &arrB) -> np_array<dtype> {
    auto n = arrA.shape(0);
    auto m = arrA.shape(1);
    auto l = arrB.shape(1);

    if (m != arrB.shape(0)) {
        throw std::invalid_argument("Shape mismatch, A.shape[1] must match B.shape[0].");
    }

    auto A = arrA.template unchecked<2>();
    auto B = arrB.template unchecked<2>();

    auto Carr = np_array<dtype>({n, l});

    auto C = Carr.template mutable_unchecked<2>();

    for (ssize_t i = 0; i < n; ++i) {
        for (ssize_t j = 0; j < l; ++j) {
            dtype err{0};
            dtype sum{0};
            for (ssize_t k = 0; k < m; ++k) {
                auto y = A(i, k) * B(k, j) - err;
                auto t = sum + y;
                err = (t - sum) - y;
                sum = t;
            }
            C(i, j) = sum;
        }
    }

    return Carr;
}

/***************************************************************************************************
*   sorting
***************************************************************************************************/
template<typename dtype>
extern void _mixed_sort(dtype *array, int L, int R)
/* _mixed_sort() is based on examples from http://www.linux-related.de (2004) */
{
    int l, r;
    dtype swap;
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
*   logspace Kahan summation
***************************************************************************************************/
template<typename dtype>
extern double logsumexp_kahan_inplace(dtype *array, int size, dtype array_max)
{
    int i;
    if(0 == size) return -INFINITY;
    if(-INFINITY == array_max)
        return -INFINITY;
    for(i=0; i<size; ++i)
        array[i] = exp(array[i] - array_max);
    return array_max + log(ksum(array, array + size));
}

template<typename dtype>
extern double logsumexp_sort_kahan_inplace(dtype *array, int size)
{
    if(0 == size) return -INFINITY;
    _mixed_sort(array, 0, size - 1);
    return logsumexp_kahan_inplace(array, size, array[size - 1]);
}

template<typename dtype>
extern dtype logsumexp_pair(dtype a, dtype b)
{
    if((-INFINITY == a) && (-INFINITY == b))
        return -INFINITY;
    if(b > a)
        return b + log(1.0 + exp(a - b));
    return a + log(1.0 + exp(b - a));
}

