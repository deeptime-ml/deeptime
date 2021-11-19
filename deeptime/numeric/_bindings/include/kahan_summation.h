//
// Created by Maaike on 18/11/2021.
//

#pragma once

#include <algorithm>
#include <cmath>

#include "common.h"

namespace deeptime {
namespace numeric {
namespace kahan {

/***************************************************************************************************
*   Kahan summation
***************************************************************************************************/
template<typename dtype>
auto ksum(const dtype *const begin, const dtype *const end) -> dtype {
    dtype sum{0};
    ssize_t o{0};
    dtype correction{0};

    for (auto n = std::distance(begin, end); n > 0; --n) {
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
*   logspace Kahan summation
***************************************************************************************************/
template<typename dtype>
dtype logsumexp_kahan_inplace(dtype *array, int size, dtype array_max) {
    if (0 == size) return -std::numeric_limits<dtype>::infinity();
    if (-std::numeric_limits<dtype>::infinity() == array_max)
        return -std::numeric_limits<dtype>::infinity();
    for (int i = 0; i < size; ++i)
        array[i] = std::exp(array[i] - array_max);
    return array_max + std::log(ksum(array, array + size));
}

template<typename dtype>
dtype logsumexp_sort_kahan_inplace(dtype *array, int size) {
    if (0 == size) return -std::numeric_limits<dtype>::infinity();
    std::sort(array, array + size);
    return logsumexp_kahan_inplace(array, size, array[size - 1]);
}

template<typename dtype>
dtype logsumexp_pair(dtype a, dtype b) {
    if ((-std::numeric_limits<dtype>::infinity() == a) && (-std::numeric_limits<dtype>::infinity() == b))
        return -std::numeric_limits<dtype>::infinity();
    if (b > a) {
        return b + std::log(1.0 + std::exp(a - b));
    } else {
        return a + std::log(1.0 + std::exp(b - a));
    }
}

}
}
}
