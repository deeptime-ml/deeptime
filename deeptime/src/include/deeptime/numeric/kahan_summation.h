//
// Created by Maaike on 18/11/2021.
//

#pragma once

#include <algorithm>
#include <cmath>

#include "deeptime/common.h"

namespace deeptime::numeric::kahan {

/***************************************************************************************************
*   Kahan summation
***************************************************************************************************/
template<typename Iterator, typename dtype = typename std::iterator_traits<Iterator>::value_type,
        typename ntype = typename std::iterator_traits<Iterator>::difference_type>
auto ksum(Iterator begin, Iterator end) -> dtype {
    dtype sum{0};
    ntype o{0};
    dtype correction{0};

    for (auto n = std::distance(begin, end); n > 0; --n) {
        auto y = *std::next(begin, o) - correction;
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

    for (py::ssize_t i = 0; i < n; ++i) {
        for (py::ssize_t j = 0; j < l; ++j) {
            dtype err{0};
            dtype sum{0};
            for (py::ssize_t k = 0; k < m; ++k) {
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
template<typename Iterator, typename dtype = typename std::iterator_traits<Iterator>::value_type>
auto logsumexp_kahan_inplace(Iterator begin, Iterator end, dtype array_max) {
    if (begin == end) return -std::numeric_limits<dtype>::infinity();
    if (-std::numeric_limits<dtype>::infinity() == array_max)
        return -std::numeric_limits<dtype>::infinity();
    std::transform(begin, end, begin, [array_max](auto element) { return std::exp(element - array_max); });

    return array_max + std::log(ksum(begin, end));
}


template<typename Iterator>
auto logsumexp_sort_kahan_inplace(Iterator begin, Iterator end) {
    using dtype = typename std::iterator_traits<Iterator>::value_type;
    if (begin == end) return -std::numeric_limits<dtype>::infinity();
    std::sort(begin, end);
    return logsumexp_kahan_inplace(begin, end, *std::prev(end));
}

template<typename Iterator>
auto logsumexp_sort_kahan_inplace(Iterator begin, std::size_t size) {
    return logsumexp_sort_kahan_inplace(begin, std::next(begin, size));
}

template <typename dtype>
auto logsumexp(const np_array_nfc<dtype> &arr) -> dtype {
    std::vector<dtype> vec(arr.data(), arr.data() + arr.size());
    return logsumexp_sort_kahan_inplace(vec.begin(), vec.end());
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
