//
// Created by mho on 8/6/20.
//

#include "common.h"

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


PYBIND11_MODULE(kahandot, m) {
    m.def("kdot", &kdot<float>);
    m.def("kdot", &kdot<double>);
    m.def("kdot", &kdot<long double>);
    m.def("ksum", &ksumArr<float>);
    m.def("ksum", &ksumArr<double>);
    m.def("ksum", &ksumArr<long double>);
}
