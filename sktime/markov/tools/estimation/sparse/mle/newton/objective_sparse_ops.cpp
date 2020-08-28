//
// Created by mho on 8/28/20.
//

#include "common.h"

template<typename dtype>
void convertImpl(ssize_t M, const np_array_nfc<dtype> &xArr, const np_array_nfc<dtype> &yArr,
                 const np_array_nfc<dtype> &dataArr, np_array_nfc<dtype> &nuArr,
                 np_array_nfc<dtype> &dataPArr, np_array_nfc<dtype> &diagPArr,
                 const np_array<std::int32_t> &indicesArr, const np_array<std::int32_t> &indptrArr) {
    auto x = xArr.template unchecked<1>();
    auto y = yArr.template unchecked<1>();
    auto data = dataArr.template unchecked<1>();

    auto dataP = dataPArr.template mutable_unchecked<1>();
    auto diagP = diagPArr.template mutable_unchecked<1>();

    auto indices = indicesArr.unchecked<1>();
    auto indptr = indptrArr.unchecked<1>();

    auto nu = nuArr.template mutable_unchecked<1>();

    // Loop over rows of Cs
    for (ssize_t k = 0; k < M; ++k) {
        nu(k) = std::exp(y(k));
        // Loop over nonzero entries in row of Cs
        for (std::int32_t l = indptr(k); l < indptr(k + 1); ++l) {
            // Column index of current element
            auto j = indices(l);

            if (k != j) {
                // Current element of Cs at (k, j)
                auto cs_kj = data(l);
                // Exponential of difference
                auto ekj = std::exp(y(k) - y(j));
                // Compute off diagonal element
                dataP(l) = cs_kj / (x(k) + x(j) * ekj);
                // Update diagonal element
                diagP(k) -= dataP(l);
            }
        }
        diagP(k) += 1.0;
    }

}

template<typename dtype>
void FImpl(ssize_t M, const np_array_nfc<dtype> &xArr, const np_array_nfc<dtype> &yArr,
           const np_array_nfc<dtype> &cArr, const np_array_nfc<dtype> &dataArr,
           np_array_nfc<dtype> &FvalArr,
           const np_array<std::int32_t> &indicesArr, const np_array<std::int32_t> &indptrArr) {
    auto x = xArr.template unchecked<1>();
    auto y = yArr.template unchecked<1>();

    auto indices = indicesArr.unchecked<1>();
    auto indptr = indptrArr.unchecked<1>();

    auto c = cArr.template unchecked<1>();
    auto data = dataArr.template unchecked<1>();
    auto Fval = FvalArr.template mutable_unchecked<1>();

    // Loop over rows of Cs
    for (ssize_t k = 0; k < M; ++k) {
        Fval(k) += 1.0;
        Fval(k + M) -= c(k);

        // Loop over nonzero entries in row of Cs
        for (std::int32_t l = indptr(k); l < indptr(k + 1); ++l) {
            // Column index of current element
            auto j = indices(l);
            // Current element of Cs at (k, j)
            auto cs_kj = data(l);
            // Exponential of difference
            auto ekj = std::exp(y(k) - y(j));
            // Update Fx
            Fval(k) += -cs_kj / (x(k) + x(j) * ekj);
            // Update Fy
            Fval(k + M) -= -cs_kj * x(j) / (x(k) / ekj + x(j));
        }

    }
}

template<typename dtype>
void dfImpl(ssize_t M, const np_array_nfc<dtype> &xArr, const np_array_nfc<dtype> &yArr,
            const np_array_nfc<dtype> &dataArr,
            np_array_nfc<dtype> &dataHxxArr, np_array_nfc<dtype> &dataHyyArr, np_array_nfc<dtype> &dataHyxArr,
            np_array_nfc<dtype> &diagDxxArr, np_array_nfc<dtype> &diagDyyArr, np_array_nfc<dtype> &diagDyxArr,
            const np_array<std::int32_t> &indicesArr, const np_array<std::int32_t> &indptrArr) {
    auto indices = indicesArr.unchecked<1>();
    auto indptr = indptrArr.unchecked<1>();
    auto x = xArr.template unchecked<1>();
    auto y = yArr.template unchecked<1>();
    auto data = dataArr.template unchecked<1>();

    auto diagDxx = diagDxxArr.template mutable_unchecked<1>();
    auto diagDyy = diagDyyArr.template mutable_unchecked<1>();
    auto diagDyx = diagDyxArr.template mutable_unchecked<1>();

    auto dataHxx = dataHxxArr.template mutable_unchecked<1>();
    auto dataHyy = dataHyyArr.template mutable_unchecked<1>();
    auto dataHyx = dataHyxArr.template mutable_unchecked<1>();


    // Loop over rows of Cs
    for (ssize_t k = 0; k < M; ++k) {
        // Loop over nonzero entries in row of Cs
        for (std::int32_t l = indptr(k); l < indptr(k + 1); ++l) {
            // Column index of current element
            auto j = indices(l);

            // Current element of Cs at (k, j)
            auto cs_kj = data(l);

            auto ekj = std::exp(y(k) - y(j));

            auto tmp1 = cs_kj / ((x(k) + x(j) * ekj) * (x(k) / ekj + x(j)));
            auto tmp2 = cs_kj / ((x(k) + x(j) * ekj) * (x(k) + x(j) * ekj));

            dataHxx(l) = tmp1;
            diagDxx(k) += tmp2;

            dataHyy(l) = tmp1 * x(k) * x(j);
            diagDyy(k) -= tmp1 * x(k) * x(j);

            dataHyx(l) = -tmp1 * x(k);
            diagDyx(k) += tmp1 * x(j);
        }
    }
}

PYBIND11_MODULE(objective_sparse_ops, m) {
    m.def("DFImpl", &dfImpl<float>);
    m.def("DFImpl", &dfImpl<double>);
    m.def("FImpl", &FImpl<float>);
    m.def("FImpl", &FImpl<double>);
    m.def("convertImpl", &convertImpl<float>);
    m.def("convertImpl", &convertImpl<double>);
}
