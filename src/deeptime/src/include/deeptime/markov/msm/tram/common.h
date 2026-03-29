//
// Created by Maaike on 14/12/2021.
//

#pragma once
#include "deeptime/common.h"
#include <deeptime/numeric/kahan_summation.h>

namespace deeptime::markov::tram{

using DTraj = np_array<std::int32_t>;
using DTrajs = std::vector<DTraj>;
using DiscreteState = DTraj::value_type;
using StateMap = std::map<DiscreteState, std::vector<std::size_t>>;

using TTrajs = DTrajs;

template<typename dtype>
using BiasMatrix = np_array_nfc<dtype>;

template<typename dtype>
using BiasMatrices = std::vector<BiasMatrix<dtype>>;

using StateIndex = py::ssize_t;

using CountsMatrix = np_array<std::int32_t>;

namespace detail {
template<py::ssize_t Dims, typename Array>
auto mutableBuf(Array &&array) {
    return array.template mutable_unchecked<Dims>();
}
}

template<typename dtype, py::ssize_t Dims>
class ExchangeableArray {
    // This is used as a helper class in TRAM for the case where a value updated, and for each update, the current and
    // previous values are stored. new values are generally computed in the first buffer, and to update the values,
    // exchange() is called, so that the first buffer becomes the second, and the second buffer the first, which can
    // subsequently be overwritten with the updated values. See e.g. use of thermStateEnergies_ in TRAM.
    using MutableBufferType = decltype(detail::mutableBuf<Dims>(std::declval<np_array < dtype>>()));
public:
    template<typename Shape = std::vector<py::ssize_t>>
    ExchangeableArray(Shape shape, dtype fillValue) : arrays(
            std::make_tuple(np_array<dtype>(shape), np_array<dtype>(shape))) {
        std::fill(std::get<0>(arrays).mutable_data(), std::get<0>(arrays).mutable_data() + std::get<0>(arrays).size(),
                  fillValue);
        std::fill(std::get<1>(arrays).mutable_data(), std::get<1>(arrays).mutable_data() + std::get<1>(arrays).size(),
                  fillValue);
        buffers = std::make_tuple(
                std::make_unique<MutableBufferType>(std::get<0>(arrays).template mutable_unchecked<Dims>()),
                std::make_unique<MutableBufferType>(std::get<1>(arrays).template mutable_unchecked<Dims>())
        );
    }

    ExchangeableArray(const ExchangeableArray &) = delete;

    ExchangeableArray &operator=(const ExchangeableArray &) = delete;

    void exchange() {
        current = !current;
    }

    auto *first() {
        return current ? &std::get<0>(arrays) : &std::get<1>(arrays);
    }

    const auto *first() const {
        return current ? &std::get<0>(arrays) : &std::get<1>(arrays);
    }

    auto *second() {
        return current ? &std::get<1>(arrays) : &std::get<0>(arrays);
    }

    const auto *second() const {
        return current ? &std::get<1>(arrays) : &std::get<0>(arrays);
    }

    auto &firstBuf() {
        return current ? *std::get<0>(buffers) : *std::get<1>(buffers);
    }

    const auto &firstBuf() const {
        return current ? *std::get<0>(buffers) : *std::get<1>(buffers);
    }

    auto &secondBuf() {
        return current ? *std::get<1>(buffers) : *std::get<0>(buffers);
    }

    const auto &secondBuf() const {
        return current ? *std::get<1>(buffers) : *std::get<0>(buffers);
    }

private:
    char current = 0;
    std::tuple<np_array < dtype>, np_array <dtype>> arrays;
    std::tuple<std::unique_ptr<MutableBufferType>, std::unique_ptr<MutableBufferType>> buffers;
};

// Get the error in the energies between this iteration and the previous one.
template<typename dtype, auto ndims>
dtype computeError(const ExchangeableArray<dtype, ndims> &exchangeableArray, StateIndex bufferSize) {
    const auto* newBuf = exchangeableArray.first()->data();
    const auto* oldBuf = exchangeableArray.second()->data();

    dtype maxError = 0;

    for (auto k = 0; k < bufferSize; ++k) {
        auto energyDelta = std::abs(newBuf[k] - oldBuf[k]);
        maxError = std::max(maxError, energyDelta);
    }
    return maxError;
}

}
