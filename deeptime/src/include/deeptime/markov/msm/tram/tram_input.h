//
// Created by Maaike on 14/12/2021.
//

#pragma once

#include "common.h"

namespace deeptime::markov::tram {

namespace detail {
constexpr void throwIfInvalid(bool isValid, const std::string &message) {
    if (!isValid) {
        throw std::runtime_error(message);
    }
}
}


template<typename dtype>
class TRAMInput {
public:
    using size_type = typename BiasMatrices<dtype>::size_type;

    TRAMInput(CountsMatrix &&stateCounts, CountsMatrix &&transitionCounts, BiasMatrices<dtype> biasMatrices)
            : stateCounts_(std::move(stateCounts)),
              transitionCounts_(std::move(transitionCounts)),
              biasMatrices_(std::move(biasMatrices)),
              cumNSamples_() {
        cumNSamples_.resize(nMarkovStates());
        for(std::size_t i = 1; i < cumNSamples_.size(); ++i) {
            cumNSamples_[i] += cumNSamples_[i-1] + nSamples(i-1);
        }
        validateInput();
    }

    TRAMInput() = default;

    TRAMInput(const TRAMInput &) = delete;

    TRAMInput &operator=(const TRAMInput &) = delete;

    TRAMInput(TRAMInput &&) noexcept = default;

    TRAMInput &operator=(TRAMInput &&) noexcept = default;

    ~TRAMInput() = default;

    void validateInput() const {
        detail::throwIfInvalid(stateCounts_.shape(0) == transitionCounts_.shape(0),
                               "stateCounts.shape(0) should equal transitionCounts.shape(0)");
        detail::throwIfInvalid(stateCounts_.shape(1) == transitionCounts_.shape(1),
                               "stateCounts.shape(1) should equal transitionCounts.shape(1)");
        detail::throwIfInvalid(transitionCounts_.shape(1) == transitionCounts_.shape(2),
                               "transitionCounts.shape(1) should equal transitionCounts.shape(2)");
        detail::throwIfInvalid(!biasMatrices_.empty(), "We need bias matrices.");
        std::for_each(begin(biasMatrices_), end(biasMatrices_), [nThermStates = stateCounts_.shape(0)](const auto &biasMatrix) {
            detail::throwIfInvalid(biasMatrix.ndim() == 2,
                                   "biasMatrix has an incorrect number of dimension. ndims should be 2.");
            detail::throwIfInvalid(biasMatrix.shape(1) == nThermStates,
                                   "biasMatrix.shape[1] should be equal to transitionCounts.shape[0].");
        });
    }

    const auto& biasMatrix(size_type i) const {
        return biasMatrices_[i];
    }

    const auto &biasMatrices() const {
        return biasMatrices_;
    }

    const auto& transitionCounts() const {
        return transitionCounts_;
    }

    auto transitionCountsBuf() const {
        return transitionCounts_.template unchecked<3>();
    }

    const auto& stateCounts() const {
        return stateCounts_;
    }

    auto stateCountsBuf() const {
        return stateCounts_.template unchecked<2>();
    }

    auto nSamples(size_type i) const {
        return biasMatrices_[i].shape(0);
    }

    auto nSamples() const {
        decltype(nSamples(0)) total {};
        for(size_type i = 0; i < biasMatrices_.size(); ++i) {
            total += nSamples(i);
        }
        return total;
    }

    const auto &cumNSamples() const {
        return cumNSamples_;
    }

    auto nThermStates() const {
        return transitionCounts_.shape(0);
    }

    auto nMarkovStates() const {
        return stateCounts_.shape(1);
    }


private:
    CountsMatrix stateCounts_;
    CountsMatrix transitionCounts_;
    BiasMatrices<dtype> biasMatrices_;
    std::vector<size_type> cumNSamples_;
};

}
