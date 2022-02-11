//
// Created by Maaike on 14/12/2021.
//

#pragma once

#include "shared_methods.h"

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
    TRAMInput(CountsMatrix &&stateCounts, CountsMatrix &&transitionCounts,
              DTraj dtraj, BiasMatrix<dtype> biasMatrix)
            : stateCounts_(std::move(stateCounts)),
              transitionCounts_(std::move(transitionCounts)),
              dtraj_(std::move(dtraj)),
              biasMatrix_(std::move(biasMatrix)) {
        validateInput();
    }

    TRAMInput() = default;

    TRAMInput(const TRAMInput &) = delete;

    TRAMInput &operator=(const TRAMInput &) = delete;

    TRAMInput(TRAMInput &&) noexcept = default;

    TRAMInput &operator=(TRAMInput &&) noexcept = default;

    ~TRAMInput() = default;

    void validateInput() const {

        if (dtraj_.shape(0) != biasMatrix_.shape(0)) {
            std::stringstream ss;
            ss << "Input invalid. Number of samples in dtrajs be equal to the size of the first dimension "
                  "of the bias matrix.";
            ss << "\nNumber of samples: " << dtraj_.shape(0) << "\nNumber of samples in bias matrix: "
               << biasMatrix_.shape(0);
            throw std::runtime_error(ss.str());
        }
        detail::throwIfInvalid(stateCounts_.shape(0) == transitionCounts_.shape(0),
                               "stateCounts.shape(0) should equal transitionCounts.shape(0)");
        detail::throwIfInvalid(stateCounts_.shape(1) == transitionCounts_.shape(1),
                               "stateCounts.shape(1) should equal transitionCounts.shape(1)");
        detail::throwIfInvalid(transitionCounts_.shape(1) == transitionCounts_.shape(2),
                               "transitionCounts.shape(1) should equal transitionCounts.shape(2)");

        detail::throwIfInvalid(dtraj_.ndim() == 1,
                               "dtraj has an incorrect number of dimension. ndims should be 1.");
        detail::throwIfInvalid(biasMatrix_.ndim() == 2,
                               "biasMatrix has an incorrect number of dimension. ndims should be 2.");
        detail::throwIfInvalid(biasMatrix_.shape(1) == transitionCounts_.shape(0),
                               "biasMatrix.shape[1] should be equal to transitionCounts.shape[0].");
    }

    auto & biasMatrix() const {
        return biasMatrix_;
    }

    auto biasMatrixBuf() const {
        return biasMatrix_.template unchecked<2>();
    }

    const auto & dtraj() const {
        return dtraj_;
    }

    const auto dtrajBuf() const {
        return dtraj_.template unchecked<1>();
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

    auto nSamples() const {
        return dtraj_.size();
    }


private:
    CountsMatrix stateCounts_;
    CountsMatrix transitionCounts_;
    DTraj dtraj_;
    BiasMatrix<dtype> biasMatrix_;
};

}
