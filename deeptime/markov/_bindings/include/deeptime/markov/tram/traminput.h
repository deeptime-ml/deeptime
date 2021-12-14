//
// Created by Maaike on 14/12/2021.
//

#pragma once

#include "typedef.h"

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
    TRAMInput(np_array_nfc<std::int32_t> &&stateCounts, np_array_nfc<std::int32_t> &&transitionCounts,
              DTrajs dtrajs, BiasMatrices<dtype> biasMatrices)
            : stateCounts_(std::move(stateCounts)),
              transitionCounts_(std::move(transitionCounts)),
              dtrajs_(std::move(dtrajs)),
              biasMatrices_(std::move(biasMatrices)) {
        validateInput();
    }

    TRAMInput() = default;

    TRAMInput(const TRAMInput &) = delete;

    TRAMInput &operator=(const TRAMInput &) = delete;

    TRAMInput(TRAMInput &&) noexcept = default;

    TRAMInput &operator=(TRAMInput &&) noexcept = default;

    ~TRAMInput() = default;

    void validateInput() const {

        if (dtrajs_.size() != biasMatrices_.size()) {
            std::stringstream ss;
            ss << "Input invalid. Number of trajectories should be equal to the size of the first dimension "
                  "of the bias matrix.";
            ss << "\nNumber of trajectories: " << dtrajs_.size() << "\nNumber of bias matrices: "
               << biasMatrices_.size();
            throw std::runtime_error(ss.str());
        }
        detail::throwIfInvalid(stateCounts_.shape(0) == transitionCounts_.shape(0),
                               "stateCounts.shape(0) should equal transitionCounts.shape(0)");
        detail::throwIfInvalid(stateCounts_.shape(1) == transitionCounts_.shape(1),
                               "stateCounts.shape(1) should equal transitionCounts.shape(1)");
        detail::throwIfInvalid(transitionCounts_.shape(1) == transitionCounts_.shape(2),
                               "transitionCounts.shape(1) should equal transitionCounts.shape(2)");

        for (std::size_t i = 0; i < dtrajs_.size(); ++i) {
            const auto &dtraj = dtrajs_.at(i);
            const auto &biasMatrix = biasMatrices_.at(i);

            detail::throwIfInvalid(dtraj.ndim() == 1,
                                   "dtraj at index {i} has an incorrect number of dimension. ndims should be 1.");
            detail::throwIfInvalid(biasMatrix.ndim() == 2,
                                   "biasMatrix at index {i} has an incorrect number of dimension. ndims should be 2.");
            detail::throwIfInvalid(dtraj.shape(0) == biasMatrix.shape(0),
                                   "dtraj and biasMatrix at index {i} should be of equal length.");
            detail::throwIfInvalid(biasMatrix.shape(1) == transitionCounts_.shape(0),
                                   "biasMatrix{i}.shape[1] should be equal to transitionCounts.shape[0].");
        }
    }

    auto biasMatrix(std::size_t i) const {
        return biasMatrices_.at(i).template unchecked<2>();
    }

    auto dtraj(std::size_t i) const {
        return dtrajs_[i].template unchecked<1>();
    }

    auto transitionCounts() const {
        return transitionCounts_.template unchecked<3>();
    }

    auto stateCounts() const {
        return stateCounts_.template unchecked<2>();
    }

    auto sequenceLength(std::size_t i) const {
        return dtrajs_[i].size();
    }

    auto nTrajectories() const {
        return dtrajs_.size();
    };

private:
    np_array_nfc<std::int32_t> stateCounts_;
    np_array_nfc<std::int32_t> transitionCounts_;
    DTrajs dtrajs_;
    BiasMatrices<dtype> biasMatrices_;
};

}