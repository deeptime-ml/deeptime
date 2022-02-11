//
// Created by Maaike on 11/02/2022.
//

#pragma once

#include <deeptime/numeric/kahan_summation.h>
#include "tram_types.h"


namespace deeptime::markov::tram {


// Shift all energies by min(biasedConfEnergies_) so the energies don't drift to
// very large values.
template<typename dtype>
void shiftEnergiesToHaveZeroMinimum(ExchangeableArray<dtype, 1> &thermStateEnergies, StateIndex nThermStates) {
    auto thermStateEnergiesBuf = thermStateEnergies.firstBuf();
    auto newThermStateEnergies = thermStateEnergies.first();
    auto shift = *std::min_element(newThermStateEnergies->data(), newThermStateEnergies->data() + nThermStates);

    for (StateIndex k = 0; k < nThermStates; ++k) {
        thermStateEnergiesBuf(k) -= shift;
    }
}

// Get the error in the energies between this iteration and the previous one.
template<typename dtype>
dtype computeError(const ExchangeableArray<dtype, 1> &thermStateEnergies, StateIndex nThermStates) {
    auto thermEnergiesBuf = thermStateEnergies.firstBuf();
    auto oldThermEnergiesBuf = thermStateEnergies.secondBuf();

    dtype maxError = 0;

    for (StateIndex k = 0; k < nThermStates; ++k) {
        auto energyDelta = std::abs(thermEnergiesBuf(k) - oldThermEnergiesBuf(k));
        maxError = std::max(maxError, energyDelta);
    }
    return maxError;
}

template<typename dtype>
void selfConsistentUpdate(ExchangeableArray<dtype, 1> &thermStateEnergies,
                          const ArrayBuffer<BiasMatrix<dtype>, 2> &biasMatrixBuf,
                          const std::vector<dtype> &stateCountsLog, StateIndex nThermStates, std::size_t nSamples) {
    thermStateEnergies.exchange();
    auto newThermStateEnergies = thermStateEnergies.first();

    /* reset all therm_energies to INF */
    std::fill(newThermStateEnergies->mutable_data(),
              newThermStateEnergies->mutable_data() + newThermStateEnergies->size(),
              std::numeric_limits<dtype>::infinity());

    std::vector<dtype> scratch(nThermStates);

    auto newThermEnergies = thermStateEnergies.firstBuf();
    auto oldThermEnergies = thermStateEnergies.secondBuf();

    for (auto x = 0; x < nSamples; ++x) {
        for (auto l = 0; l < nThermStates; ++l) {
            scratch[l] = stateCountsLog[l] + oldThermEnergies[l] - biasMatrixBuf(x, l);
        }

        dtype divisor = numeric::kahan::logsumexp_sort_kahan_inplace(scratch.begin(), nThermStates);

        for (auto k = 0; k < nThermStates; ++k) {
            newThermEnergies[k] = -numeric::kahan::logsumexp_pair(-newThermEnergies[k], -(biasMatrixBuf(x, k) + divisor));
        }
    }
}

// initialize free energies using MBAR.
template<typename dtype>
np_array <dtype>
initialize_MBAR(BiasMatrix <dtype> biasMatrix, CountsMatrix stateCounts, std::size_t maxIter = 1000,
                double maxErr = 1e-6) {

    auto nThermStates = stateCounts.shape(0);
    auto stateCountsLog = std::vector<dtype>(nThermStates);

    std::transform(stateCounts.data(), stateCounts.data() + nThermStates, stateCountsLog.begin(),
                   [](const auto counts) { return std::log(counts); }
    );

    std::size_t nSamples = biasMatrix.shape(1);

    ExchangeableArray<dtype, 1> thermStateEnergies(std::vector<StateIndex>{nThermStates}, 0.);
    ArrayBuffer<BiasMatrix<dtype>, 2> biasMatrixBuf{biasMatrix};

    for (decltype(maxIter) iterationCount = 0; iterationCount < maxIter; ++iterationCount) {

        selfConsistentUpdate(thermStateEnergies, biasMatrixBuf, stateCountsLog, nThermStates, nSamples);

        auto iterationError = computeError(thermStateEnergies, nThermStates);

        if (iterationError < maxErr) {
            // We have converged!
            break;
        } else {
            // We are not finished. But before the next iteration, we shift all energies by min(energies)
            // so that the minimum energy equals zero (we are only interested in energy differences!).
            shiftEnergiesToHaveZeroMinimum(thermStateEnergies, nThermStates);
        }

    }
    return *thermStateEnergies.first();
}
}
