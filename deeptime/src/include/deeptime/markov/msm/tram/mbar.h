//
// Created by Maaike on 11/02/2022.
//

#pragma once

#include "common.h"

using namespace pybind11::literals;
namespace deeptime::markov::tram {


// Shift all energies by min(biasedConfEnergies_) so the energies don't drift to
// very large values.
template<typename dtype>
void shiftEnergiesToHaveZeroMinimum(ExchangeableArray<dtype, 1> &thermStateEnergies, StateIndex nThermStates) {
    auto thermStateEnergiesBuf = thermStateEnergies.firstBuf();
    auto newThermStateEnergies = thermStateEnergies.first();

    if (nThermStates > 0) {
        auto shift = *std::min_element(newThermStateEnergies->data(), newThermStateEnergies->data() + nThermStates);

        for (StateIndex k = 0; k < nThermStates; ++k) {
            thermStateEnergiesBuf(k) -= shift;
        }
    }
}

template<typename dtype, typename Size>
void selfConsistentUpdate(ExchangeableArray<dtype, 1> &thermStateEnergies,
                          const ArrayBuffer<BiasMatrix<dtype>, 2> &biasMatrixBuf,
                          const std::vector<dtype> &stateCountsLog, StateIndex nThermStates, Size nSamples) {
    thermStateEnergies.exchange();
    auto newThermStateEnergies = thermStateEnergies.first();

    /* reset all therm_energies to INF */
    std::fill(newThermStateEnergies->mutable_data(),
              newThermStateEnergies->mutable_data() + newThermStateEnergies->size(),
              std::numeric_limits<dtype>::infinity());

    std::vector<dtype> scratch(nThermStates);

    auto newThermEnergies = thermStateEnergies.firstBuf();
    auto oldThermEnergies = thermStateEnergies.secondBuf();

    for (Size x = 0; x < nSamples; ++x) {
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
                double maxErr = 1e-6, std::size_t callbackInterval = 1, const py::function *callback = nullptr) {
    // get dimensions...
    auto nThermStates = stateCounts.shape(0);
    auto nSamples = biasMatrix.shape(0);

    // work in log space so compute the log of the statecounts beforehand
    auto stateCountsLog = std::vector<dtype>(nThermStates);
    std::transform(stateCounts.data(), stateCounts.data() + nThermStates, stateCountsLog.begin(),
                   [](const auto counts) { return std::log(counts); }
    );

    // energies get computed here. We need the old ones to compute the new ones, and both to compute an iteration
    // error, so store them in an exchangable buffer
    ExchangeableArray<dtype, 1> thermStateEnergies(std::vector<StateIndex>{nThermStates}, 0.);

    // Get buffer for the bias energies
    ArrayBuffer<BiasMatrix<dtype>, 2> biasMatrixBuf{biasMatrix};

    py::gil_scoped_release gil;
    for (decltype(maxIter) iterationCount = 0; iterationCount < maxIter; ++iterationCount) {

        // The magic happens here
        selfConsistentUpdate(thermStateEnergies, biasMatrixBuf, stateCountsLog, nThermStates, nSamples);

        // Shift all energies by min(energies) so that the minimum energy equals zero (we are only interested in energy
        // differences!).
        shiftEnergiesToHaveZeroMinimum(thermStateEnergies, nThermStates);

        auto iterationError = computeError(thermStateEnergies, nThermStates);

        // keep the python user up to date on the progress by a callback
        if (callback && callbackInterval > 0 && iterationCount % callbackInterval == 0) {
            py::gil_scoped_acquire guard;
            (*callback)("inc"_a=callbackInterval, "error"_a=iterationError);
        }

        if (iterationError < maxErr) {
            // We have converged!
            break;
        }
    }
    return *thermStateEnergies.first();
}
}
