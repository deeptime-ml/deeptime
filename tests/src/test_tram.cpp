#include <random>
#include <pybind11/embed.h>
#include <catch2/catch.hpp>
#include "deeptime/markov/msm/tram/tram.h"
//
// Created by Maaike on 01/12/2021.
//
using namespace deeptime;

auto generateDtraj(int nMarkovStates, int length) {
    std::vector<float> weights(nMarkovStates, 0.);
    std::iota(begin(weights), end(weights), 0.);
    std::transform(begin(weights), end(weights), begin(weights), [nMarkovStates](auto weight) {
        return weight - nMarkovStates / 2.;
    });
    std::transform(begin(weights), end(weights), begin(weights), [](auto x) {
        return std::exp(-x * x);
    });
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> distribution(begin(weights), end(weights));

    auto generator = [&distribution, &gen]() {
        return distribution(gen);
    };
    np_array_nfc<int> traj(std::vector<int>{length});
    std::generate(traj.mutable_data(), traj.mutable_data() + length, generator);

    return traj;
}

template<typename dtype>
auto generateBiasMatrix(int nThermStates, np_array_nfc<int> dtraj) {
    auto biasMatrix = np_array_nfc<dtype>({(int) dtraj.size(), nThermStates});
    auto biasMatrixBuf = biasMatrix.template mutable_unchecked<2>();

    std::uniform_real_distribution<> distribution(-0.5, 0.5);
    std::mt19937 gen;

    for (int i = 0; i < dtraj.size(); ++i) {
        dtype coord = dtraj.data()[i] + distribution(gen);
        for (int K = 0; K < nThermStates; ++K) {
            biasMatrixBuf(i, K) = (coord - K) * (coord - K);
        };
    }
    return biasMatrix;
}

template<typename DTraj>
auto countStates(int nThermStates, int nMarkovStates, const DTraj &dtraj) {
    auto stateCounts = np_array_nfc<int>({nThermStates, nMarkovStates});
    auto transitionCounts = np_array_nfc<int>({nThermStates, nMarkovStates, nMarkovStates});
    std::fill(stateCounts.mutable_data(), stateCounts.mutable_data() + stateCounts.size(), 0);
    std::fill(transitionCounts.mutable_data(), transitionCounts.mutable_data() + transitionCounts.size(), 0);

    int state = -1;
    int prevstate = -1;

    for (int i = 0; i < dtraj.size(); ++i) {
        // get therm state from whatever section in the traj we're in. First few samples get 0, the 1, etc.
        int K = (i * nThermStates) / dtraj.size();

        prevstate = state;
        state = dtraj.at(i);

        stateCounts.mutable_at(K, state)++;
        if (prevstate != -1) {
            transitionCounts.mutable_at(K, prevstate, state)++;
        }
    }
    return std::tuple(stateCounts, transitionCounts);
}

template<typename dtype>
bool areFinite(np_array_nfc<dtype> arr) {
    auto finiteBools = np_array_nfc<bool>(std::vector<int>(arr.shape(), arr.shape() + arr.ndim()));
    std::transform(arr.data(), arr.data() + arr.size(),
                   finiteBools.mutable_data(), [](dtype x) { return std::isfinite(x); });
    return std::accumulate(finiteBools.data(), finiteBools.data() + finiteBools.size(), true, std::logical_and<bool>());
}


TEMPLATE_TEST_CASE("TRAM", "[tram]", double, float) {

    GIVEN("Input") {
        py::scoped_interpreter guard;

        int nThermStates = 3;
        int nMarkovStates = 5;

        auto dtraj = generateDtraj(nMarkovStates, 100);
        auto biasMatrix = generateBiasMatrix<TestType>(nThermStates, dtraj);

        auto [stateCounts, transitionCounts] = countStates(nThermStates, nMarkovStates, dtraj);

        auto inputPtr = std::make_shared<deeptime::markov::tram::TRAMInput<TestType>>(
                std::move(stateCounts), std::move(transitionCounts), dtraj,
                biasMatrix);


        WHEN("TRAM is constructed") {
            auto tram = deeptime::markov::tram::TRAM<TestType>(nThermStates, nMarkovStates);

            THEN("Result matrices are initialized") {
                REQUIRE(tram.thermStateEnergies().size() == nThermStates);
                REQUIRE(tram.biasedConfEnergies().ndim() == 2);
                REQUIRE(tram.markovStateEnergies().size() == nMarkovStates);
                REQUIRE(tram.biasedConfEnergies().data()[0] == 0);
            }

            AND_WHEN("estimate() is called") {
                tram.estimate(inputPtr, 3, 1e-8, true);

                TestType LL = deeptime::markov::tram::computeLogLikelihood(
                        inputPtr->dtraj(), inputPtr->biasMatrix(), tram.biasedConfEnergies(),
                        tram.modifiedStateCountsLog(), tram.thermStateEnergies(), inputPtr->stateCounts(),
                        inputPtr->transitionCounts(), tram.transitionMatrices());

                THEN("Energies are finite") {
                    auto thermStateEnergies = tram.thermStateEnergies();
                    auto markovStateEnergies = tram.markovStateEnergies();

                    REQUIRE(areFinite<TestType>(thermStateEnergies));
                    REQUIRE(areFinite<TestType>(markovStateEnergies));

                }AND_THEN("Transition matrices are transition matrices") {
                    auto transitionMatrices = tram.transitionMatrices();

                    REQUIRE(areFinite<TestType>(transitionMatrices));
                    auto matrixSize = nMarkovStates * nMarkovStates;
                    auto rowSize = nMarkovStates;

                    for (int K = 0; K < nThermStates; ++K) {
                        for (int i = 0; i < nMarkovStates; ++i) {
                            auto rowSum = std::accumulate(transitionMatrices.data() + K * matrixSize + i * rowSize,
                                                          transitionMatrices.data() + K * matrixSize +
                                                          (i + 1) * rowSize,
                                                          0.0, std::plus<TestType>());
                            REQUIRE(Approx(rowSum) == 1.0);
                        }
                    }
                }
            }
        }
    }
}

