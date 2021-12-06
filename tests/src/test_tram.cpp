#include <random>
#include <pybind11/embed.h>
#include <catch2/catch.hpp>
#include <deeptime/markov/tram/tram.h>
#include <iostream>
#include <fstream>
//
// Created by Maaike on 01/12/2021.
//
using namespace deeptime;

auto generateDtraj(int nMarkovStates, int length, float shift) {
    std::vector<float> weights(nMarkovStates, 0.);
    std::iota(begin(weights), end(weights), 0.);
    std::transform(begin(weights), end(weights), begin(weights), [nMarkovStates, shift](auto weight) {
        return weight + shift - nMarkovStates / 2.;
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

auto generateDtrajs(int nThermsStates, int nMarkovStates, int *trajLengths) {
    std::vector<np_array_nfc<int>> dtrajs;

    for (int K = 0; K < nThermsStates; ++K) {
        dtrajs.push_back(generateDtraj(nMarkovStates, trajLengths[K], 0.1 * K));
        // a very ugly way to ensure all markovstates are samples at least once.
        for (int i = 0; i < nMarkovStates; ++i) {
            dtrajs[K].mutable_at(1 + i) = i;
        }
    }
    return dtrajs;

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

template<typename dtype>
auto generateBiasMatrices(int nThermsStates, std::vector<np_array_nfc<int>> dtrajs) {
    std::vector<np_array_nfc<dtype>> matrices;

    for (int K = 0; K < nThermsStates; ++K) {
        matrices.push_back(generateBiasMatrix<dtype>(nThermsStates, dtrajs[K]));
    }
    return matrices;
}

auto countStates(int nThermStates, int nMarkovStates, std::vector<np_array_nfc<int>> dtrajs) {
    auto stateCounts = np_array_nfc<int>({nThermStates, nMarkovStates});
    auto transitionCounts = np_array_nfc<int>({nThermStates, nMarkovStates, nMarkovStates});
    std::fill(stateCounts.mutable_data(), stateCounts.mutable_data() + stateCounts.size(), 0);
    std::fill(transitionCounts.mutable_data(), transitionCounts.mutable_data() + transitionCounts.size(), 0);

    int state = -1;
    int prevstate = -1;

    for (int K = 0; K < nThermStates; ++K) {
        for (int i = 0; i < dtrajs[K].size(); ++i) {
            prevstate = state;
            state = dtrajs[K].at(i);

            stateCounts.mutable_at(K, state)++;
            if (prevstate != -1) {
                transitionCounts.mutable_at(K, prevstate, state)++;
            }
        }
    }
    return std::tuple(stateCounts, transitionCounts);
}

template<typename dtype>
bool areFinite(np_array_nfc<dtype> arr) {
    return std::transform_reduce(
            arr.data(), arr.data() + arr.size(),
            true, std::logical_and<bool>(),
            [](dtype x) { return std::isfinite(x); });
}

TEMPLATE_TEST_CASE("TRAM", "[tram]", double, float) {

    GIVEN("Input") {
        py::scoped_interpreter guard;

        int nThermStates = 3;
        int nMarkovStates = 5;

        int trajLengths[] = {100, 100, 100};

        auto dtrajs = generateDtrajs(nThermStates, nMarkovStates, trajLengths);
        auto biasMatrices = generateBiasMatrices<TestType>(nThermStates, dtrajs);

        np_array_nfc<int> stateCounts, transitionCounts;
        std::tie(stateCounts, transitionCounts) = countStates(nThermStates, nMarkovStates, dtrajs);

        auto inputPtr = std::make_shared<deeptime::tram::TRAMInput<TestType>>(
                std::move(stateCounts), std::move(transitionCounts), dtrajs,
                biasMatrices);


        WHEN("TRAM is constructed") {
            auto tram = deeptime::tram::TRAM<TestType>(inputPtr, 0);

            THEN("Result matrices are initialized") {
                REQUIRE(tram.getEnergiesPerThermodynamicState().size() == nThermStates);
                REQUIRE(tram.getBiasedConfEnergies().ndim() == 2);
                REQUIRE(tram.getEnergiesPerMarkovState().size() == nMarkovStates);
                REQUIRE(tram.getBiasedConfEnergies().data()[0] == 0);
            }

            AND_WHEN("estimate() is called") {
                tram.estimate(1, 1e-8, true);

                TestType LL = tram.computeLogLikelihood();

                THEN("Energies are finite") {
                    auto thermStateEnergies = tram.getEnergiesPerThermodynamicState();
                    auto markovStateEnergies = tram.getEnergiesPerMarkovState();

                    REQUIRE(areFinite<TestType>(thermStateEnergies));
                    REQUIRE(areFinite<TestType>(markovStateEnergies));

                }
                AND_WHEN("estimate() is called again") {
                    tram.estimate(1, (TestType) 1e-8, true);

                    THEN("loglikelihood increases") {
                        TestType newLL = tram.computeLogLikelihood();
                        REQUIRE(std::isfinite(LL));
                        REQUIRE(std::isfinite(newLL));
                        REQUIRE(newLL > LL);
                        REQUIRE(newLL < 0);
                    }
                }
            }
        }
    }
}

