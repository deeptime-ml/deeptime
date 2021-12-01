#include <pybind11/embed.h>
#include <catch2/catch.hpp>
#include <tram.h>

//
// Created by Maaike on 01/12/2021.
//

TEST_CASE("TRAMInput", "[TRAMInput]") {
    const int nThermStates = 3;
    const int nMarkovStates = 5;
    int trajlengths[nThermStates];

    auto stateCounts = np_array_nfc<int>({nThermStates, nMarkovStates});
    auto transitionCounts = np_array_nfc<int>({nThermStates, nMarkovStates, nMarkovStates});

    std::vector<np_array_nfc<int>> dtrajs;
    std::vector<np_array_nfc<double>> biasMatrices;

    for (int i = 0; i < nThermStates; ++i) {
        int length = i + 10;
        trajlengths[i] = length;
        dtrajs.push_back(np_array_nfc<double>(std::vector<int>{length}));
        biasMatrices.push_back(np_array_nfc<double>({length, nThermStates}));
    }

    SECTION("correct dimensions should not throw") {
        CHECK_NOTHROW(deeptime::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts), dtrajs,
                                                        biasMatrices));
        auto input = deeptime::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts), dtrajs,
                                                       biasMatrices);
        for (int K = 0; K < nThermStates; ++K) {
            REQUIRE(input.sequenceLength(K) == trajlengths[K]);
        }
    }SECTION("statecounts don't match transition counts at first index should throw") {
        stateCounts = np_array_nfc<int>({nThermStates + 1, nMarkovStates});
        REQUIRE_THROWS_AS(deeptime::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts), dtrajs,
                                                            biasMatrices), std::runtime_error);
    }SECTION("statecounts don't match transition counts at second index should throw") {
        transitionCounts = np_array_nfc<int>({nThermStates, nMarkovStates + 1, nMarkovStates});
        REQUIRE_THROWS_AS(deeptime::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts), dtrajs,
                                                            biasMatrices), std::runtime_error);
    }SECTION("transitioncounts matrices are not square should throw") {
        transitionCounts = np_array_nfc<int>({nThermStates, nMarkovStates, nMarkovStates + 1});
        REQUIRE_THROWS_AS(deeptime::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts), dtrajs,
                                                            biasMatrices), std::runtime_error);
    }SECTION("dtrajs and bias matrices lengths don't match should throw") {
        dtrajs.append(np_array_nfc<int>(std::vector<int>{10}));
        REQUIRE_THROWS_AS(deeptime::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts), dtrajs,
                                                            biasMatrices), std::runtime_error);
    }SECTION("bias matrix and transition counts dimensions don't match should throw") {
        // lengthen the last dtraj by one
        int length = trajlengths[nThermStates - 1];
        dtrajs[nThermStates - 1] = np_array_nfc<double>(std::vector<int>{length + 1});
        REQUIRE_THROWS_AS(deeptime::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts), dtrajs,
                                                            biasMatrices), std::runtime_error);
    }SECTION("bias matrix and transition counts dimensions don't match should throw") {
        // lengthen last biasMatrix by 1
        biasMatrices[nThermStates - 1] = np_array_nfc<double>({trajlengths[nThermStates - 1] + 1, nThermStates});
        REQUIRE_THROWS_AS(deeptime::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts), dtrajs,
                                                            biasMatrices), std::runtime_error);
    }SECTION("dtraj has dimension not equal to 1 should throw") {
        // change dimensions of last dtraj
        dtrajs[nThermStates - 1] = np_array_nfc<double>({trajlengths[nThermStates - 1], 1});
        REQUIRE_THROWS_AS(deeptime::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts), dtrajs,
                                                            biasMatrices), std::runtime_error);
    }SECTION("bias matrix dimension not equal to 2 should throw") {
        // change length of last biasMatrix
        biasMatrices[nThermStates - 1] = np_array_nfc<double>({trajlengths[nThermStates - 1], nThermStates + 1});
        REQUIRE_THROWS_AS(deeptime::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts), dtrajs,
                                                            biasMatrices), std::runtime_error);
    }
}
