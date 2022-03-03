#include <pybind11/embed.h>
#include <catch2/catch.hpp>
#include "deeptime/markov/msm/tram/tram.h"
#include "deeptime/markov/msm/tram/common.h"

//
// Created by Maaike on 01/12/2021.
//
using namespace deeptime;

TEST_CASE("TRAMInput", "[tram]") {
    using Input = deeptime::markov::tram::TRAMInput<double>;

    GIVEN("Input") {
        py::scoped_interpreter guard;

        const int nThermStates = 3;
        const int nMarkovStates = 5;

        auto stateCounts = np_array_nfc<int>({nThermStates, nMarkovStates});
        auto transitionCounts = np_array_nfc<int>({nThermStates, nMarkovStates, nMarkovStates});

        int length = 10;
        int nTrajs = 5;

        deeptime::markov::tram::BiasMatrices<double> biasMatrices = std::vector<markov::tram::BiasMatrix<double>>(nTrajs);

        for (auto i=0; i < nTrajs; ++i) {
            biasMatrices[i] = deeptime::markov::tram::BiasMatrix<double>({length, nThermStates});
        }

        WHEN("TRAMInput is constructed") {

            auto input = deeptime::markov::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts),
                                                                    biasMatrices);
            THEN("TRAMInput contains the correct input") {
                for (int K = 0; K < nThermStates; ++K) {
                    REQUIRE(input.nSamples(K) == length);
                    REQUIRE(input.biasMatrix(K).ndim() == 2);
                }
                REQUIRE(input.stateCounts().ndim() == 2);
                REQUIRE(input.transitionCounts().ndim() == 3);
            }
        }
        WHEN("statecounts don't match transition counts at first index") {
            stateCounts = np_array_nfc<int>({nThermStates + 1, nMarkovStates});
            THEN("construction should throw runtime error") {
                REQUIRE_THROWS_AS(
                        deeptime::markov::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts),
                                                                  biasMatrices), std::runtime_error);
            }
        } WHEN("statecounts don't match transition counts at second index should throw") {
            transitionCounts = np_array_nfc<int>({nThermStates, nMarkovStates + 1, nMarkovStates});
            THEN("construction should throw runtime error") {
                REQUIRE_THROWS_AS(
                        deeptime::markov::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts),
                                                                  biasMatrices), std::runtime_error);
            }
        } WHEN("transitioncounts matrices are not square") {
            transitionCounts = np_array_nfc<int>({nThermStates, nMarkovStates, nMarkovStates + 1});
            THEN("construction should throw runtime error") {
                REQUIRE_THROWS_AS(
                        deeptime::markov::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts),
                                                                  biasMatrices), std::runtime_error);
            }
        } WHEN("bias matrix second dimension neq n_therm_states") {
            // lengthen last biasMatrix by 1
            biasMatrices[1] = np_array_nfc<double>({length, nThermStates + 1});
            THEN("construction should throw runtime error") {
                REQUIRE_THROWS_AS(
                        deeptime::markov::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts),
                                                                  biasMatrices), std::runtime_error);
            }
        } WHEN("bias matrix dimension not equal to 2") {
            // change length of last biasMatrix
            biasMatrices[1] = np_array_nfc<double>({length, nThermStates, 1});

            THEN("construction should throw runtime error") {
                REQUIRE_THROWS_AS(
                        deeptime::markov::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts),
                                                                  biasMatrices), std::runtime_error);
            }
        }
    }
}

