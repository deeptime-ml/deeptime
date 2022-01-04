#include <pybind11/embed.h>
#include <catch2/catch.hpp>
#include "deeptime/markov/msm/tram/tram.h"

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
        int trajlengths[nThermStates];

        auto stateCounts = np_array_nfc<int>({nThermStates, nMarkovStates});
        auto transitionCounts = np_array_nfc<int>({nThermStates, nMarkovStates, nMarkovStates});

        deeptime::markov::tram::DTrajs dtrajs;
        deeptime::markov::tram::BiasMatrices<double> biasMatrices;

        for (int i = 0; i < nThermStates; ++i) {
            int length = i + 10;
            trajlengths[i] = length;
            dtrajs.push_back(np_array_nfc<int>(std::vector<int>{length}));
            biasMatrices.push_back(np_array_nfc<double>({length, nThermStates}));
        }

        WHEN("TRAMInput is constructed") {

            auto input = deeptime::markov::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts),
                                                                   dtrajs, biasMatrices);

            THEN("TRAMInput contains the correct input") {
                for (int K = 0; K < nThermStates; ++K) {
                    REQUIRE(input.sequenceLength(K) == trajlengths[K]);
                    REQUIRE(input.biasMatrix(K).ndim() == 2);
                    REQUIRE(input.dtraj(K).ndim() == 1);
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
                                                                  dtrajs, biasMatrices), std::runtime_error);
            }
        } WHEN("statecounts don't match transition counts at second index should throw") {
            transitionCounts = np_array_nfc<int>({nThermStates, nMarkovStates + 1, nMarkovStates});
            THEN("construction should throw runtime error") {
                REQUIRE_THROWS_AS(
                        deeptime::markov::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts),
                                                                  dtrajs, biasMatrices), std::runtime_error);
            }
        } WHEN("transitioncounts matrices are not squarew") {
            transitionCounts = np_array_nfc<int>({nThermStates, nMarkovStates, nMarkovStates + 1});
            THEN("construction should throw runtime error") {
                REQUIRE_THROWS_AS(
                        deeptime::markov::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts),
                                                                  dtrajs, biasMatrices), std::runtime_error);
            }
        } WHEN("dtrajs and bias matrices lengths don't match") {
            dtrajs.push_back(np_array_nfc<int>(std::vector<int>{10}));

            THEN("construction should throw runtime error") {
                REQUIRE_THROWS_AS(
                        deeptime::markov::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts),
                                                                  dtrajs,
                                                                  biasMatrices), std::runtime_error);
            }
        } WHEN("dtraj size neq size of bias matrix") {
            // lengthen the last dtraj by one
            int length = trajlengths[nThermStates - 1];
            dtrajs[nThermStates - 1] = np_array_nfc<int>(std::vector<int>{length + 1});
            THEN("construction should throw runtime error") {
                REQUIRE_THROWS_AS(
                        deeptime::markov::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts),
                                                                  dtrajs,
                                                                  biasMatrices), std::runtime_error);
            }
        }WHEN("bias matrix dimensions neq transition counts dimensions") {
            // lengthen last biasMatrix by 1
            int length = trajlengths[nThermStates - 1];
            biasMatrices[nThermStates - 1] = np_array_nfc<double>({length + 1, nThermStates});
            THEN("construction should throw runtime error") {
                REQUIRE_THROWS_AS(
                        deeptime::markov::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts),
                                                                  dtrajs, biasMatrices), std::runtime_error);
            }
        }WHEN("dtraj has dimension not equal to 1 should throw") {
            // change dimensions of last dtraj
            dtrajs[nThermStates - 1] = np_array_nfc<int>({trajlengths[nThermStates - 1], 1});
            THEN("construction should throw runtime error") {
                REQUIRE_THROWS_AS(
                        deeptime::markov::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts),
                                                                  dtrajs, biasMatrices), std::runtime_error);
            }
        }WHEN("bias matrix dimension not equal to 2") {
            // change length of last biasMatrix
            biasMatrices[nThermStates - 1] = np_array_nfc<double>({trajlengths[nThermStates - 1], nThermStates + 1});

            THEN("construction should throw runtime error") {
                REQUIRE_THROWS_AS(
                        deeptime::markov::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts),
                                                                  dtrajs, biasMatrices), std::runtime_error);
            }
        }
    }
}

