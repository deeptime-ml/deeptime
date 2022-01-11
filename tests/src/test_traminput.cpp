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

        int length = 10;
        deeptime::markov::tram::DTraj dtraj(std::vector<int>{length});
        deeptime::markov::tram::BiasMatrix<double> biasMatrix({length, nThermStates});

        WHEN("TRAMInput is constructed") {

            auto input = deeptime::markov::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts),
                                                                   dtraj, biasMatrix);

            THEN("TRAMInput contains the correct input") {
                for (int K = 0; K < nThermStates; ++K) {
                    REQUIRE(input.nSamples() == trajlengths[K]);
                    REQUIRE(input.biasMatrix().ndim() == 2);
                    REQUIRE(input.dtraj().ndim() == 1);
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
                                                                  dtraj, biasMatrix), std::runtime_error);
            }
        } WHEN("statecounts don't match transition counts at second index should throw") {
            transitionCounts = np_array_nfc<int>({nThermStates, nMarkovStates + 1, nMarkovStates});
            THEN("construction should throw runtime error") {
                REQUIRE_THROWS_AS(
                        deeptime::markov::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts),
                                                                  dtraj, biasMatrix), std::runtime_error);
            }
        } WHEN("transitioncounts matrices are not squarew") {
            transitionCounts = np_array_nfc<int>({nThermStates, nMarkovStates, nMarkovStates + 1});
            THEN("construction should throw runtime error") {
                REQUIRE_THROWS_AS(
                        deeptime::markov::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts),
                                                                  dtraj, biasMatrix), std::runtime_error);
            }
        } WHEN("dtrajs and bias matrices lengths don't match") {
            dtraj = np_array_nfc<int>(std::vector<int>{length + 1});

            THEN("construction should throw runtime error") {
                REQUIRE_THROWS_AS(
                        deeptime::markov::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts),
                                                                  dtraj, biasMatrix), std::runtime_error);
            }
        } WHEN("bias matrix second dimension neq n_therm_states") {
            // lengthen last biasMatrix by 1
            biasMatrix = np_array_nfc<double>({length, nThermStates + 1});
            THEN("construction should throw runtime error") {
                REQUIRE_THROWS_AS(
                        deeptime::markov::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts),
                                                                  dtraj, biasMatrix), std::runtime_error);
            }
        }WHEN("dtraj has dimension not equal to 1 should throw") {
            // change dimensions of last dtraj
            dtraj = np_array_nfc<int>({length, 1});
            THEN("construction should throw runtime error") {
                REQUIRE_THROWS_AS(
                        deeptime::markov::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts),
                                                                  dtraj, biasMatrix), std::runtime_error);
            }
        }WHEN("bias matrix dimension not equal to 2") {
            // change length of last biasMatrix
            biasMatrix = np_array_nfc<double>({length, nThermStates, 1});

            THEN("construction should throw runtime error") {
                REQUIRE_THROWS_AS(
                        deeptime::markov::tram::TRAMInput<double>(std::move(stateCounts), std::move(transitionCounts),
                                                                  dtraj, biasMatrix), std::runtime_error);
            }
        }
    }
}

