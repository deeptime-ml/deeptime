#include <random>
#include <pybind11/embed.h>
#include <catch2/catch.hpp>
#include "deeptime/markov/msm/tram/tram.h"
//
// Created by Maaike on 01/12/2021.
//
using namespace deeptime;

template<typename dtype>
np_array_nfc<dtype> createFilledArray(const std::vector<py::ssize_t> &dims) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, 1);

    auto generator = [&dist, &gen]() {
        return dist(gen);
    };

    np_array_nfc<dtype> array(dims);
    std::generate(array.mutable_data(), array.mutable_data() + array.size(), generator);

    return array;
}


TEMPLATE_TEST_CASE("TRAM", "[tram]", double, float) {

    GIVEN("Input") {
        py::scoped_interpreter guard;

        int nThermStates = 3;
        int nMarkovStates = 5;

        auto biasedConfEnergies = createFilledArray<TestType>({nThermStates, nMarkovStates});
        auto lagrangianMultLog = createFilledArray<TestType>({nThermStates, nMarkovStates});
        auto modifiedStateCountsLog = createFilledArray<TestType>({nThermStates, nMarkovStates});

        WHEN("TRAM is constructed") {
            auto tram = deeptime::markov::tram::TRAM<TestType>(biasedConfEnergies, lagrangianMultLog,
                                                               modifiedStateCountsLog);

            THEN("Result matrices are initialized") {
                REQUIRE(tram.thermStateEnergies().size() == nThermStates);
                REQUIRE(tram.biasedConfEnergies().ndim() == 2);
                REQUIRE(tram.markovStateEnergies().size() == nMarkovStates);

                REQUIRE(std::equal(tram.biasedConfEnergies().begin(), tram.biasedConfEnergies().end(),
                                   biasedConfEnergies.begin(), biasedConfEnergies.end()));
                REQUIRE(std::equal(tram.lagrangianMultLog().begin(), tram.lagrangianMultLog().end(),
                                   lagrangianMultLog.begin(), lagrangianMultLog.end()));
                REQUIRE(std::equal(tram.modifiedStateCountsLog().begin(), tram.modifiedStateCountsLog().end(),
                                   modifiedStateCountsLog.begin(), modifiedStateCountsLog.end()));
            }
        }
    }
}

