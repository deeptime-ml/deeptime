#include <random>
#include <pybind11/embed.h>
#include <catch2/catch.hpp>
#include <tram.h>
#include <iostream>
#include <fstream>
//
// Created by Maaike on 01/12/2021.
//
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
    }
    return dtrajs;
}

template<typename dtype>
auto generateBiasMatrix(int nThermStates, np_array_nfc<int> dtraj) {
    auto biasMatrix = np_array_nfc<dtype>({(int)dtraj.size(), nThermStates});
    auto biasMatrixBuf = biasMatrix.template mutable_unchecked<2>();

    std::uniform_int_distribution<int> distribution(0, 1);
    std::mt19937 gen;

    for (int i = 0; i < dtraj.size(); ++i) {
        for (int K = 0; K < nThermStates; ++K) {
            biasMatrixBuf(i,K) = (dtraj.data()[i] - K) * (dtraj.data()[i] - K) + distribution(gen);
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

TEMPLATE_TEST_CASE("TRAM", "[tram]", double, float) {
    GIVEN("Input") {
        py::scoped_interpreter guard;

        int nThermStates = 3;
        int nMarkovStates = 5;

        auto stateCounts = np_array_nfc<int>({nThermStates, nMarkovStates});
        auto transitionCounts = np_array_nfc<int>({nThermStates, nMarkovStates, nMarkovStates});

        int trajLengths[] = {10, 100, 50};

        auto dtrajs = generateDtrajs(nThermStates, nMarkovStates, trajLengths);
        auto biasMatrices = generateBiasMatrices<TestType>(nThermStates, dtrajs);

        auto inputPtr = std::make_shared<deeptime::tram::TRAMInput<TestType>>(
                std::move(stateCounts), std::move(transitionCounts), dtrajs,
                biasMatrices);

        WHEN("TRAM is constructed") {
            auto tram = deeptime::tram::TRAM<TestType>(inputPtr, 0);

            THEN("Result matrices have correct shape") {
                REQUIRE(tram.getEnergiesPerThermodynamicState()->size() == nThermStates);
                REQUIRE(tram.getBiasedConfEnergies().ndim() == 2);
                REQUIRE(tram.getEnergiesPerMarkovState().size() == nMarkovStates);
            }
	    THEN("... and are initialized with zeros") {
	        REQUIRE(tram.getBiasedConfEnergies().data()[0] == 0);
	    }
	    std::ofstream f;
	    f.open("~/tram_test_log.txt");
	    f << "logging input data" << std::endl;
	    f.close();

	    tram.estimate(1, 1e-8);
            TestType logLikelihood = tram.computeLogLikelihood();
            AND_WHEN("estimate() is called") {
                tram.estimate(13, 1e-8);
                THEN("loglikelihood increases") {
                    REQUIRE(tram.computeLogLikelihood() > logLikelihood);
                }
            }
        }
    }
}

