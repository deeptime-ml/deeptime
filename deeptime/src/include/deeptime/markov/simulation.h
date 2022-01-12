//
// Created by mho on 2/3/20.
//

#pragma once

#include <chrono>
#include <random>

#include <deeptime/common.h>
#include <deeptime/util/distribution_utils.h>

namespace deeptime::markov {

template<typename dtype>
np_array<int> trajectory(std::size_t N, int start, const np_array<dtype> &P, const py::object& stop, long seed) {
    np_array<int> result (static_cast<py::ssize_t>(N));
    auto nStates = P.shape(0);
    auto* data = result.mutable_data(0);

    std::vector<int> stopState;
    bool hasStop = false;
    if(!stop.is_none()) {
        stopState = py::cast<std::vector<int>>(stop);
        hasStop = true;
    }

    py::gil_scoped_release gilRelease;

    data[0] = start;
    if (seed == -1) {
        seed = std::chrono::system_clock::now().time_since_epoch().count();
    }

    auto generator = seed < 0 ? deeptime::rnd::randomlySeededGenerator() : deeptime::rnd::seededGenerator(seed);
    std::discrete_distribution<> ddist;

    const dtype* pPtr = P.data();

    if(!hasStop) {
        for (std::size_t t = 1; t < N; ++t) {
            auto prevState = data[t - 1];
            ddist.param({pPtr + prevState * nStates, pPtr + (prevState + 1) * nStates});
            data[t] = ddist(generator);
        }
    } else {
        for (std::size_t t = 1; t < N; ++t) {
            if(std::find(stopState.begin(), stopState.end(), data[t-1]) != stopState.end()) {
                result.resize({std::distance(data, data + t)});
                break;
            }
            auto prevState = data[t - 1];
            ddist.param({pPtr + prevState * nStates, pPtr + (prevState + 1) * nStates});
            data[t] = ddist(generator);
        }
    }
    return result;
}

}
