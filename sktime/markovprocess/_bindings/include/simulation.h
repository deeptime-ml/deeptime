//
// Created by mho on 2/3/20.
//

#pragma once

#include <chrono>
#include <random>

#include "common.h"

template<typename dtype, bool RELEASE_GIL>
np_array<int> trajectory(std::size_t N, int start, const np_array<dtype> &P, const py::object& stop, long seed) {
    std::unique_ptr<py::gil_scoped_release> gil;
    if (RELEASE_GIL) {
        gil = std::make_unique<py::gil_scoped_release>();
    }

    auto nStates = P.shape(0);

    np_array<int> result (N);
    int* data = result.mutable_data(0);
    data[0] = start;
    if (seed == -1) {
        seed = std::chrono::system_clock::now().time_since_epoch().count();
    }
    std::default_random_engine generator (seed);

    std::discrete_distribution<> ddist;

    const dtype* pPtr = P.data();

    int stopState = -1;
    bool hasStop = false;
    if(!stop.is_none()) {
        stopState = py::cast<int>(stop);
        hasStop = true;
    }

    if(!hasStop) {
        for (std::size_t t = 1; t < N; ++t) {
            auto prevState = data[t - 1];
            ddist.param({pPtr + prevState * nStates, pPtr + (prevState + 1) * nStates});
            data[t] = ddist(generator);
        }
    } else {
        for (std::size_t t = 1; t < N; ++t) {
            auto prevState = data[t - 1];
            ddist.param({pPtr + prevState * nStates, pPtr + (prevState + 1) * nStates});
            data[t] = ddist(generator);
            if(data[t] == stopState) {
                result.resize({std::distance(data, data + t + 1)});
                break;
            }
        }
    }
    return result;
}
