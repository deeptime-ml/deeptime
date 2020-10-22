/********************************************************************************
 * This file is part of scikit-time.                                            *
 *                                                                              *
 * Copyright (c) 2020 AI4Science Group, Freie Universitaet Berlin (GER)         *
 *                                                                              *
 * scikit-time is free software: you can redistribute it and/or modify          *
 * it under the terms of the GNU Lesser General Public License as published by  *
 * the Free Software Foundation, either version 3 of the License, or            *
 * (at your option) any later version.                                          *
 *                                                                              *
 * This program is distributed in the hope that it will be useful,              *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of               *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
 * GNU General Public License for more details.                                 *
 *                                                                              *
 * You should have received a copy of the GNU Lesser General Public License     *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.        *
 ********************************************************************************/

//
// Created by mho on 2/3/20.
//

#pragma once

#include <chrono>
#include <random>

#include "common.h"
#include "distribution_utils.h"

template<typename dtype>
np_array<int> trajectory(std::size_t N, int start, const np_array<dtype> &P, const py::object& stop, long seed) {
    std::unique_ptr<py::gil_scoped_release> gil;
    auto nStates = P.shape(0);

    np_array<int> result (N);
    auto* data = result.mutable_data(0);

    data[0] = start;
    if (seed == -1) {
        seed = std::chrono::system_clock::now().time_since_epoch().count();
    }

    auto generator = seed < 0 ? sktime::rnd::randomlySeededGenerator() : sktime::rnd::seededGenerator(seed);
    std::discrete_distribution<> ddist;

    const dtype* pPtr = P.data();

    std::vector<int> stopState;
    bool hasStop = false;
    if(!stop.is_none()) {
        stopState = py::cast<std::vector<int>>(stop);
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
