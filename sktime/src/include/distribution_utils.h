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

#pragma once

#include <thread>
#include <random>
#include <ctime>

#include "common.h"

namespace sktime {
namespace rnd {

template<typename Generator = std::default_random_engine>
Generator seededGenerator(std::uint32_t seed) {
    return std::default_random_engine(seed);
}

template<typename Generator = std::default_random_engine>
Generator randomlySeededGenerator() {
    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    return Generator(seed);
}

template<typename Generator = std::default_random_engine>
Generator& staticThreadLocalGenerator() {
    static thread_local Generator generator(randomlySeededGenerator<Generator>());
    return generator;
}

template<typename RealType>
class dirichlet_distribution {
public:
    dirichlet_distribution() : gammas() {}
    template<typename InputIterator>
    dirichlet_distribution(InputIterator wbegin, InputIterator wend) {
        params(wbegin, wend);
    }

    template<typename Generator>
    std::vector<RealType> operator()(Generator& gen) {
        std::vector<RealType> xs;
        xs.reserve(gammas.size());
        for(auto& gdist : gammas) {
            // ignore zeros
            xs.push_back(gdist(gen));
            /*if(gdist.alpha() != 0) {
            } else {
                xs.push_back(0);
            }*/
        }
        auto sum = std::accumulate(xs.begin(), xs.end(), 0.);
        for(auto it = xs.begin(); it != xs.end(); ++it) {
            *it /= sum;
        }
        return xs;
    }

    template<typename InputIterator>
    void params(InputIterator wbegin, InputIterator wend) {
        gammas.resize(0);
        std::transform(wbegin, wend, std::back_inserter(gammas), [](const auto& weight) {
            return std::gamma_distribution<RealType>(weight, 1);
        });
    }
private:
    std::vector<std::gamma_distribution<RealType>> gammas;
};

}
}
