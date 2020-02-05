//
// Created by mho on 2/3/20.
//

#pragma once

#include <random>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;

template<typename dtype>
using np_array = py::array_t<dtype, py::array::c_style>;

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
            if(gdist.alpha() != 0) {
                xs.push_back(gdist(gen));
            } else {
                xs.push_back(0);
            }
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
