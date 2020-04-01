#pragma once

#include <thread>
#include <random>
#include <ctime>

#include "common.h"

namespace sktime::rnd {

template<typename Generator = std::default_random_engine>
Generator& staticGenerator() {
    static thread_local Generator generator(clock() + std::hash<std::thread::id>()(std::this_thread::get_id()));
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
