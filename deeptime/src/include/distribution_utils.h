#pragma once

#include <thread>
#include <random>
#include <ctime>

#include "common.h"

namespace deeptime {
namespace rnd {

template<typename Generator = std::mt19937>
Generator seededGenerator(std::uint32_t seed) {
    return Generator{seed};
}

template<typename Generator = std::mt19937>
Generator randomlySeededGenerator() {
    std::random_device r;
    std::random_device::result_type threadId = std::hash<std::thread::id>()(std::this_thread::get_id());
    std::random_device::result_type clck = clock();
    std::seed_seq seed{threadId, r(), r(), r(), clck, r(), r(), r(), r(), r()};
    return Generator(seed);
}

template<typename Generator = std::mt19937>
Generator &staticThreadLocalGenerator() {
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
    std::vector<RealType> operator()(Generator &gen) {
        std::vector<RealType> xs;
        xs.reserve(gammas.size());
        for (auto &gdist : gammas) {
            // ignore zeros
            xs.push_back(gdist(gen));
            /*if(gdist.alpha() != 0) {
            } else {
                xs.push_back(0);
            }*/
        }
        auto sum = std::accumulate(xs.begin(), xs.end(), 0.);
        for (auto it = xs.begin(); it != xs.end(); ++it) {
            *it /= sum;
        }
        return xs;
    }

    template<typename InputIterator>
    void params(InputIterator wbegin, InputIterator wend) {
        gammas.resize(0);
        std::transform(wbegin, wend, std::back_inserter(gammas), [](const auto &weight) {
            return std::gamma_distribution<RealType>(weight, 1);
        });
    }

private:
    std::vector<std::gamma_distribution<RealType>> gammas;
};

template<typename RealType = double>
class beta_distribution {
    // from https://gist.github.com/sftrabbit/5068941
public:
    typedef RealType result_type;

    class param_type {
    public:
        typedef beta_distribution distribution_type;

        explicit param_type(RealType a = 2.0, RealType b = 2.0)
                : a_param(a), b_param(b) {}

        RealType a() const { return a_param; }

        RealType b() const { return b_param; }

        bool operator==(const param_type &other) const {
            return (a_param == other.a_param &&
                    b_param == other.b_param);
        }

        bool operator!=(const param_type &other) const {
            return !(*this == other);
        }

    private:
        RealType a_param, b_param;
    };

    explicit beta_distribution(RealType a = 2.0, RealType b = 2.0)
            : a_gamma(a), b_gamma(b) {}

    explicit beta_distribution(const param_type &param)
            : a_gamma(param.a()), b_gamma(param.b()) {}

    void reset() {}

    param_type param() const {
        return param_type(a(), b());
    }

    void param(const param_type &param) {
        a_gamma = gamma_dist_type(param.a());
        b_gamma = gamma_dist_type(param.b());
    }

    template<typename URNG>
    result_type operator()(URNG &engine) {
        return generate(engine, a_gamma, b_gamma);
    }

    template<typename URNG>
    result_type operator()(URNG &engine, const param_type &param) {
        gamma_dist_type a_param_gamma(param.a()),
                b_param_gamma(param.b());
        return generate(engine, a_param_gamma, b_param_gamma);
    }

    result_type min() const { return 0.0; }

    result_type max() const { return 1.0; }

    RealType a() const { return a_gamma.alpha(); }

    RealType b() const { return b_gamma.alpha(); }

    bool operator==(const beta_distribution<result_type> &other) const {
        return (param() == other.param() &&
                a_gamma == other.a_gamma &&
                b_gamma == other.b_gamma);
    }

    bool operator!=(const beta_distribution<result_type> &other) const {
        return !(*this == other);
    }

private:
    typedef std::gamma_distribution<result_type> gamma_dist_type;

    gamma_dist_type a_gamma, b_gamma;

    template<typename URNG>
    result_type generate(URNG &engine,
                         gamma_dist_type &x_gamma,
                         gamma_dist_type &y_gamma) {
        result_type x = x_gamma(engine);
        auto denom = x + y_gamma(engine);
        auto r = x / denom;
        return r;
    }
};

}
}
