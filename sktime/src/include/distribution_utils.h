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

template<typename Generator>
float genbet ( float aa, float bb, Generator &generator )

/******************************************************************************/
/*
  Purpose:

    GENBET generates a beta random deviate.

  Discussion:

    This procedure returns a single random deviate from the beta distribution
    with parameters A and B.  The density is

      x^(a-1) * (1-x)^(b-1) / Beta(a,b) for 0 < x < 1

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    19 September 2014

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Reference:

    Russell Cheng,
    Generating Beta Variates with Nonintegral Shape Parameters,
    Communications of the ACM,
    Volume 21, Number 4, April 1978, pages 317-322.

  Parameters:

    Input, float AA, the first parameter of the beta distribution.
    0.0 < AA.

    Input, float BB, the second parameter of the beta distribution.
    0.0 < BB.

    Output, float GENBET, a beta random variate.
*/
{
    float a;
    float alpha;
    float b;
    float beta;
    float delta;
    float gamma;
    float k1;
    float k2;
    const float log4 = 1.3862943611198906188;
    const float log5 = 1.6094379124341003746;
    float r;
    float s;
    float t;
    float u1;
    float u2;
    float v;
    float value;
    float w;
    float y;
    float z;

    std::uniform_real_distribution<float> unif( 0, 1);

    if ( aa <= 0.0 )
    {
        fprintf ( stderr, "\n" );
        fprintf ( stderr, "GENBET - Fatal error!\n" );
        fprintf ( stderr, "  AA <= 0.0\n" );
        exit ( 1 );
    }

    if ( bb <= 0.0 )
    {
        fprintf ( stderr, "\n" );
        fprintf ( stderr, "GENBET - Fatal error!\n" );
        fprintf ( stderr, "  BB <= 0.0\n" );
        exit ( 1 );
    }
/*
  Algorithm BB
*/
    if ( 1.0 < aa && 1.0 < bb )
    {
        a = std::min ( aa, bb );
        b = std::max ( aa, bb );
        alpha = a + b;
        beta = std::sqrt ( ( alpha - 2.0 ) / ( 2.0 * a * b - alpha ) );
        gamma = a + 1.0 / beta;

        for ( ; ; )
        {
            u1 = unif ( generator);
            u2 = unif(generator);
            v = beta * log ( u1 / ( 1.0 - u1 ) );
/*
  exp ( v ) replaced by r4_exp ( v )
*/
            w = a * std::exp ( v );

            z = u1 * u1 * u2;
            r = gamma * v - log4;
            s = a + r - w;

            if ( 5.0 * z <= s + 1.0 + log5 )
            {
                break;
            }

            t = log ( z );
            if ( t <= s )
            {
                break;
            }

            if ( t <= ( r + alpha * log ( alpha / ( b + w ) ) ) )
            {
                break;
            }
        }
    }
/*
  Algorithm BC
*/
    else
    {
        a = std::max ( aa, bb );
        b = std::min ( aa, bb );
        alpha = a + b;
        beta = 1.0 / b;
        delta = 1.0 + a - b;
        k1 = delta * ( 1.0 / 72.0 + b / 24.0 )
             / ( a / b - 7.0 / 9.0 );
        k2 = 0.25 + ( 0.5 + 0.25 / delta ) * b;

        for ( ; ; )
        {
            u1 = unif (generator );
            u2 = unif ( generator);

            if ( u1 < 0.5 )
            {
                y = u1 * u2;
                z = u1 * y;

                if ( k1 <= 0.25 * u2 + z - y )
                {
                    continue;
                }
            }
            else
            {
                z = u1 * u1 * u2;

                if ( z <= 0.25 )
                {
                    v = beta * log ( u1 / ( 1.0 - u1 ) );
                    w = a * exp ( v );

                    if ( aa == a )
                    {
                        value = w / ( b + w );
                    }
                    else
                    {
                        value = b / ( b + w );
                    }
                    return value;
                }

                if ( k2 < z )
                {
                    continue;
                }
            }

            v = beta * log ( u1 / ( 1.0 - u1 ) );
            w = a * exp ( v );

            if ( log ( z ) <= alpha * ( log ( alpha / ( b + w ) ) + v ) - log4 )
            {
                break;
            }
        }
    }

    if ( aa == a )
    {
        value = w / ( b + w );
    }
    else
    {
        value = b / ( b + w );
    }
    return value;
}

}
}
