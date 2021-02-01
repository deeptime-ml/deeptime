#include <algorithm>
#include <cctype>
#include <locale>

#include "common.h"

// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

template<typename T>
T binomialCoefficient(T n, T k) {
    if (k > n) return 0;
    if (k * 2 > n) k = n - k;
    if (k == 0) return 1;

    T result = n;
    for (int i = 2; i <= k; ++i) {
        result *= (n - i + 1);
        result /= i;
    }
    return result;
}

template<typename T>
void nextMonomialPowers(T *const x, std::size_t n) {
    /* Returns powers for the next monomial. Implementation based on John Burkardt's MONOMIAL toolbox, see
    http://people.sc.fsu.edu/~jburkardt/m_src/monomial/monomial.html.*/
    std::size_t j = 0;
    for (std::size_t i = 1; i < n; ++i) {
        if (x[i] > 0) {
            j = i;
            break;
        }
    }

    if (j == 0) {
        auto tmp = x[0];
        x[0] = 0;
        x[n - 1] = tmp + 1;
    } else if (j < n - 1) {
        x[j]--;
        auto tmp = x[0] + 1;
        x[0] = 0;
        x[j - 1] += tmp;
    } else if (j == n - 1) {
        auto tmp = x[0];
        x[0] = 0;
        x[j - 1] = tmp + 1;
        x[j]--;
    }
}

np_array<std::int32_t> computePowerMatrix(std::size_t stateSpaceDim, std::size_t nMonomials) {
    std::vector<std::int32_t> powers(stateSpaceDim, 0);

    np_array<std::int32_t> powerMatrixArr({static_cast<std::size_t>(stateSpaceDim),
                                           static_cast<std::size_t>(nMonomials)});
    std::fill(powerMatrixArr.mutable_data(), powerMatrixArr.mutable_data() + stateSpaceDim * nMonomials, 0);
    /* Example: For d = 3 and p = 2, we obtain
    *
    * [[ 0  0  0  1  0  0  1  0  1  2]
    *  [ 0  0  1  0  0  1  0  2  1  0]
    *  [ 0  1  0  0  2  1  1  0  0  0]]
    **/
    auto powerMatrix = powerMatrixArr.template mutable_unchecked<2>();

    for (std::size_t i = 1; i < nMonomials; ++i) {
        nextMonomialPowers(powers.data(), stateSpaceDim);
        for (std::size_t k = 0; k < stateSpaceDim; ++k) {
            powerMatrix(k, i) = powers.at(k);
        }
    }

    return powerMatrixArr;
}

template<typename dtype>
np_array<dtype> evaluateMonomials(ssize_t p, const np_array_nfc<dtype> &xArr,
                                  const np_array<std::int32_t> &powerMatrixArr) {
    auto x = xArr.template unchecked<2>();
    auto stateSpaceDim = x.shape(0);
    auto nTestPoints = x.shape(1);
    auto nMonomials = binomialCoefficient(p + stateSpaceDim, p);

    // auto powerMatrixArr = computePowerMatrix(stateSpaceDim, nMonomials);
    auto powerMatrix = powerMatrixArr.template unchecked<2>();

    np_array<dtype> outArr({static_cast<std::size_t>(nMonomials), static_cast<std::size_t>(nTestPoints)});
    std::fill(outArr.mutable_data(), outArr.mutable_data() + outArr.size(), static_cast<dtype>(1));
    auto out = outArr.template mutable_unchecked<2>();

    for (ssize_t i = 0; i < nMonomials; ++i) {
        for (ssize_t j = 0; j < stateSpaceDim; ++j) {
            auto power = powerMatrix(stateSpaceDim - 1 - j, i);
            for (ssize_t k = 0; k < nTestPoints; ++k) {
                out(i, k) *= std::pow(x(j, k), power);
            }
        }
    }
    return outArr;
}


PYBIND11_MODULE(_basis_bindings, m) {
    m.def("evaluate_monomials", &evaluateMonomials<float>);
    m.def("evaluate_monomials", &evaluateMonomials<double>);
    m.def("evaluate_monomials", &evaluateMonomials<long double>);
    m.def("power_matrix", &computePowerMatrix);
    m.def("feature_names", [](const std::vector<std::string> &inputFeatures,
                              const np_array<std::int32_t> &powerMatrix) {
        auto stateSpaceDim = powerMatrix.shape(0);
        auto nMonomials = powerMatrix.shape(1);

        std::vector<std::string> out;
        out.reserve(nMonomials);

        for (ssize_t i = 0; i < nMonomials; ++i) {
            std::string monFeature {};
            for (ssize_t j = 0; j < stateSpaceDim; ++j) {
                auto power = powerMatrix.at(stateSpaceDim - 1 - j, i);

                if (power != 0) {
                    if (power == 1) {
                        monFeature += " " + inputFeatures.at(j);
                    } else {
                        monFeature += " " + inputFeatures.at(j) + "^" + std::to_string(power);
                    }
                }
            }
            trim(monFeature);
            out.push_back(monFeature);
        }

        return out;
    });
}
