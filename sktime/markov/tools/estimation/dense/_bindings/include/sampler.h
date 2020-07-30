//
// Created by mho on 7/30/20.
//

#pragma once

#include <random>
#include <distribution_utils.h>
#include "common.h"
#include "mle_trev.h"

namespace util {
template<typename dtype>
auto isPositive(dtype x) -> bool {
    auto eps = std::numeric_limits<dtype>::min();  // smallest positive number for floating points
    return x >= eps && !std::isnan(x) && !std::isinf(x);
}
}

template<typename dtype, typename Generator = std::default_random_engine>
class RevSampler {

public:
    explicit RevSampler(int seed) : uniform(0, 1) {
        if (seed < 0) {
            generator = sktime::rnd::randomlySeededGenerator();
        } else {
            generator = sktime::rnd::seededGenerator(static_cast<std::uint32_t>(seed));
        }
    }

    void updateSparse(const np_array<dtype> &arrC, const np_array<dtype> &arrSumC, np_array<dtype> &arrX,
                      const np_array<int> &arrI, const np_array<int> &arrJ,
                      int n_step) {
        dtype *X = arrX.mutable_data();
        auto n = arrC.shape(0);
        auto n_idx = arrI.shape(0);
        std::vector<dtype> arrSumX(arrX.shape(0), static_cast<dtype>(0));
        {
            auto Xaccess = arrX.template unchecked<2>();
            for (decltype(arrX.shape(0)) i = 0; i < arrX.shape(0); ++i) {
                for (decltype(arrX.shape(1)) j = 0; j < arrX.shape(1); ++j) {
                    arrSumX.at(i) += Xaccess(i, j);
                }
            }
        }

        auto *sumX = arrSumX.data();
        auto C = arrC.template unchecked<2>();
        auto sumC = arrSumC.template unchecked<1>();

        const int *const I = arrI.data();
        const int *const J = arrJ.data();

        // row indexes
        std::unique_ptr<int[]> rowIndices(new int[n + 1]);
        generateRowIndices(I, n, n_idx, rowIndices.get());

        for (int iter = 0; iter < n_step; iter++) {
            // update all X row sums once every iteration and then only do cheap updates.
            for (int i = 0; i < n; i++) {
                sumX[i] = _sumRowSparse(X, n, i, J, rowIndices[i], rowIndices[i + 1]);
            }

            for (int k = 0; k < n_idx; k++) {
                auto i = I[k];
                auto j = J[k];
                if (i == j) {
                    if (util::isPositive(C(i, i)) && util::isPositive(sumC(i) - C(i, i))) {
                        beta.param((typename decltype(beta)::param_type) {C(i, i), sumC(i) - C(i, i)});
                        auto tmp1 = beta(generator);
                        // auto tmp1 = sktime::rnd::genbet(C(i, i), sumC(i) - C(i, i), generator);
                        auto tmp2 = tmp1 / (1 - tmp1) * (sumX[i] - X[i * n + i]);
                        if (util::isPositive(tmp2)) {
                            sumX[i] += tmp2 - X[i * n + i];  // update sumX
                            X[i * n + i] = tmp2;
                        }
                    }
                }
                if (i < j)  // only work on the upper triangle, because we have symmetry.
                {
                    auto tmp1 = sumX[i] - X[i * n + j];
                    auto tmp2 = sumX[j] - X[j * n + i];
                    X[i * n + j] = _updateStep(X[i * n + j], tmp1, tmp2, C(i, j) + C(j, i),
                                               sumC(i), sumC(j), 1);
                    X[j * n + i] = X[i * n + j];
                    // update X
                    sumX[i] = tmp1 + X[i * n + j];
                    sumX[j] = tmp2 + X[j * n + i];
                }
            }

            _normalizeAllSparse(X, I, J, n, n_idx);
        }

    }

private:

    Generator generator;
    std::normal_distribution<dtype> normal{0, 1};
    std::gamma_distribution<dtype> gamma;
    sktime::rnd::beta_distribution<dtype> beta;
    std::uniform_real_distribution<dtype> uniform;


    bool acceptStep(dtype log_prob_old, dtype log_prob_new) {
        return log_prob_new > log_prob_old ||
               uniform(generator) < std::exp(std::min(log_prob_new - log_prob_old, static_cast<dtype>(0)));
    }

    void generateRowIndices(const int *const I, int n, int n_idx, int *row_indexes) {
        row_indexes[0] = 0;  // starts with row 0
        int current_row = 0;
        for (int k = 0; k < n_idx; k++) {
            // still at same row? do nothing
            if (I[k] == current_row)
                continue;
            // row has advanced one or multiple times. Update multiple row indexes until we are equal
            while (I[k] > current_row) {
                current_row++;
                row_indexes[current_row] = k;
            }
        }
        // stop sign
        row_indexes[n] = n_idx;
    }

    double _sumRowSparse(dtype *X, int n, int i, const int *const J, int from, int to) {
        int j;
        double sum = 0.0;
        for (j = from; j < to; j++)
            sum += X[i * n + J[j]];
        return sum;
    }

    void _normalizeAllSparse(dtype *X, const int *const I, const int *const J, int n, int n_idx) {
        // sum all
        dtype sum = 0.0;
        for (int k = 0; k < n_idx; k++)
            sum += X[I[k] * n + J[k]];
        // normalize all
        for (int k = 0; k < n_idx; k++)
            X[I[k] * n + J[k]] /= sum;
    }

    dtype _updateStep(dtype v0, dtype v1, dtype v2, dtype c0, dtype c1, dtype c2, int random_walk_stepsize) {
        /*
        update the sample v0 according to
        the distribution v0^(c0-1)*(v0+v1)^(-c1)*(v0+v2)^(-c2)
        */
        dtype a = c1 + c2 - c0;
        dtype b = (c1 - c0) * v2 + (c2 - c0) * v1;
        dtype c = -c0 * v1 * v2;
        dtype v_bar = 0.5 * (-b + sqrt(b * b - 4 * a * c)) / a;
        dtype h = c1 / (v_bar + v1) * ((v_bar + v1))
                  + c2 / ((v_bar + v2) * (v_bar + v2))
                  - c0 / (v_bar * v_bar);
        dtype k = -h * v_bar * v_bar;
        dtype theta = -1.0 / (h * v_bar);
        dtype log_v0 = log(v0);
        dtype v0_new = 0.0;
        dtype log_v0_new = 0.0;
        dtype log_prob_old = 0.0;
        dtype log_prob_new = 0.0;

        // about 1.5 sec: gamma and normf generation
        // about 1 sec: logs+exps in else blocks

        if (util::isPositive(k) && util::isPositive(theta)) {
            gamma.param((typename decltype(gamma)::param_type) {static_cast<dtype>(1) / theta, k});
            v0_new = gamma(generator); // gengam(1.0 / theta, k);
            log_v0_new = log(v0_new);
            if (util::isPositive(v0_new)) {
                if (v0 == 0) {
                    v0 = v0_new;
                    log_v0 = log_v0_new;
                } else {
                    log_prob_new = (c0 - 1) * log_v0_new - c1 * log(v0_new + v1) - c2 * log(v0_new + v2);
                    log_prob_new -= (k - 1) * log_v0_new - v0_new / theta;
                    log_prob_old = (c0 - 1) * log_v0 - c1 * log(v0 + v1) - c2 * log(v0 + v2);
                    log_prob_old -= (k - 1) * log_v0 - v0 / theta;
                    if (acceptStep(log_prob_old, log_prob_new)) {
                        v0 = v0_new;
                        log_v0 = log_v0_new;
                    }
                }
            }
        }

        v0_new = v0 * exp(random_walk_stepsize * normal(generator));
        log_v0_new = log(v0_new);
        if (util::isPositive(v0_new)) {
            if (v0 == 0) {
                v0 = v0_new;
                log_v0 = log_v0_new;
            } else {
                log_prob_new = c0 * log_v0_new - c1 * log(v0_new + v1) - c2 * log(v0_new + v2);
                log_prob_old = c0 * log_v0 - c1 * log(v0 + v1) - c2 * log(v0 + v2);
                if (acceptStep(log_prob_old, log_prob_new)) {
                    v0 = v0_new;
                    log_v0 = log_v0_new;
                }
            }
        }

        return v0;
    }


};
