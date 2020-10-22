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
    return x > 0 && !std::isnan(x) && !std::isinf(x);
}
}

template<typename dtype, typename Generator = std::mt19937>
class RevPiSampler {
public:
    explicit RevPiSampler(int seed) : uniform(0, 1) {
        if (seed < 0) {
            generator = deeptime::rnd::randomlySeededGenerator<Generator>();
        } else {
            generator = deeptime::rnd::seededGenerator<Generator>(static_cast<std::uint32_t>(seed));
        }
    }

    void update(const np_array<dtype> &arrC, np_array<dtype> &arrX, const np_array<dtype> &arrB) {
        auto M = arrC.shape(0);

        auto C = arrC.template unchecked<2>();
        auto b = arrB.template unchecked<1>();
        auto X = arrX.template mutable_unchecked<2>();

        for (ssize_t k = 0; k < M; ++k) {
            for (ssize_t l = 0; l < k; ++l) {
                if (C(k, l) + C(k, l) > 0) {
                    auto xkl = X(k, l);
                    auto xkl_new = sample_quad(X(k, l), X(k, k), X(l, l),
                                               C(k, l), C(l, k), C(k, k), C(l, l),
                                               b(k), b(l));
                    X(k, l) = xkl_new;
                    X(k, k) += (xkl - xkl_new);
                    X(l, k) = xkl_new;
                    X(l, l) += (xkl - xkl_new);

                    xkl = X(k, l);
                    xkl_new = sample_quad_rw(X(k, l), X(k, k), X(l, l),
                                             C(k, l), C(l, k), C(k, k), C(l, l),
                                             b(k), b(l));
                    X(k, l) = xkl_new;
                    X(k, k) += (xkl - xkl_new);
                    X(l, k) = xkl_new;
                    X(l, l) += (xkl - xkl_new);
                }
            }
        }
    }

private:
    Generator generator;
    std::gamma_distribution<dtype> gamma;
    std::uniform_real_distribution<dtype> uniform;
    std::normal_distribution<dtype> normal;

    dtype maximum_point(dtype s, dtype a1, dtype a2, dtype a3) const {
        dtype a = a2 + static_cast<dtype>(1);
        dtype b = a2 - a1 + (a2 + a3 + static_cast<dtype>(1)) / (s - static_cast<dtype>(1));
        dtype c = (a1 + static_cast<dtype>(1)) * s / (static_cast<dtype>(1) - s);
        dtype vbar = (-b + std::sqrt(b * b - static_cast<dtype>(4) * a * c)) / (static_cast<dtype>(2) * a);
        return vbar;
    }

    dtype f(dtype v, dtype s, dtype a1, dtype a2, dtype a3) const {
        dtype r = s / (s - static_cast<dtype>(1));
        return (a1 + static_cast<dtype>(1)) * std::log(v) + a3 * std::log(r + v)
                - (a1 + a2 + a3 + static_cast<dtype>(1)) * std::log(static_cast<dtype>(1) + v);
    }

    dtype F(dtype v, dtype s, dtype a1, dtype a2, dtype a3) const {
        dtype r = s / (s - static_cast<dtype>(1));
        return (a1 + static_cast<dtype>(1)) / v + a3 / (r + v)
                - (a1 + a2 + a3 + static_cast<dtype>(2)) / (static_cast<dtype>(1) + v);
    }

    dtype DF(dtype v, dtype s, dtype a1, dtype a2, dtype a3) const {
        dtype r = s / (s - static_cast<dtype>(1));
        return -(a1 + static_cast<dtype>(1)) / (v * v) - a3 / ((r + v) * (r + v))
               + (a1 + a2 + a3 + static_cast<dtype>(2)) / ((static_cast<dtype>(1) + v) * (static_cast<dtype>(1) + v));
    }

    dtype qacc(dtype w, dtype v, dtype s,
         dtype a1, dtype a2, dtype a3,
         dtype alpha, dtype beta) const {
        auto r = s / (s - static_cast<dtype>(1));
        return beta * (w - v) + (a1 + static_cast<dtype>(1) - alpha) * std::log(w / v) + a3 * std::log((r + w) / (r + v)) -
               (a1 + a2 + a3 + static_cast<dtype>(2)) * std::log((static_cast<dtype>(1) + w) / (static_cast<dtype>(1) + v));
    }

    dtype qacc_rw(dtype w, dtype v, dtype s, dtype a1, dtype a2, dtype a3) const {
        auto r = s / (s - static_cast<dtype>(1));
        return (a1 + static_cast<dtype>(1)) * std::log(w / v) + a3 * std::log((r + w) / (r + v))
               - (a1 + a2 + a3 + static_cast<dtype>(2)) * std::log((static_cast<dtype>(2) + w) / (static_cast<dtype>(1) + v));
    }

    dtype sample_quad(dtype xkl, dtype xkk, dtype xll,
                      dtype ckl, dtype clk, dtype ckk, dtype cll,
                      dtype bk, dtype bl) {
        dtype xlk, skl, slk, s2, s3, s, a1, a2, a3;
        dtype vbar, alpha, beta, v, w;
        dtype q, U;

        xlk = xkl;

        skl = xkk + xkl;
        slk = xll + xlk;

        if (skl <= slk) {
            s2 = skl;
            s3 = slk;
            s = s3 / s2;
            a1 = ckl + clk - static_cast<dtype>(1);
            a2 = ckk + bk - static_cast<dtype>(1);
            a3 = cll + bl - static_cast<dtype>(1);
        } else {
            s2 = slk;
            s3 = skl;
            s = s3 / s2;
            a1 = ckl + clk - static_cast<dtype>(1);
            a2 = cll + bl - static_cast<dtype>(1);
            a3 = ckk + bk - static_cast<dtype>(1);
        }

        //Check if s-1>0
        if (util::isPositive(s - static_cast<dtype>(1))) {
            vbar = maximum_point(s, a1, a2, a3);
            beta = -static_cast<dtype>(1) * DF(vbar, s, a1, a2, a3) * vbar;
            alpha = beta * vbar;

            //Check if s2-xkl > 0
            if (util::isPositive(s2 - xkl)) {
                //Old sample
                v = xkl / (s2 - xkl);

                //Check if alpha > 0 and 1/beta > 0
                auto betaInv = static_cast<dtype>(1) / beta;
                if (util::isPositive(alpha) && util::isPositive(betaInv)) {
                    //Proposal
                    gamma.param(typename decltype(gamma)::param_type {alpha, betaInv});
                    w = gamma(generator);

                    //If w=0 -> reject
                    if (util::isPositive(w)) {
                        // If v=0 accept
                        if (!util::isPositive(v)) {
                            return s2 * w / (static_cast<dtype>(1) + w);
                        } else {
                            // Log acceptance probability
                            q = qacc(w, v, s, a1, a2, a3, alpha, beta);

                            // Metropolis step
                            U = uniform(generator);
                            if (std::log(U) < std::min(static_cast<dtype>(0), q)) {
                                return s2 * w / (static_cast<dtype>(1) + w);
                            }
                        }
                    }
                }
            }
        }
        return xkl;
    }

    dtype sample_quad_rw(dtype xkl, dtype xkk, dtype xll,
                         dtype ckl, dtype clk, dtype ckk, dtype cll,
                         dtype bk, dtype bl) {
        dtype xlk, skl, slk, s2, s3, s, a1, a2, a3;
        dtype v, w;
        dtype q, U;

        xlk = xkl;

        skl = xkk + xkl;
        slk = xll + xlk;

        if (skl <= slk) {
            s2 = skl;
            s3 = slk;
            s = s3 / s2;
            a1 = ckl + clk - static_cast<dtype>(1);
            a2 = ckk + bk - static_cast<dtype>(1);
            a3 = cll + bl - static_cast<dtype>(1);
        } else {
            s2 = slk;
            s3 = skl;
            s = s3 / s2;
            a1 = ckl + clk - static_cast<dtype>(1);
            a2 = cll + bl - static_cast<dtype>(1);
            a3 = ckk + bk - static_cast<dtype>(1);
        }
        //Check if s2-xkl > 0
        if (util::isPositive(s2 - xkl)) {
            //Old sample
            v = xkl / (s2 - xkl);
            //Proposal
            w = v * std::exp(normal(generator));
            //If w=0 -> reject
            if (util::isPositive(w)) {
                //If v=0 accept
                if (!util::isPositive(v)) {
                    return s2 * w / (static_cast<dtype>(1) + w);
                } else {
                    q = qacc_rw(w, v, s, a1, a2, a3);
                    //Metropolis step
                    U = uniform(generator);
                    if (std::log(U) < std::min(static_cast<dtype>(0), q)) {
                        return s2 * w / (static_cast<dtype>(1) + w);
                    }
                }
            }
        }
        return xkl;
    }
};

template<typename dtype, typename Generator = std::mt19937>
class RevSampler {

public:
    explicit RevSampler(int seed) : uniform(0, 1) {
        if (seed < 0) {
            generator = deeptime::rnd::randomlySeededGenerator<Generator>();
        } else {
            generator = deeptime::rnd::seededGenerator<Generator>(static_cast<std::uint32_t>(seed));
        }
    }

    dtype updateStep(dtype v0, dtype v1, dtype v2, dtype c0, dtype c1, dtype c2, dtype random_walk_stepsize) {
        /*
        update the sample v0 according to
        the distribution v0^(c0-1)*(v0+v1)^(-c1)*(v0+v2)^(-c2)
        */
        dtype a = c1 + c2 - c0;
        dtype b = (c1 - c0) * v2 + (c2 - c0) * v1;
        dtype c = -c0 * v1 * v2;
        dtype v_bar = static_cast<dtype>(.5) * (-b + std::sqrt(b * b - static_cast<dtype>(4) * a * c)) / a;
        // dtype h = c1 + c2 - c0;
        dtype h = c1 / ((v_bar + v1) * (v_bar + v1))
                  + c2 / ((v_bar + v2) * (v_bar + v2))
                  - c0 / (v_bar * v_bar);
        dtype k = -h * v_bar * v_bar;
        dtype theta = -static_cast<dtype>(1) / (h * v_bar);
        dtype log_v0 = std::log(v0);
        dtype log_prob_old;
        dtype log_prob_new;

        // about 1.5 sec: gamma and normf generation
        // about 1 sec: logs+exps in else blocks

        if (util::isPositive(k) && util::isPositive(theta)) {
            gamma.param(typename decltype(gamma)::param_type {k, theta});
            auto v0_new = gamma(generator);
            auto log_v0_new = std::log(v0_new);
            if (util::isPositive(v0_new)) {
                if (v0 == 0) {
                    v0 = v0_new;
                    log_v0 = log_v0_new;
                } else {
                    log_prob_new = (c0 - static_cast<dtype>(1)) * log_v0_new - c1 * std::log(v0_new + v1) - c2 * std::log(v0_new + v2);
                    log_prob_new -= (k - static_cast<dtype>(1)) * log_v0_new - v0_new / theta;
                    log_prob_old = (c0 - static_cast<dtype>(1)) * log_v0 - c1 * std::log(v0 + v1) - c2 * std::log(v0 + v2);
                    log_prob_old -= (k - static_cast<dtype>(1)) * log_v0 - v0 / theta;
                    if (acceptStep(log_prob_old, log_prob_new)) {
                        v0 = v0_new;
                        log_v0 = log_v0_new;
                    }
                }
            }
        }

        auto v0_new = v0 * std::exp(static_cast<dtype>(random_walk_stepsize) * normal(generator));
        auto log_v0_new = std::log(v0_new);
        if (util::isPositive(v0_new)) {
            if (v0 == 0) {
                v0 = v0_new;
                log_v0 = log_v0_new;
            } else {
                log_prob_new = c0 * log_v0_new - c1 * std::log(v0_new + v1) - c2 * std::log(v0_new + v2);
                log_prob_old = c0 * log_v0 - c1 * std::log(v0 + v1) - c2 * std::log(v0 + v2);
                if (acceptStep(log_prob_old, log_prob_new)) {
                    v0 = v0_new;
                    log_v0 = log_v0_new;
                }
            }
        }

        return v0;
    }

    void update(const np_array<dtype> &arrC, const np_array<dtype> &arrSumC, np_array<dtype> &arrX,
                const np_array<int> &arrI, const np_array<int> &arrJ,
                int n_step) {
        dtype *X = arrX.mutable_data();
        auto nStates = arrC.shape(0);
        auto nIndices = arrI.shape(0);
        // std::vector<dtype> arrSumX = sumX(arrX, nStates);
        std::unique_ptr<dtype[]> sumX(new dtype[nStates]);
        std::fill(sumX.get(), sumX.get() + nStates, 0);

        // auto *sumX = arrSumX.data();
        auto C = arrC.template unchecked<2>();
        auto sumC = arrSumC.template unchecked<1>();

        const int *const I = arrI.data();
        const int *const J = arrJ.data();

        // row indexes
        std::unique_ptr<int[]> rowIndices(new int[nStates + 1]);
        std::fill(rowIndices.get(), rowIndices.get() + nStates + 1, 0);
        generateRowIndices(I, nStates, nIndices, rowIndices.get());

        for (int iter = 0; iter < n_step; iter++) {
            // update all X row sums once every iteration and then only do cheap updates.
            for (int i = 0; i < nStates; i++) {
                sumX[i] = _sumRowSparse(X, nStates, i, J, rowIndices[i], rowIndices[i + 1]);
            }

            for (int k = 0; k < nIndices; k++) {
                auto i = I[k];
                auto j = J[k];
                if (i == j) {
                    if (util::isPositive(C(i, i)) && util::isPositive(sumC(i) - C(i, i))) {
                        beta.param(typename decltype(beta)::param_type {C(i, i), sumC(i) - C(i, i)});
                        auto tmp1 = beta(generator);
                        auto tmp2 = tmp1 / (static_cast<dtype>(1) - tmp1) * (sumX[i] - X[i * nStates + i]);
                        if (util::isPositive(tmp2)) {
                            sumX[i] += tmp2 - X[i * nStates + i];  // update sumX
                            X[i * nStates + i] = tmp2;
                        }
                    }
                } else if (i < j) {  // only work on the upper triangle, because we have symmetry.
                    auto tmp1 = sumX[i] - X[i * nStates + j];
                    auto tmp2 = sumX[j] - X[j * nStates + i];
                    X[i * nStates + j] = updateStep(X[i * nStates + j], tmp1, tmp2, C(i, j) + C(j, i),
                                                    sumC(i), sumC(j), 1);
                    X[j * nStates + i] = X[i * nStates + j];
                    // update X
                    sumX[i] = tmp1 + X[i * nStates + j];
                    sumX[j] = tmp2 + X[j * nStates + i];
                }
            }

            _normalizeAllSparse(X, I, J, nStates, nIndices);
        }
    }

private:
    Generator generator;
    std::normal_distribution<dtype> normal;  // standard normal by default ctor
    std::gamma_distribution<dtype> gamma;
    deeptime::rnd::beta_distribution<dtype> beta;
    std::uniform_real_distribution<dtype> uniform;

    bool acceptStep(dtype log_prob_old, dtype log_prob_new) {
        auto diff = log_prob_new - log_prob_old;
        return diff > 0 /* this is faster */ ||
               uniform(generator) < std::exp(std::min(diff, static_cast<dtype>(0)));
    }

    void generateRowIndices(const int *const I, int n, int n_idx, int *rowIndices) {
        rowIndices[0] = 0;  // starts with row 0
        int current_row = 0;
        for (int k = 0; k < n_idx; k++) {
            // still at same row? do nothing
            if (I[k] == current_row)
                continue;
            // row has advanced one or multiple times. Update multiple row indexes until we are equal
            while (I[k] > current_row) {
                current_row++;
                rowIndices[current_row] = k;
            }
        }
        // stop sign
        rowIndices[n] = n_idx;
    }

    dtype _sumRowSparse(const dtype *const X, int n, int i, const int *const J, int from, int to) const {
        auto sum = static_cast<dtype>(0);
        for (int j = from; j < to; j++) {
            sum += X[i * n + J[j]];
        }
        return sum;
    }

    void _normalizeAllSparse(dtype *X, const int *const I, const int *const J, int n, int n_idx) {
        // sum all
        auto sum = static_cast<dtype>(0);
        for (int k = 0; k < n_idx; k++) {
            sum += X[I[k] * n + J[k]];
        }
        // normalize all
        for (int k = 0; k < n_idx; k++) {
            X[I[k] * n + J[k]] /= sum;
        }
    }


};
