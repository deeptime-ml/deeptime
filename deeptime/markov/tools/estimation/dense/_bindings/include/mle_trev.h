//
// Created by mho on 7/29/20.
//

#pragma once

#include <cmath>
#include <memory>

#include "common.h"

template<typename dtype>
int mle_trev_dense(np_array<dtype> &T_arr, const np_array<dtype> &CCt_arr,
                   const np_array<dtype> &sum_C_arr, const std::size_t dim,
                   const dtype maxerr, const std::size_t maxiter,
                   np_array<dtype> &mu, dtype eps_mu) {
    py::gil_scoped_release gil;

    dtype rel_err, x_norm;
    auto sum_C = sum_C_arr.template unchecked<1>();
    auto T = T_arr.template mutable_unchecked<2>();
    auto CCt = CCt_arr.template unchecked<2>();

    std::unique_ptr<dtype[]> sum_x{new dtype[dim]};
    std::unique_ptr<dtype[]> sum_x_new{new dtype[dim]};

    /* ckeck sum_C */
    for (std::size_t i = 0; i < dim; i++) {
        if (sum_C[i] == 0) {
            throw std::invalid_argument("Some row and corresponding column of C have zero counts.");
        }
    }

    /* initialize sum_x_new */
    x_norm = 0;
    for (std::size_t i = 0; i < dim; i++) {
        sum_x_new[i] = 0;
        for (std::size_t j = 0; j < dim; j++) {
            sum_x_new[i] += CCt(i, j);
        }
        x_norm += sum_x_new[i];
    }
    for (std::size_t i = 0; i < dim; i++) {
        sum_x_new[i] /= x_norm;
    }

    /* iterate */
    std::size_t iteration{0};
    do {
        /* swap buffers */
        std::swap(sum_x, sum_x_new);

        x_norm = 0;
        for (std::size_t i = 0; i < dim; i++) {
            sum_x_new[i] = 0;
            for (std::size_t j = 0; j < dim; j++) {
                sum_x_new[i] += CCt(i, j) / (sum_C[i] / sum_x[i] + sum_C[j] / sum_x[j]);
            }
            if (sum_x_new[i] == 0 || std::isnan(sum_x_new[i])) {
                throw std::logic_error("The update of the stationary distribution produced zero or NaN.");
            }
            x_norm += sum_x_new[i];
        }

        /* normalize sum_x */
        for (std::size_t i = 0; i < dim; i++) {
            sum_x_new[i] /= x_norm;
            if (sum_x_new[i] <= eps_mu) {
                throw std::logic_error("Stationary distribution contains entries smaller "
                                       "than " + std::to_string(eps_mu) + "  during iteration.");
            }
        }

        iteration += 1;
        rel_err = util::relativeError(dim, sum_x.get(), sum_x_new.get());
    } while (rel_err > maxerr && iteration < maxiter);

    /* calculate T*/
    for (std::size_t i = 0; i < dim; i++) {
        sum_x[i] = 0;  // updated sum
        for (std::size_t j = 0; j < dim; j++) {
            T(i, j) = CCt(i, j) / (sum_C[i] / sum_x_new[i] + sum_C[j] / sum_x_new[j]);  // X_ij
            sum_x[i] += T(i, j);  // update sum with X_ij
        }
        /* normalize X to T*/
        for (std::size_t j = 0; j < dim; j++) {
            T(i, j) /= sum_x[i];
        }
    }

    std::copy(sum_x_new.get(), sum_x_new.get() + dim, mu.mutable_data());

    if (iteration == maxiter) {
        return -5;
    } else {
        return 0;
    }
}

template<typename dtype>
int mle_trev_given_pi_dense(np_array<dtype>& T_arr, const np_array<dtype> &C_arr, const np_array<dtype> &mu_arr,
                            const std::size_t n, const dtype maxerr, const std::size_t maxiter) {
    py::gil_scoped_release gil;

    auto T = T_arr.template mutable_unchecked<2>();
    auto C = C_arr.template unchecked<2>();
    auto mu = mu_arr.template unchecked<1>();

    std::unique_ptr<dtype[]> lam{new dtype[n]};
    std::unique_ptr<dtype[]> lam_new{new dtype[n]};

    /* check mu */
    for (std::size_t i = 0; i < n; i++) {
        if (mu[i] == 0) {
            throw std::logic_error("Some element of pi is zero.");
        }
    }

    /* initialise lambdas */
    for (std::size_t i = 0; i < n; i++) {
        lam_new[i] = 0.0;
        for (std::size_t j = 0; j < n; j++) {
            lam_new[i] += static_cast<dtype>(.5) * (C(i, j) + C(j, i));
        }
        if (lam_new[i] == 0) {
            throw std::logic_error("Some row and corresponding column of C have zero counts.");
        }
    }

    /* iterate lambdas */
    std::size_t iteration = 0;
    dtype d_sq {0};
    do {
        /* swap buffers */
        std::swap(lam, lam_new);

        auto lam_ptr = lam.get();
        auto lam_new_ptr = lam_new.get();

        #pragma omp parallel for default(none) firstprivate(C, lam_ptr, lam_new_ptr, n, mu)
        for (std::size_t j = 0; j < n; j++) {
            lam_new_ptr[j] = 0.0;
            for (std::size_t i = 0; i < n; i++) {
                auto C_ij = C(i, j) + C(j, i);
                if (C_ij != 0) {
                    lam_new_ptr[j] += C_ij / ((mu[j] * lam_ptr[i]) / (mu[i] * lam_ptr[j]) + 1);
                }
            }
            if (std::isnan(lam_new_ptr[j])) {
                throw std::logic_error("The update of the Lagrange multipliers produced NaN.");
            }
        }
        iteration += 1;
        d_sq = util::distsq(n, lam_ptr, lam_new_ptr);
    } while (d_sq > maxerr * maxerr && iteration < maxiter);

    /* calculate T */
    for (std::size_t i = 0; i < n; i++) {
        dtype norm = 0;
        for (std::size_t j = 0; j < n; j++) {
            auto C_ij = C(i, j) + C(j, i);
            if (i != j) {
                if (C_ij > 0.0) {
                    T(i, j) = C_ij / (lam_new[i] + lam_new[j] * mu[i] / mu[j]);
                    norm += T(i, j);
                } else {
                    T(i, j) = 0.0;
                }
            }
        }
        if (norm > 1.0) {
            T(i, i) = 0.0;
        } else {
            T(i, i) = 1.0 - norm;
        }
    }

    if (iteration == maxiter) {
        return -5;
    }
    return 0;
}
